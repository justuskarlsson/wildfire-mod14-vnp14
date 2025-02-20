from dataclasses import dataclass
from datetime import datetime
from glob import glob
import os
import time
from typing import TypedDict
from wildfire.data_types import *
import math
from osgeo import gdal
import torch
from torch import Tensor
import numpy as np
import json
import h5py

gdal.UseExceptions()


def json_dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def json_load(path):
    with open(path, "r") as f:
        return json.load(f)


def raster_aus_mask(geojson_path, bounds, xres, yres):
    """Load the geojson shape of australia.
    Rasterize it to a potentially different bound (than the mask) with a given
    resolution. Good for masking out data not part of mainland australia."""
    shp_ds = gdal.OpenEx(geojson_path)
    opt = gdal.RasterizeOptions(
        outputBounds=bounds,  # [minx, miny, maxx, maxy]
        xRes=xres,
        yRes=yres,
        allTouched=True,
        outputType=gdal.GDT_Byte,
        burnValues=1,
    )
    ds: gdal.Dataset = gdal.Rasterize(
        "/vsimem/tmpfile.tif", shp_ds, options=opt
    )
    mask = ds.ReadAsArray()
    ds = None
    gdal.Unlink("/vsimem/tmpfile.tif")
    mask = mask > 0
    mask = np.flip(mask, 0)
    return mask


# ===========================
# ======== GRID UTILS
def grid_get_dimensions(bounds: Bounds):
    """
    Returns (xs, ys)
    """
    cell_size = config.cell_size
    xs = round((bounds[2] - bounds[0]) / cell_size)
    ys = round((bounds[3] - bounds[1]) / cell_size)
    return xs, ys


def grid_get_2d_offset(global_bounds: Bounds, local_bounds: Bounds):
    """
    Returns (x_offset, y_offset)
    """
    cell_size = config.cell_size
    x_offset = round((local_bounds[0] - global_bounds[0]) / cell_size)
    y_offset = round((local_bounds[1] - global_bounds[1]) / cell_size)
    return x_offset, y_offset


def grid_clipped_bounds(min_lon, min_lat, max_lon, max_lat) -> Bounds:
    cell_size = config.cell_size
    min_lat = math.ceil(min_lat / cell_size) * cell_size
    max_lat = math.floor(max_lat / cell_size) * cell_size
    min_lon = math.ceil(min_lon / cell_size) * cell_size
    max_lon = math.floor(max_lon / cell_size) * cell_size
    return (min_lon, min_lat, max_lon, max_lat)


def grid_expanded_bounds(min_lon, min_lat, max_lon, max_lat) -> Bounds:
    cell_size = config.cell_size
    min_lat = math.floor(min_lat / cell_size) * cell_size
    max_lat = math.ceil(max_lat / cell_size) * cell_size
    min_lon = math.floor(min_lon / cell_size) * cell_size
    max_lon = math.ceil(max_lon / cell_size) * cell_size
    return (min_lon, min_lat, max_lon, max_lat)


def grid_clip_bounds_to_global(bounds: Bounds) -> Bounds:
    global_bounds = grid_expanded_bounds(*config.aoi)
    return (
        max(bounds[0], global_bounds[0]),
        max(bounds[1], global_bounds[1]),
        min(bounds[2], global_bounds[2]),
        min(bounds[3], global_bounds[3]),
    )


def grid_discretize_axis(coords, min_val, max_val, size) -> torch.Tensor:
    """
    Returns 1D grid cell (idx).
    """
    coords = torch.tensor(coords)
    axis_norm = (coords - min_val) / (max_val - min_val)
    axis_idx = axis_norm * size
    # Center the index
    axis_idx = torch.round(axis_idx).long()
    return axis_idx.flatten()


def h5_get_nested(h5_group: h5py.Group, path: list[str], write: bool = False):
    group = h5_group
    for p in path:
        p = str(p)
        if p not in group:
            if not write:
                return None
            group = group.create_group(p)
        else:
            group = group[p]
    return group


def h5_get_path(sensor: str, test: bool):
    prefix = "test/" if test else ""
    prefix += sensor
    pattern = os.path.join(config.h5_dir, f"{prefix}*.h5")
    matches = glob(pattern)
    if len(matches) > 1:
        raise Exception(
            f"Found {len(matches)} files in {prefix}. Please clear."
        )
    return matches[0]


def h5_replace_sensor(h5_path: str, new_sensor: str):
    h5_dir = os.path.dirname(h5_path)
    h5_name = os.path.basename(h5_path)
    sensor = h5_name.split("_")[0]
    h5_name = h5_name.replace(sensor, new_sensor)
    return os.path.join(h5_dir, h5_name)


@dataclass
class H5Grid:
    h5: h5py.File
    bounds: Bounds
    xs: int
    ys: int
    cells_in_aoi_mask: np.ndarray

    def __init__(self, h5_path: str, write: bool = False):
        mode = "r"
        if write:
            mode = "w"
        self.h5 = h5py.File(h5_path, mode)
        self.bounds = grid_expanded_bounds(*config.aoi)
        self.xs, self.ys = grid_get_dimensions(self.bounds)
        if write:
            self.h5.create_group("num_fire_pixels_by_day")
            self.h5.create_group("cells")
            self.cells_in_aoi_mask = raster_aus_mask(
                "australia.geojson",
                self.bounds,
                config.cell_size,
                config.cell_size,
            )

    @staticmethod
    def get_cells():
        """Return iterable[(x, y, lon, lat)]"""
        bounds = grid_expanded_bounds(*config.aoi)
        min_lon, min_lat, *_ = bounds
        xs, ys = grid_get_dimensions(bounds)
        for x in range(xs):
            for y in range(ys):
                yield (
                    x,
                    y,
                    min_lon + x * config.cell_size,
                    min_lat + y * config.cell_size,
                )

    def __del__(self):
        try:
            self.h5.close()
        except Exception as e:
            print(f"Error closing h5: {e}")

    # ===========================
    # ======== Writing
    def add_for_day(
        self, values: dict[tuple[float, float], FullDay], date: datetime
    ):
        date_path = self.get_date_path(date)
        num_fire_pixels_by_cell = np.zeros((self.ys, self.xs), dtype=np.int32)
        h5_cells = self.h5["cells"]
        comp = dict(compression="gzip", compression_opts=3)
        for (gx, gy), cell_day in values.items():
            cell_path = self.get_cell_path(gx, gy)
            if cell_path not in h5_cells:
                h5_cells.create_group(cell_path)
            cell = h5_cells[cell_path]
            cell.create_group(date_path)
            date_group = cell[date_path]
            # Insert FullDay into date_group
            for period_key, half_day in cell_day.items():  # 'day' or 'night'
                half_day: HalfDay
                period_group = date_group.create_group(period_key)
                num_fire_pixels_by_cell[gy, gx] = half_day["num_fire_pixels"]
                for data_key, data_value in half_day.items():
                    if isinstance(data_value, np.ndarray):
                        # Create a dataset for numpy arrays
                        period_group.create_dataset(
                            data_key, data=data_value, **comp
                        )
                    else:
                        # Store scalar floats as attributes
                        period_group.attrs[data_key] = data_value
        self.h5["num_fire_pixels_by_day"].create_dataset(
            date_path, data=num_fire_pixels_by_cell, **comp
        )

    def get_cell(self, x, y, date: datetime) -> h5py.Group | None:
        date_path = self.get_date_path(date)
        cell_path = self.get_cell_path(x, y)
        cells = self.h5["cells"]
        if cell_path not in cells:
            return None
        cell = cells[cell_path]
        if date_path not in cell:
            return None
        return cell[date_path]

    @staticmethod
    def get_cell_path(x, y):
        return f"cell_{x:03d}_{y:03d}"

    @staticmethod
    def get_date_path(date: datetime):
        return date.strftime("%Y-%m-%d")

    def get_local_grid(self, clipped_bounds):
        grid_discretize_axis(
            clipped_bounds,
            clipped_bounds[0],
            clipped_bounds[2],
            config.cell_size,
        )


def get_h5_paths_by_year(h5_paths: list[str]):
    by_year = {}
    for path in h5_paths:
        # 2019-10-15_2019-10-31.h5
        name = os.path.basename(path)
        year = int(name[:4])
        month = int(name[5:7])
        if month <= 6:
            year -= 1
        by_year.setdefault(year, []).append(path)
    return by_year


def raster_fill_no_data(
    bands: np.ndarray, no_data_val, dtype=np.float32
) -> np.ndarray:
    """ """

    if len(bands.shape) == 2:
        raise Exception("Expects N Y X dims, even if N=1")
    is_uint8 = dtype == np.uint8
    driver = gdal.GetDriverByName("GTiff")
    tmp_file = "/vsimem/tmp.tif"
    ds: gdal.Dataset = driver.Create(
        tmp_file,
        bands.shape[2],
        bands.shape[1],
        bands.shape[0],
        gdal.GDT_Byte if is_uint8 else gdal.GDT_Float32,
    )
    filled = [None] * len(bands)
    interp_mode = "nearest" if is_uint8 else "inv_dist"
    max_search_dist = 2 if is_uint8 else 10
    for i in range(len(bands)):
        band: gdal.Band = ds.GetRasterBand(i + 1)
        band.WriteArray(bands[i])
        band.SetNoDataValue(no_data_val)
        band.FlushCache()
        gdal.FillNodata(
            targetBand=band,
            maskBand=None,
            maxSearchDist=max_search_dist,
            smoothingIterations=1,
            options=[f"INTERPOLATION={interp_mode}"],
        )
        band.FlushCache()
        filled[i] = band.ReadAsArray()
    gdal.Unlink(tmp_file)
    return np.stack(filled)


class Timer:
    def __init__(self, title):
        self.start = time.time()
        self.title = title

    def finish(self):
        end = time.time()
        seconds = end - self.start
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        time_str = f"{f'{hours}h ' if hours > 0 else ''}{f'{minutes}m ' if minutes > 0 else ''}{f'{seconds:.1f}s'}"
        print(f"{self.title} finished in {time_str}")
