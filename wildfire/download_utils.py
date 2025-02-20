from dataclasses import dataclass
import os
from typing import Optional, TypedDict
import torch
from torch import Tensor
import earthaccess
import numpy as np
import netCDF4
from datetime import datetime, timedelta
from wildfire.data_utils import *
from wildfire.data_types import *
from wildfire.fires import fires as fires_cpp

import pyhdf.SD as SD


def nc_download(
    short_names,
    out_dir,
    temporal,
    bounding_box,
    min_types=set(),
    num_retries=10,
) -> dict[str, dict[str, str]]:
    """Download nc files from earthaccess
    Will retry n times until all queries were successfully completed.
    Returns a {<timestamp>: {<short_name>: <file_path>}}
    """
    min_types = set(min_types)

    def get_items():
        all_items = []
        items_by_date_sn = {}
        for short_name in short_names:
            print("Finding items for:", short_name)
            items = earthaccess.search_data(
                short_name=short_name,
                bounding_box=bounding_box,
                temporal=temporal,
                count=-1,
            )
            all_items.extend(items)
            for item in items:
                date_str = item["umm"]["TemporalExtent"]["RangeDateTime"][
                    "BeginningDateTime"
                ]
                date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
                if config.is_modis:
                    # TODO: Think about best way. Delay or rewind?
                    # - -> We looked into the future
                    # + -> We received it delayed
                    date = date + timedelta(hours=4)
                    # -> Modis night for same date will be a bit before VIIRS
                if date not in items_by_date_sn:
                    items_by_date_sn[date] = {}
                # INFO: Just change used date, not filename
                basename = os.path.basename(item.data_links()[0])
                items_by_date_sn[date][short_name] = os.path.join(
                    out_dir, basename
                )
        items_by_date_sn = dict(sorted(items_by_date_sn.items()))
        filtered_by_date = dict()
        all_correct_length = True
        for date, paths in items_by_date_sn.items():
            if (
                len(paths) != len(short_names)
                and set(paths.keys()) != min_types
            ):
                print(
                    f"Expected {len(short_names)} items for {date}, got {paths.keys()}"
                )
                all_correct_length = False
            else:
                filtered_by_date[date] = paths
        # Can keep items where not all exists, as long as we dont have them in
        # the dict
        return filtered_by_date, all_items, all_correct_length

    num_query_retries = 1
    for i in range(num_query_retries + 1):
        items_by_date_sn, all_items, ok = get_items()
        if ok:
            break
        else:

            print(
                f"get_items(), all short_names not in all dates. Retrying ({i+1}/{num_query_retries})"
            )

    earthaccess.login()

    def all_exists():
        print("Checking if all files exist")
        for date, items in items_by_date_sn.items():
            if len(items) != len(short_names) and len(min_types) == len(
                short_names
            ):
                raise Exception(
                    f"Expected {len(short_names)} items for {date}, got {len(items)}"
                )
            for item in items.values():
                if not os.path.exists(item):
                    return False

        return True

    for i in range(num_retries + 1):
        try:
            _ = earthaccess.download(all_items, out_dir)
            if not all_exists():
                print(
                    f"Failed to download all items, retrying ({i}/{num_retries})"
                )
                continue
            print("All files downloaded")
            break
        except Exception as e:
            print(
                f"Error downloading files: {e}, retrying ({i+1}/{num_retries})"
            )
            continue

    else:
        raise Exception("Reached max retries, failed to download all files")
    return items_by_date_sn


class GridDay:

    def __init__(self):
        self.cur_date: datetime = None

    def should_commit(self, date: datetime):
        """
        Returns True if day and year is different from the current date
        """
        if self.cur_date is None:
            return True
        return date.strftime("%Y-%m-%d") != self.cur_date.strftime("%Y-%m-%d")

    def commit(self, h5_grid: H5Grid):
        if self.cur_date is None:
            return
        print(f"Committing {self.cur_date}")
        h5_grid.add_for_day(self.grid, self.cur_date)

    def reset(self, date: datetime):
        self.cur_date = date
        self.is_night = False
        self.grid: dict[tuple[int, int], FullDay] = {}

    def set_is_night(self, is_night: bool):
        self.is_night = is_night

    @property
    def night_key(self):
        return "night" if self.is_night else "day"

    def get_grid_keys(
        self, res: "FireMaskExtractor.Result", h5_grid: H5Grid, patch_size: int
    ) -> list[int]:

        grid_xs, grid_ys = grid_get_dimensions(res.bounds)
        grid_y, grid_x = np.meshgrid(
            np.arange(grid_ys),
            np.arange(grid_xs),
            indexing="ij",
        )
        grid_y, grid_x = grid_y.flatten(), grid_x.flatten()
        x_offset, y_offset = grid_get_2d_offset(h5_grid.bounds, res.bounds)
        # print(f"Global: {h5_grid.bounds}")
        # print(f"Local: {res.bounds} {x_offset=} {y_offset=}")
        # NOTE:
        # - grid_y is a vector of idxs
        # - Clipped to global grid, so will get (grid_ys, grid_xs) dims
        # - If min=global_max, firemask would have returned None (zero pixels)
        inside_mask = h5_grid.cells_in_aoi_mask[
            grid_y + y_offset, grid_x + x_offset
        ]
        grid_y = grid_y[inside_mask]
        grid_x = grid_x[inside_mask]

        valid_keys_xy = []
        for gx, gy in zip(grid_x, grid_y):
            n = patch_size

            # Img coords, top down
            gpy = (grid_ys - 1) - gy
            has_data = res.has_data[
                gpy * (n) : (gpy + 1) * n, gx * (n) : (gx + 1) * n
            ]
            if has_data.size == 0:
                continue
            ratio = has_data.sum() / has_data.size
            cell_key = int(gx + x_offset), int(gy + y_offset)
            cell = self.get_cell(cell_key)
            cmp_ratio = cell["ratio_data"]
            if ratio > cmp_ratio:
                valid_keys_xy.append((gx, gy))
                cell["ratio_data"] = ratio
        return valid_keys_xy, x_offset, y_offset

    def add_raster(
        self,
        raster_key,
        data,
        bounds: Bounds,
        local_keys_xy,
        grid_x_offset,
        grid_y_offset,
        patch_size: int,
    ):
        grid_xs, grid_ys = grid_get_dimensions(bounds)
        for gx, gy in local_keys_xy:
            n = patch_size
            # Img coords, top down
            gpy = (grid_ys - 1) - gy
            crop = data[:, gpy * n : (gpy + 1) * n, gx * n : (gx + 1) * n]
            cell_key = gx + grid_x_offset, gy + grid_y_offset
            cell = self.get_cell(cell_key)
            cell[raster_key] = crop
            if raster_key == "fire_mask":
                num_fire = crop > (FireCls.LOW + 0.5)
                cell["num_fire_pixels"] = num_fire.sum().item()

    def get_cell(self, cell_key):
        if cell_key not in self.grid:
            self.grid[cell_key] = {
                "night": {"ratio_data": 0, "num_fire_pixels": 0},
                "day": {"ratio_data": 0, "num_fire_pixels": 0},
            }
        return self.grid[cell_key][self.night_key]


class Extractor:
    resolution = Resolution.HIGH
    H: int
    W: int
    bounds: Bounds
    warp_pixel_idx: Tensor
    inside_bounds_mask: Tensor
    lon: np.ndarray
    lat: np.ndarray

    def __init__(
        self, img_ds_path, geo_ds_path, resolution: float, mpp: int = None
    ):
        print(f"Loading {img_ds_path}")
        if config.is_modis:
            self.img_ds = SD.SD(img_ds_path)
            self.geo_ds = SD.SD(geo_ds_path)
        else:
            self.img_ds = netCDF4.Dataset(img_ds_path, "r")
            self.geo_ds = netCDF4.Dataset(geo_ds_path, "r")
        self.resolution = resolution
        self.mpp = mpp

    def __del__(self):
        try:
            if config.is_modis:
                self.img_ds.end()
                self.geo_ds.end()
            else:
                self.img_ds.close()
                self.geo_ds.close()
        except Exception as e:
            pass

    def set_lon_lat(self, geo_ds):
        if config.is_modis:
            lat = np.array(geo_ds.select("Latitude").get())
            lon = np.array(geo_ds.select("Longitude").get())
            if self.mpp < 1000:
                lat = torch.nn.functional.interpolate(
                    torch.tensor(lat).unsqueeze(0).unsqueeze(0),
                    size=self.img_size,
                    mode="bilinear",
                ).numpy()[0, 0]
                lon = torch.nn.functional.interpolate(
                    torch.tensor(lon).unsqueeze(0).unsqueeze(0),
                    size=self.img_size,
                    mode="bilinear",
                ).numpy()[0, 0]
        else:
            geo_data = geo_ds["geolocation_data"]
            lat = np.array(geo_data["latitude"])
            lon = np.array(geo_data["longitude"])
        lon[lon < 0] += 360
        self.lat = lat
        self.lon = lon

    def prepare_warp(self, lat, lon):
        min_lon, min_lat, max_lon, max_lat = self.bounds
        self.H = round((max_lat - min_lat) / self.resolution)
        self.W = round((max_lon - min_lon) / self.resolution)

        lat_idx = grid_discretize_axis(lat, min_lat, max_lat, self.H)
        lat_idx = (self.H - 1) - lat_idx
        lon_idx = grid_discretize_axis(lon, min_lon, max_lon, self.W)

        # Create masks and pixel indices
        inside_bounds_mask = (
            (lat_idx >= 0)
            & (lon_idx >= 0)
            & (lat_idx <= self.H - 1)
            & (lon_idx <= self.W - 1)
        )

        self.warp_pixel_idx = lat_idx * self.W + lon_idx
        self.inside_bounds_mask = inside_bounds_mask

    def get_projection_coords(self, lat, lon):
        min_lon, min_lat, max_lon, max_lat = self.bounds
        self.H = round((max_lat - min_lat) / self.resolution)
        self.W = round((max_lon - min_lon) / self.resolution)

        def axis_coords(coords, min_val, max_val, size):
            coords = torch.tensor(coords, dtype=torch.float32)
            axis_norm = (coords - min_val) / (max_val - min_val)
            axis_idx = axis_norm * size
            return axis_idx

        y_idx = axis_coords(lat, min_lat, max_lat, self.H)
        # Why this?
        y_idx = (self.H - 1) - y_idx
        x_idx = axis_coords(lon, min_lon, max_lon, self.W)
        xy_idx = torch.stack([x_idx, y_idx], dim=2)
        return xy_idx


class FireMaskExtractor(Extractor):

    @dataclass
    class Result:
        filled: np.ndarray
        bounds: Bounds
        has_data: np.ndarray
        is_night: bool

    def _process(self) -> Optional["FireMaskExtractor.Result"]:
        self.set_lon_lat(self.geo_ds)
        self.bounds = grid_clipped_bounds(
            self.lon.min(), self.lat.min(), self.lon.max(), self.lat.max()
        )
        self.bounds = grid_clip_bounds_to_global(self.bounds)
        xy_idx = self.get_projection_coords(self.lat, self.lon)
        if self.H <= 0 or self.W <= 0:
            print("Out of bounds")
            print(
                f"Org bounds: {self.lon.min()=} {self.lat.min()=} {self.lon.max()=} {self.lat.max()=}"
            )
            return None
        # ======== Proj ========
        # Unique value since fire_cls 1-9
        if config.is_modis:
            fire_cls = np.array(self.img_ds.select("fire mask").get())
        else:
            fire_cls = np.array(self.img_ds["fire mask"])
        # Ideally:
        # - fill_data: 0 where in FIRE_CLS.NO_DATA (do we want CLOUD as separate class?)
        # - has_data: calculated before fill_no_data
        no_data_val = FireCls.NO_DATA  # Not here before warp
        pre_warp = fire_cls.astype(np.float32)
        pre_warp = torch.tensor(pre_warp).unsqueeze(0)
        kernel_size = 3
        fill_data: Tensor = fires_cpp.project(
            pre_warp,
            self.W,
            self.H,
            xy_idx,
            "nearest",
            kernel_size,
            no_data_val,
        )
        fill_data: np.ndarray = fill_data.numpy()
        fill_data = (np.round(fill_data)).astype(np.uint8)
        # Should this be based on 1) mpp (cur) or 2) data avail (old)
        has_data = (fill_data != no_data_val).reshape(self.H, self.W)
        for val in FireCls.no_data_values():
            if val != FireCls.CLOUD:
                fill_data[fill_data == val] = no_data_val
        if config.is_modis:
            attr = self.img_ds.attributes()
            is_night = attr["NightPix"] > attr["DayPix"]
        else:
            is_night = self.img_ds.DayNightFlag != "Day"

        result = FireMaskExtractor.Result(
            fill_data, self.bounds, has_data, is_night
        )
        return result

    @staticmethod
    def process(*args, **kwargs) -> "FireMaskExtractor.Result":
        extractor = FireMaskExtractor(*args, **kwargs)
        return extractor._process()


class RawExtractor(Extractor):

    def __init__(
        self,
        img_ds_path,
        geo_ds_path,
        bounds: Bounds,
        is_night: bool,
        resolution: float,
        mpp: int = None,
    ):
        super().__init__(img_ds_path, geo_ds_path, resolution, mpp=mpp)
        self.resolution = resolution
        self.is_night = is_night
        self.bounds = bounds
        self.kernel_size = 3
        if self.mpp == 1000:
            self.kernel_size = 5

    def process_viirs(self) -> np.ndarray:

        self.set_lon_lat(self.geo_ds)
        observations = self.img_ds["observation_data"]
        band_names = get_band_names(self.resolution, self.is_night)
        no_data_val = config.uint16_no_data
        pre_warp = []
        for band_name in band_names:
            var = observations[band_name]
            data = np.array(var).astype(np.float32)
            has_data_mask = data <= var.valid_max
            data = (data - var.add_offset) / var.scale_factor
            data[~has_data_mask] = no_data_val
            pre_warp.append(data)
        pre_warp = np.stack(pre_warp)
        pre_warp = torch.tensor(pre_warp, dtype=torch.float32)
        return self.process_final(pre_warp)

    def process_modis(self):
        if self.mpp == 1000:
            emi = self.img_ds.select("EV_1KM_Emissive").get()
            if not self.is_night:
                refl = self.img_ds.select("EV_1KM_RefSB").get()
                pre_warp = np.concatenate([emi, refl], axis=0)
            else:
                pre_warp = emi
        elif self.mpp == 500:
            pre_warp = self.img_ds.select("EV_500_RefSB").get()
        elif self.mpp == 250:
            pre_warp = self.img_ds.select("EV_250_RefSB").get()
        else:
            raise ValueError(f"Unsupported mpp for modis: {self.mpp}")
        self.img_size = pre_warp.shape[1:]
        self.set_lon_lat(self.geo_ds)
        pre_warp = torch.tensor(pre_warp, dtype=torch.float32)
        has_data_mask = pre_warp < (32767 + 10)
        pre_warp[~has_data_mask] = config.uint16_no_data
        return self.process_final(pre_warp)

    def process_final(self, data):
        xy_idx = self.get_projection_coords(self.lat, self.lon)
        fill_data = fires_cpp.project(
            data,
            self.W,
            self.H,
            xy_idx,
            "inv_dist",
            self.kernel_size,
            config.uint16_no_data,
        )
        fill_data = fill_data.numpy()
        fill_data = (np.round(fill_data)).astype(np.uint16)
        return fill_data

    @staticmethod
    def process(*args, **kwargs) -> np.ndarray:
        extractor = RawExtractor(*args, **kwargs)
        if config.is_modis:
            return extractor.process_modis()
        else:
            return extractor.process_viirs()
