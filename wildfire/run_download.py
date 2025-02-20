from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable
import earthaccess
import fire


from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from glob import glob
import os
import time
import ee
import requests
import urllib3
from osgeo import gdal, osr
import h5py
import numpy as np
from wildfire.data_types import config, era5_bands
from wildfire.data_utils import (
    H5Grid,
    h5_get_nested,
    h5_replace_sensor,
)


def map_parallell(
    fn: Callable,
    args_list: list[list[Any]],
    max_workers=20,
):
    with ThreadPoolExecutor(
        max_workers=max_workers,
    ) as executor:
        futures = []
        for args in args_list:
            future = executor.submit(fn, *args)
            futures.append(future)
        results = []
        for future in futures:
            result = future.result()
            results.append(result)
        return results


def download_raw(url, dst):
    r = requests.get(url, stream=True, verify=False)
    if r.status_code == 200:
        with open(f"{dst}", "wb") as f:
            f.write(r.content)
    else:
        raise Exception(f"Error downloading {dst} {r.content} ")


def download_factory(prefix_path: str, scale: int, include_hour: bool = False):
    date_to_str = "YYYY_MM_dd"
    date_from_str = "%Y_%m_%d"
    if include_hour:
        date_to_str = "YYYY_MM_dd_HH"
        date_from_str = "%Y_%m_%d_%H"

    def download(img: ee.Image, rect: ee.Geometry.Rectangle, x, y):
        date_str = (
            ee.Date(img.get("system:time_start")).format(date_to_str).getInfo()
        )
        dst_path = prefix_path + f"{date_str}_{x}_{y}.tif"
        img = img.clip(rect)
        if os.path.exists(dst_path):
            print(f"Skipping {dst_path} because it already exists")
        else:
            url = img.getDownloadUrl(
                {
                    "name": dst_path,
                    "crs": "EPSG:4326",
                    "scale": scale,
                    "format": "GeoTIFF",
                    "bands": img.bandNames().getInfo(),
                    "region": rect,
                }
            )
            download_raw(url, dst_path)
            print(f"Downloaded {dst_path}")
        return dst_path, datetime.strptime(date_str, date_from_str), x, y

    return download


def get_dates_exclusive(h5_proto: str):
    with h5py.File(h5_proto, "r") as h5:
        dates = list(h5["num_fire_pixels_by_day"].keys())
    start_date, end_date = dates[0], dates[-1]
    end_date = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    end_date = end_date.strftime("%Y-%m-%d")
    return start_date, end_date


def era5_download(
    h5_proto: str,
):
    urllib3.disable_warnings()
    ee.Authenticate()
    ee.Initialize(project="ee-karlssonjustus")

    aoi = config.aoi
    rect = ee.Geometry.Rectangle(aoi)
    collection = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")

    h5_path = h5_replace_sensor(h5_proto, "era5")
    h5 = h5py.File(h5_path, "w")
    start_date, end_date = get_dates_exclusive(h5_proto)
    collection = collection.filterDate(start_date, end_date)
    collection = collection.filterBounds(rect)
    collection = collection.select(era5_bands)
    image_list = collection.toList(collection.size())
    images = [
        ee.Image(image_list.get(i)) for i in range(collection.size().getInfo())
    ]
    ee_path = os.path.join(config.root_path, "ee")
    dir_path = os.path.join(ee_path, f"era5")
    os.makedirs(dir_path, exist_ok=True)
    prefix_path = os.path.join(dir_path, f"era5_")
    download = download_factory(prefix_path, scale=11132)
    args = []
    for img in images:
        args.append((img, rect, 0, 0))
    results = map_parallell(download, args)
    results.sort()
    for tif_path, date, x, y in results:
        ds: gdal.Dataset = gdal.Open(tif_path)
        sr: osr.SpatialReference = ds.GetSpatialRef()
        for x, y, lon, lat in H5Grid.get_cells():

            min_lon, min_lat = lon, lat
            max_lon, max_lat = lon + config.cell_size, lat + config.cell_size
            xRes = 0.05
            yRes = -0.05
            options = gdal.WarpOptions(
                format="GTiff",
                outputBounds=[
                    min_lon,
                    min_lat,
                    max_lon,
                    max_lat,
                ],  # [ulx, uly, lrx, lry]
                xRes=xRes,
                yRes=yRes,
                resampleAlg="bilinear",
                srcSRS=sr,
                dstSRS="EPSG:4326",
            )
            ds_crop: gdal.Dataset = gdal.Warp(
                "/vsimem/crop.tif", ds, options=options
            )
            patch = ds_crop.ReadAsArray()
            group = h5_get_nested(
                h5,
                [
                    "cells",
                    H5Grid.get_cell_path(x, y),
                ],
                write=True,
            )
            group.create_dataset(H5Grid.get_date_path(date), data=patch)
        print(f"Inserted {tif_path}")
    print(f"Wrote to {h5_path}")
    h5.close()


def drought_download(
    h5_proto: str,
):

    urllib3.disable_warnings()
    ee.Authenticate()
    ee.Initialize(project="ee-karlssonjustus")
    # list[4] -> rect

    aoi = config.aoi
    rect = ee.Geometry.Rectangle(aoi)
    h5_path = h5_replace_sensor(h5_proto, "drought")
    h5 = h5py.File(h5_path, "w")
    start_date, end_date = get_dates_exclusive(h5_proto)

    collection = ee.ImageCollection("UTOKYO/WTLAB/KBDI/v1")
    collection = collection.filterDate(start_date, end_date)
    collection = collection.filterBounds(rect)
    image_list = collection.toList(collection.size())
    images = [
        ee.Image(image_list.get(i)) for i in range(collection.size().getInfo())
    ]
    ee_path = os.path.join(config.root_path, "ee")
    dir_path = os.path.join(ee_path, f"drought_kbdi")
    os.makedirs(dir_path, exist_ok=True)
    prefix_path = os.path.join(dir_path, f"drought_kbdi")
    download = download_factory(prefix_path, scale=4000)
    args = []
    for img in images:
        args.append((img, rect, 0, 0))
    results = map_parallell(download, args)
    results.sort()
    for tif_path, date, x, y in results:
        ds: gdal.Dataset = gdal.Open(tif_path)
        sr: osr.SpatialReference = ds.GetSpatialRef()
        for x, y, lon, lat in H5Grid.get_cells():

            min_lon, min_lat = lon, lat
            max_lon, max_lat = lon + config.cell_size, lat + config.cell_size
            xRes = 0.025
            yRes = -0.025
            options = gdal.WarpOptions(
                format="GTiff",
                outputBounds=[
                    min_lon,
                    min_lat,
                    max_lon,
                    max_lat,
                ],  # [ulx, uly, lrx, lry]
                xRes=xRes,
                yRes=yRes,
                resampleAlg="bilinear",
                srcSRS=sr,
                dstSRS="EPSG:4326",
            )
            ds_crop: gdal.Dataset = gdal.Warp(
                "/vsimem/crop.tif", ds, options=options
            )
            patch = ds_crop.ReadAsArray()
            patch = np.expand_dims(patch, axis=0)
            group = h5_get_nested(
                h5,
                [
                    "cells",
                    H5Grid.get_cell_path(x, y),
                ],
                write=True,
            )
            group.create_dataset(H5Grid.get_date_path(date), data=patch)

        print(f"Inserted {tif_path}")
    print(f"Wrote to {h5_path}")
    h5.close()


if __name__ == "__main__":
    import fire

    fire.Fire()
