import time
import fire
import torch
from tqdm import tqdm
import wandb

# NOTE: netCDF4 HAS to be imported before h5py
import netCDF4
from wildfire.download_utils import *
from wildfire.data_types import *
from wildfire.data_utils import *
from wildfire.training_utils import *
import os
from datetime import datetime, timedelta


def _gridify_modis(nc_files_by_date_sn, h5_grid, types):
    grid_day = GridDay()
    debug_i = 0
    for date, items in nc_files_by_date_sn.items():
        debug_i += 1
        print(f"Processing {date}")
        if len(items) == 5:
            (modis_250, modis_500, modis_1000, geo, mod14) = [
                items[sn] for sn in types
            ]
        else:
            (modis_1000, geo, mod14) = [items[sn] for sn in types[2:]]
        # if config.debug:
        #     if debug_i % 10 != 0:
        #         continue
        res = FireMaskExtractor.process(mod14, geo, Resolution.KM, mpp=1000)
        if res is None:
            continue

        if grid_day.should_commit(date):
            grid_day.commit(h5_grid)
            grid_day.reset(date)

        grid_day.set_is_night(res.is_night)
        local_keys_xy, grid_x_offset, grid_y_offset = grid_day.get_grid_keys(
            res, h5_grid, config.patch_size_km
        )
        shared_args = (
            res.bounds,
            local_keys_xy,
            grid_x_offset,
            grid_y_offset,
        )
        grid_day.add_raster(
            "fire_mask",
            res.filled,
            *shared_args,
            patch_size=config.patch_size_km,
        )

        data_1000 = RawExtractor.process(
            modis_1000, geo, res.bounds, res.is_night, Resolution.KM, mpp=1000
        )
        grid_day.add_raster(
            "modis_1000",
            data_1000,
            *shared_args,
            patch_size=config.patch_size_km,
        )
        if len(items) == 5:
            data_500 = RawExtractor.process(
                modis_500,
                geo,
                res.bounds,
                res.is_night,
                Resolution.MID,
                mpp=500,
            )
            grid_day.add_raster(
                "modis_500",
                data_500,
                *shared_args,
                patch_size=config.patch_size_mid,
            )
            data_250 = RawExtractor.process(
                modis_250,
                geo,
                res.bounds,
                res.is_night,
                Resolution.HIGH,
                mpp=250,
            )
            grid_day.add_raster(
                "modis_250",
                data_250,
                *shared_args,
                patch_size=config.patch_size,
            )
    grid_day.commit(h5_grid)


def _gridify_viirs(nc_files_by_date_sn, h5_grid, types):
    grid_day = GridDay()
    debug_i = 0
    for date, items in nc_files_by_date_sn.items():
        debug_i += 1
        print(f"Processing {date}")
        img_hi, img_mid, geo_hi, geo_mid, vnp_14 = [items[sn] for sn in types]
        # if config.debug:
        #     if debug_i % 10 != 0:
        #         continue
        res = FireMaskExtractor.process(vnp_14, geo_hi, Resolution.HIGH)
        if res is None:
            continue

        if grid_day.should_commit(date):
            grid_day.commit(h5_grid)
            grid_day.reset(date)

        grid_day.set_is_night(res.is_night)
        local_keys_xy, grid_x_offset, grid_y_offset = grid_day.get_grid_keys(
            res, h5_grid, config.patch_size
        )
        grid_day.add_raster(
            "fire_mask",
            res.filled,
            res.bounds,
            local_keys_xy,
            grid_x_offset,
            grid_y_offset,
            patch_size=config.patch_size,
        )

        hi_data = RawExtractor.process(
            img_hi, geo_hi, res.bounds, res.is_night, Resolution.HIGH
        )
        grid_day.add_raster(
            "hi",
            hi_data,
            res.bounds,
            local_keys_xy,
            grid_x_offset,
            grid_y_offset,
            patch_size=config.patch_size,
        )
        mid_data = RawExtractor.process(
            img_mid, geo_mid, res.bounds, res.is_night, Resolution.MID
        )
        grid_day.add_raster(
            "mid",
            mid_data,
            res.bounds,
            local_keys_xy,
            grid_x_offset,
            grid_y_offset,
            patch_size=config.patch_size_mid,
        )
    grid_day.commit(h5_grid)


def download(
    tmp_dir: str = "/scratch/local",
    start_date: str = "2019-10-15",
    end_date: str = "2020-01-15",
    test: bool = False,
    job_index: int = 0,
    job_count: int = 1,
    **kwargs,
):
    """1st step of data pipeline
    Download nc data using earthaccess. Then project onto GEOD grid, 0.25 degree
    anchored. When running in parallel, splits the date range across job_count
    workers.
    """
    init_config(**kwargs)
    if config.debug:
        # DEV
        tmp_dir = "/proj/cvl/users/x_juska/data/wildfire/nc"
        # Look at ee script big fire
        start_date = "2019-11-25"
        end_date = "2019-11-26"

    if config.local:
        tmp_dir = "/data/scratch"
        # Look at ee script big fire
        start_date = "2019-11-25"
        end_date = "2019-11-25"
    # Convert string dates to datetime
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    if config.is_modis:
        start = start - timedelta(days=1)
    # Calculate total days and days per job
    # Inclusive end date
    total_days = (end - start).days + 1
    days_per_job = math.ceil(total_days / job_count)

    # Calculate this job's date range
    job_start_date = start + timedelta(days=job_index * days_per_job)
    end_days_ahead = (job_index + 1) * days_per_job
    end_days_ahead = min(end_days_ahead, total_days)
    job_end_date = start + timedelta(days=end_days_ahead - 1)
    if config.is_modis:
        types = ["MOD02QKM", "MOD02HKM", "MOD021KM", "MOD03", "MOD14"]
        min_types = types[2:]
    else:
        types = ["VNP02IMG", "VNP02MOD", "VNP03IMG", "VNP03MOD", "VNP14IMG"]
        min_types = types
    print(
        f"Job {job_index}/{job_count} processing {job_start_date} to {job_end_date}"
    )
    time_range = tuple(
        d.strftime("%Y-%m-%d") for d in (job_start_date, job_end_date)
    )
    if not config.debug:
        nc_files_by_date_sn = nc_download(
            types, tmp_dir, time_range, config.aoi, min_types
        )
    else:
        if not os.path.exists("items.json"):
            nc_files_by_date_sn = nc_download(
                types, tmp_dir, time_range, config.aoi, min_types
            )
            nc_dump = {
                key.strftime("%Y-%m-%d %H:%M:%S"): items
                for key, items in nc_files_by_date_sn.items()
            }
            json_dump(nc_dump, "items.json")

        nc_dump = json_load("items.json")
        nc_files_by_date_sn = {
            datetime.strptime(key, "%Y-%m-%d %H:%M:%S"): items
            for key, items in nc_dump.items()
        }
    start_time = time.time()
    # ======== Output path ========
    modis_or_viirs = "modis" if config.is_modis else "viirs"
    h5_dir = config.h5_dir
    # if config.debug:
    #     h5_dir += "_dev"
    if test:
        h5_dir = os.path.join(h5_dir, "test")
    os.makedirs(h5_dir, exist_ok=True)
    t0_d = job_start_date
    if config.is_modis and job_index == 0:
        t0_d = job_start_date + timedelta(days=1)
    t0 = t0_d.strftime("%Y-%m-%d")
    t1 = job_end_date.strftime("%Y-%m-%d")
    h5_path = os.path.join(h5_dir, f"{modis_or_viirs}_{t0}_{t1}.h5")
    print(h5_path)
    # ======== Create h5 file ========
    h5_grid = H5Grid(h5_path, write=True)
    if config.is_modis:
        # 6 hour rewind
        nc_files_by_date_sn = {
            date: v
            for date, v in nc_files_by_date_sn.items()
            if (
                date.strftime("%Y-%m-%d") >= start_date
                and date.strftime("%Y-%m-%d") <= end_date
            )
        }
        _gridify_modis(nc_files_by_date_sn, h5_grid, types)
    else:
        _gridify_viirs(nc_files_by_date_sn, h5_grid, types)
    end_time = time.time()
    print(f"Saved h5 to {h5_path}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")


fire.Fire()
