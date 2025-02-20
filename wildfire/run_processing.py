from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
import os
import fire
import torch

from wildfire.data_types import H5Root, config, init_config
from wildfire.data_utils import *
from wildfire.fires import fires as fires_cpp


def _iter_blocks(grid_xs, grid_ys, block_size_x, block_size_y):
    for x in range(0, grid_xs, block_size_x):
        for y in range(0, grid_ys, block_size_y):
            yield x, y


def _iter_cells(num_dates, y0, x0, block_size_y, block_size_x):
    for t in range(num_dates):
        for y in range(y0, y0 + block_size_y):
            for x in range(x0, x0 + block_size_x):
                yield t, y, x


def _get_fire_cls(date, y, x, h5):
    cell_path = H5Grid.get_cell_path(x, y)
    if cell_path not in h5["cells"]:
        return None
    if date not in h5["cells"][cell_path]:
        return None
    cell = h5["cells"][cell_path][date]
    in_day = "fire_mask" in cell["day"]
    in_night = "fire_mask" in cell["night"]
    if in_day:
        day = cell["day"]["fire_mask"][...]
    if in_night:
        night = cell["night"]["fire_mask"][...]

    if in_day and not in_night:
        return day
    elif in_night and not in_day:
        return night
    elif in_day and in_night:
        merge = np.maximum(day, night)
        return merge
    else:
        return None


def _fire_cls_to_triary(fire_cls):
    fire_cls[fire_cls <= 2] = 0
    fire_cls[fire_cls == FireCls.CLOUD] = 0
    fire_cls[fire_cls == FireCls.UNKNOWN] = 0
    fire_cls[fire_cls == FireCls.NON_FIRE_LAND] = 1
    fire_cls[fire_cls == FireCls.NON_FIRE_WATER] = 1
    fire_cls[fire_cls >= FireCls.LOW] = 2
    return fire_cls


def find_fires(block_size=4, dev=False, is_test=False, **kwargs):
    init_config(**kwargs)
    bounds = grid_expanded_bounds(*config.aoi)
    dst_dir = os.path.join(config.root_path, "fires")
    if is_test:
        dst_dir = os.path.join(dst_dir, "test")
    os.makedirs(dst_dir, exist_ok=True)
    min_lon, min_lat, *_ = bounds
    print(min_lat)
    grid_xs, grid_ys = grid_get_dimensions(bounds)
    if not dev:
        block_size_x = grid_xs // 2
        block_size_y = grid_ys // 2
    else:
        block_size_x = block_size_y = block_size
    h5_files = [h5_get_path(config.sensor, is_test)]
    print(h5_files)
    grids = [H5Grid(h5_file) for h5_file in h5_files]
    executor = ThreadPoolExecutor(max_workers=32)
    all_blocks = list(
        _iter_blocks(grid_xs, grid_ys, block_size_x, block_size_y)
    )
    num_fire_date_yx = dict()
    for grid in grids:
        dates = list(grid.h5["num_fire_pixels_by_day"].keys())
        for date in dates:
            num_fire_date_yx[date] = grid.h5["num_fire_pixels_by_day"][date][
                ...
            ]
    all_fires = []
    # ======== Iter all blocks of AOI ========
    for block_idx, (grid_x0, grid_y0) in enumerate(all_blocks):
        if dev and block_idx != 11:
            continue
        # ======== Prepare input to cpp ========
        dates_nested = [
            list(grid.h5["num_fire_pixels_by_day"].keys()) for grid in grids
        ]
        T = sum(len(dates) for dates in dates_nested)
        im_size = config.patch_size
        block_timer = Timer(f"Block {block_idx}")
        all_fire_cls = torch.full(
            (T, im_size * block_size_y, im_size * block_size_x),
            fill_value=FireCls.NON_FIRE_LAND,
            dtype=torch.uint8,
        )
        nodes = torch.meshgrid(
            [torch.arange(n) for n in all_fire_cls.shape], indexing="ij"
        )
        nodes = torch.stack(nodes)
        nodes = nodes.permute(1, 2, 3, 0).reshape(-1, 3)
        t_offset = 0
        # ======== Iter all h5 files ========
        for grid_idx, grid in enumerate(grids):
            dates = dates_nested[grid_idx]
            futures = {}
            # ======== Iter all cells, schedule loading of fire mask ========
            for t, y, x in _iter_cells(
                len(dates), grid_y0, grid_x0, block_size_y, block_size_x
            ):
                num_fire = num_fire_date_yx[dates[t]]
                if (
                    y >= num_fire.shape[0]
                    or x >= num_fire.shape[1]
                    or num_fire[y, x] == 0
                ):
                    continue
                fut = executor.submit(_get_fire_cls, dates[t], y, x, grid.h5)
                futures[fut] = (t, y, x)
            # ======== Merge loaded cell into large image ========
            for fut in as_completed(futures):
                # t here is local
                t, y, x = futures[fut]
                t = t + t_offset
                fire_cls = fut.result()
                if fire_cls is None:
                    continue
                fire_cls = torch.from_numpy(fire_cls).squeeze(0)
                y0 = block_size_y - (y - grid_y0) - 1
                y0 = y0 * im_size
                x0 = (x - grid_x0) * im_size
                all_fire_cls[t, y0 : y0 + im_size, x0 : x0 + im_size] = fire_cls
            t_offset += len(dates)
        fire_triary = _fire_cls_to_triary(all_fire_cls)
        print(f"Block {block_idx} / {len(all_blocks) - 1}")
        block_timer.finish()
        # ======== CPP bfs, find components ========
        cpp_timer = Timer("C++")
        mask_1d = fire_triary.flatten() == 2
        nodes_filtered = nodes[mask_1d]
        components, component_dicts = fires_cpp.search(
            fire_triary, nodes_filtered
        )
        cpp_timer.finish()
        # ======== Sort components by num_fire ========
        component_dicts = sorted(
            component_dicts, key=lambda x: x["num_fire"], reverse=True
        )
        if len(component_dicts):
            print(component_dicts[0])

        # ======== Create Fire from dict ========
        tl_lon = min_lon + grid_x0 * config.cell_size
        tl_lat = (
            min_lat
            + grid_y0 * config.cell_size
            + block_size_y * config.cell_size
        )
        print(f"{tl_lat=} {min_lat=}")
        all_dates_flat = [date for dates in dates_nested for date in dates]

        def to_fire(d) -> Fire:
            im_size = config.patch_size
            gx0 = grid_x0 + d["x_min"] // im_size
            gy0 = grid_y0 + block_size_y - 1 - d["y_max"] // im_size
            gx1 = grid_x0 + d["x_max"] // im_size
            gy1 = grid_y0 + block_size_y - 1 - d["y_min"] // im_size
            px0 = d["x_min"] % im_size
            py0 = d["y_min"] % im_size
            px1 = d["x_max"] - d["x_min"] + px0
            py1 = d["y_max"] - d["y_min"] + py0
            return {
                "block_idx": block_idx,
                "idx": d["idx"],
                "start_date": all_dates_flat[d["t_min"]],
                "end_date": all_dates_flat[d["t_max"]],
                "num_fire": d["num_fire"],
                # TODO: Include pixel bounds, local - block_tl
                "bounds": [
                    tl_lon + d["x_min"] * Resolution.HIGH,  # min lon
                    tl_lat - d["y_max"] * Resolution.HIGH,  # min lat
                    tl_lon + d["x_max"] * Resolution.HIGH,  # max lon
                    tl_lat - d["y_min"] * Resolution.HIGH,  # max lat
                ],
                "grid_bounds": [gx0, gy0, gx1, gy1],
                "pixel_bounds": [px0, py0, px1, py1],
            }

        all_fires += [to_fire(d) for d in component_dicts]
        dst_tensor = os.path.join(dst_dir, f"{block_idx:02d}.pt")
        torch.save(components, dst_tensor)

    executor.shutdown(wait=True)
    # ======== Write merged fires ========
    all_fires = sorted(all_fires, key=lambda x: x["num_fire"], reverse=True)
    dst_json = os.path.join(dst_dir, f"fires.json")
    json_dump(all_fires, dst_json)
    print(f"Wrote to {dst_json}")


if __name__ == "__main__":
    fire.Fire()
