from dataclasses import dataclass
from glob import glob
from tokenize import Single
from typing import Optional, Type, TypeVar
import numpy as np
import h5py
from torch.utils.data import DataLoader, Dataset
from wildfire.data_utils import *
from wildfire.data_types import *


@dataclass
class Sample:
    x: int
    y: int
    t: int


def f32_zeros(*shape):
    return np.zeros(shape, dtype=np.float32)


size_org = config.patch_size
size_mid = config.patch_size_mid
size_km = config.patch_size_km
band_defaults = {
    "day": {
        "hi": f32_zeros(len(viirs_hi_day_bands), size_org, size_org),
        "mid": f32_zeros(len(viirs_mid_day_bands), size_mid, size_mid),
        "modis_1000": f32_zeros(len(modis_1000m_day_bands), size_km, size_km),
        "modis_500": f32_zeros(len(modis_500m_bands), size_mid, size_mid),
        "modis_250": f32_zeros(len(modis_250m_bands), size_org, size_org),
        "fire_mask": f32_zeros(1, size_org, size_org),
        "modis_fire_mask": f32_zeros(1, size_km, size_km),
    },
    "night": {
        "hi": f32_zeros(len(viirs_hi_night_bands), size_org, size_org),
        "mid": f32_zeros(len(viirs_mid_night_bands), size_mid, size_mid),
        "modis_1000": f32_zeros(len(modis_1000m_night_bands), size_km, size_km),
        "fire_mask": f32_zeros(1, size_org, size_org),
        "modis_fire_mask": f32_zeros(1, size_km, size_km),
    },
    "era5": f32_zeros(len(era5_bands), 15, 15),
    "drought": f32_zeros(1, 30, 30),
}


def open_aux(paths, new_sensor):
    return [
        h5py.File(h5_replace_sensor(path, new_sensor), "r") for path in paths
    ]


class WildfireDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        sat_h5_path: str,
        target_h5_path: str,
    ):
        self.samples = samples
        self.sat = h5py.File(sat_h5_path, "r")
        self.sat_path = sat_h5_path
        self.target = h5py.File(target_h5_path, "r")
        self.target_path = target_h5_path
        self.prepend_modis = "modis" in os.path.basename(target_h5_path)

        self.aux_h5 = {
            "era5": h5py.File(h5_replace_sensor(sat_h5_path, "era5"), "r"),
            "drought": h5py.File(
                h5_replace_sensor(sat_h5_path, "drought"), "r"
            ),
        }
        self.dates = list(self.sat["num_fire_pixels_by_day"].keys())
        self.cache = None

    def _get_bands(
        self,
        t: int,
        y: int,
        x: int,
        day_type: str,
        band_name: str,
    ):
        if band_name == "fire_mask":
            h5 = self.target
        else:
            h5 = self.sat
        cell_path = H5Grid.get_cell_path(x, y)
        date = self.dates[t]
        cell = h5_get_nested(
            h5, ["cells", cell_path, date, day_type, band_name]
        )
        if cell is None:
            if self.prepend_modis and band_name == "fire_mask":
                band_name = "modis_fire_mask"
            return band_defaults[day_type][band_name]
        val = cell[...].astype(np.float32)
        if band_name == "fire_mask":
            val[val == FireCls.NON_FIRE_WATER] = FireCls.NON_FIRE_LAND
        else:
            val[val == config.uint16_no_data] = 0
        return val

    def _get_bands_aux(self, key, t, y, x):
        h5 = self.aux_h5[key]
        cell_path = H5Grid.get_cell_path(x, y)
        date = self.dates[t]
        cell = h5_get_nested(h5, ["cells", cell_path, date])
        if cell is None:
            return band_defaults[key]
        val = cell[...].astype(np.float32)
        return val

    def __len__(self):
        # return 1 if config.debug else len(self.samples)
        return len(self.samples)

    def get_viirs_bands(self, t, y, x):
        hi = np.concatenate(
            [
                self._get_bands(t, y, x, "day", "hi"),
                self._get_bands(t, y, x, "night", "hi"),
            ]
        )
        mid = np.concatenate(
            [
                self._get_bands(t, y, x, "day", "mid"),
                self._get_bands(t, y, x, "night", "mid"),
            ]
        )
        return {
            "hi": hi,
            "mid": mid,
        }

    def get_modis_bands(self, t, y, x):
        return {
            "modis_1000": np.concatenate(
                [
                    self._get_bands(t, y, x, "day", "modis_1000"),
                    self._get_bands(t, y, x, "night", "modis_1000"),
                ]
            ),
            "modis_500": self._get_bands(t, y, x, "day", "modis_500"),
            "modis_250": self._get_bands(t, y, x, "day", "modis_250"),
        }

    def __getitem__(self, idx) -> FinetuningBatch:
        if config.debug:
            idx = 0

        if config.in_memory and idx in self.cache:
            return self.cache[idx]

        sample = self.samples[idx]
        t, y, x = sample.t, sample.y, sample.x

        next_fire_mask = np.maximum(
            self._get_bands(t + 1, y, x, "day", "fire_mask"),
            self._get_bands(t + 1, y, x, "night", "fire_mask"),
        )
        cur_fire_mask = np.maximum(
            self._get_bands(t, y, x, "day", "fire_mask"),
            self._get_bands(t, y, x, "night", "fire_mask"),
        )
        era5 = self._get_bands_aux("era5", t, y, x)
        drought = self._get_bands_aux("drought", t, y, x)
        sat_bands = (
            self.get_modis_bands(t, y, x)
            if config.is_modis
            else self.get_viirs_bands(t, y, x)
        )
        return {
            **sat_bands,
            "next_fire_mask": next_fire_mask,
            "cur_fire_mask": cur_fire_mask,
            "era5": era5,
            "drought": drought,
            "idx": idx,
            "tyx": np.array([t, y, x]),
        }

    @property
    def band_names(self):
        if config.is_modis:
            sat_bands = modis_bands
        else:
            sat_bands = viirs_hi_bands + viirs_mid_bands
        return sat_bands + era5_bands + drought_bands


def pick_samples(viirs_path, modis_path):
    viirs = h5py.File(viirs_path, "r")
    modis = h5py.File(modis_path, "r")
    dates = list(viirs["num_fire_pixels_by_day"].keys())
    dates_cmp = list(modis["num_fire_pixels_by_day"].keys())
    assert dates == dates_cmp
    num_fire_viirs = viirs["num_fire_pixels_by_day"]
    num_fire_modis = modis["num_fire_pixels_by_day"]
    samples = []
    tot_modis_fire = 0
    tot_viirs_fire = 0
    for t, date in enumerate(dates[:-1]):
        # Get fire pixels for this date
        fire_pixels_viirs = num_fire_viirs[date][...]  # Y x X array
        fire_pixels_modis = num_fire_modis[date][...]  # Y x X array
        # Find locations where number of fire pixels exceeds threshold
        mask = np.ones_like(fire_pixels_viirs, dtype=bool)
        mask = mask & (fire_pixels_viirs > (config.min_num_fire_pixels * 5))
        mask = mask & (fire_pixels_modis > config.min_num_fire_pixels)
        y, x = np.where(mask)
        y, x = y.tolist(), x.tolist()
        samples.extend([Sample(x, y, t) for x, y in zip(x, y)])
        tot_modis_fire += np.sum(fire_pixels_modis[y, x])
        tot_viirs_fire += np.sum(fire_pixels_viirs[y, x])
    print(f"Picked {len(samples)} samples from {viirs_path} and {modis_path}")
    ratio_modis = tot_modis_fire / (len(samples) * config.patch_size_km**2)
    ratio_viirs = tot_viirs_fire / (len(samples) * config.patch_size**2)
    print(f"Modis fire ratio: {ratio_modis * 100:.2f}%")
    print(f"Viirs fire ratio: {ratio_viirs * 100:.2f}%")
    return samples


def split_samples_by_date(
    samples: list[Sample], dates: list[str], split_date: str
):
    samples1, samples2 = [], []
    for sample in samples:
        date = dates[sample.t]
        if date <= split_date:
            samples1.append(sample)
        else:
            samples2.append(sample)
    print(f"Split into {len(samples1)} and {len(samples2)}")
    return samples1, samples2


def split_samples_by_xy(samples: list[Sample]):
    """
    Split spans into train/val
    """
    mod_sqrt = 2
    x_keys = set()
    y_keys = set()
    for sample in samples:
        x_keys.add(sample.x)
        y_keys.add(sample.y)
    x_keys = {x: i for i, x in enumerate(sorted(x_keys))}
    y_keys = {y: i for i, y in enumerate(sorted(y_keys))}
    train_samples = []
    val_samples = []
    for sample in samples:
        if (
            x_keys[sample.x] % mod_sqrt == 0
            and y_keys[sample.y] % mod_sqrt == 0
        ):
            val_samples.append(sample)
        else:
            train_samples.append(sample)
    print("Split into:")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    return train_samples, val_samples


def get_data_loader(
    dataset: Dataset,
    batch_size: Optional[int] = None,
    shuffle: bool = False,
    pin_memory: bool = True,
    num_workers: Optional[int] = None,
):
    batch_size = config.batch_size if batch_size is None else batch_size
    default_num_workers = 0 if config.in_memory else 31
    num_workers = default_num_workers if num_workers is None else num_workers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
