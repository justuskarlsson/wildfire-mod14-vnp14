from dataclasses import asdict, dataclass, field
import os

from typing import Literal, Any, TypedDict
import numpy.typing as npt
import numpy as np
import torch
from torch import Tensor


Bounds = tuple[float, float, float, float]


@dataclass
class Config:
    """Configuration for a project"""

    local: bool = False

    @property
    def is_viirs(self):
        return not self.is_modis

    @property
    def target_is_viirs(self):
        return self.is_viirs ^ self.other_target

    def get_target(self, is_test: bool):
        train_target = "viirs" if self.target_is_viirs else "modis"
        if not is_test:
            return train_target
        if self.swap_train_test:
            return "modis" if train_target == "viirs" else "viirs"
        return train_target

    patch_size: int = 192
    root_path: str = os.environ["DATA"] + "/wildfire"
    runs_path: str = ""
    db_path: str = ""
    finetune: bool = True
    # backbone: str = "convnextv2_large"
    # backbone: str = "vgg16"
    aoi: Bounds = (112.0, -44.0, 154.0, -10.0)
    # ======== Wandb Config========
    num_runs: int = 1
    is_modis: bool = False
    code_version = "2.0"
    backbone: str = "resnet50"
    batch_size: int = 64
    num_epochs: int = 40
    lr: float = 0.005
    pos_weight: float = 3.0
    min_num_fire_pixels: int = 1
    loss: str = "bce"
    include_fire_mask: bool = False
    train_rotation: bool = False
    other_target: bool = False
    swap_train_test: bool = False
    min_confidence: int = 8
    split_date: str = "2019-12-15"

    # ======== Other ========
    save_pred_every: int = 5
    precision: Literal[0, 1, 2] = 2
    check_val_every_n_epoch: int = 1
    debug: bool = False
    prediction_thresholds: list[float] = field(
        default_factory=lambda: [0.25, 0.5]
    )
    loss_pixel_padding: int = 6
    baseline: bool = False
    name: str = ""
    wandb: bool = True
    save_all: bool = False
    # TODO: Change to True
    test_save_all: bool = False
    uint16_no_data: int = 65535
    redo_normalization: bool = False
    in_memory: bool = False

    def __post_init__(self):
        self.runs_path = self.root_path + "/runs"
        self.db_path = self.runs_path + "/db.sqlite"
        self.local = os.environ.get("LOCAL", "0") == "1"

        if self.local:
            self.aoi = (150.05859375, -33.0234375, 151.07421875, -32.5234375)
            self.batch_size = 8
            self.num_epochs = 300
            self.lr = 1e-2

    @property
    def cell_size(self):
        return round(self.patch_size / 64) * 0.25

    @property
    def patch_size_mid(self):
        return self.patch_size // 2

    @property
    def patch_size_km(self):
        return self.patch_size // 3

    @property
    def h5_dir(self):
        return os.path.join(self.root_path, "h5")

    @property
    def sensor(self):
        return "modis" if self.is_modis else "viirs"

    @property
    def normalization_path(self):
        return os.path.join(self.h5_dir, f"{self.sensor}_normalization.json")

    def asdict(self):
        return asdict(self)

    def wandb_config(self):
        keys = [
            "is_modis",
            "other_target",
            "code_version",
            "backbone",
            "batch_size",
            "num_epochs",
            "lr",
            "pos_weight",
            "min_num_fire_pixels",
            "loss",
            "include_fire_mask",
            "train_rotation",
            "min_confidence",
            "swap_train_test",
            "split_date",
        ]
        return {k: getattr(self, k) for k in keys}


config = Config()


def init_config(**kwargs):
    # NOTE: If Config fields depend on each other, should not be applied at
    # construction. Instead use computed values for those fields.
    global config
    for k, v in kwargs.items():
        setattr(config, k, v)


class Resolution:
    HIGH = 0.25 / 64  # ~375m, MODIS qkm, hkm
    MID = 0.25 * 2 / 64  # ~750m, MODIS, hkm
    KM = 0.25 * 3 / 64  # ~ 1000 m

    @staticmethod
    def get_patch_size(resolution: float):
        if resolution == Resolution.HIGH:
            return config.patch_size
        elif resolution == Resolution.MID:
            return config.patch_size_mid
        elif resolution == Resolution.KM:
            return config.patch_size_km


class FireCls:
    NO_DATA = 0  # Custom label, not in original data
    NOT_PROCESSED_1 = 1
    NOT_PROCESSED_2 = 2
    NON_FIRE_WATER = 3
    CLOUD = 4
    NON_FIRE_LAND = 5
    UNKNOWN = 6
    LOW = 7
    MEDIUM = 8
    HIGH = 9

    @staticmethod
    def no_data_values():
        """
        In h5 format, these will be mapped to NO_DATA and CLOUD.
        """
        return [
            FireCls.NO_DATA,
            FireCls.NOT_PROCESSED_1,
            FireCls.NOT_PROCESSED_2,
            FireCls.CLOUD,
            FireCls.UNKNOWN,
        ]


class HalfDay(TypedDict):
    hi: np.ndarray
    mid: np.ndarray
    fire_mask: np.ndarray
    ratio_data: float
    num_fire_pixels: int


class FullDay(TypedDict):
    night: HalfDay
    day: HalfDay


CellDays = dict[str, FullDay]


class H5Root(TypedDict):
    num_fire_pixels_by_day: dict[str, np.ndarray]
    cells: dict[str, CellDays]


class FinetuningBatch(TypedDict):
    hi: Tensor
    mid: Tensor
    next_fire_mask: Tensor
    cur_fire_mask: Tensor
    idx: list[int]


def get_band_names(resolution, is_night):
    def get_range(prefix, start, last):
        return [f"{prefix}{i:02d}" for i in range(start, last + 1)]

    if resolution == Resolution.HIGH:
        if is_night:
            return get_range("I", 4, 5)
        else:
            return get_range("I", 1, 5)

    else:
        if is_night:
            return get_range("M", 7, 8) + get_range("M", 10, 16)
        else:
            return get_range("M", 1, 16)


class Fire(TypedDict):
    idx: int
    block_idx: int
    num_fire: int
    start_date: str
    end_date: str
    bounds: list[float]


# ===========================
# ========  Constants  ========
# ===========================


viirs_hi_day_bands = get_band_names(Resolution.HIGH, False)
viirs_hi_night_bands = get_band_names(Resolution.HIGH, True)
viirs_hi_bands = viirs_hi_day_bands + viirs_hi_night_bands

viirs_mid_day_bands = get_band_names(Resolution.MID, False)
viirs_mid_night_bands = get_band_names(Resolution.MID, True)
viirs_mid_bands = viirs_mid_day_bands + viirs_mid_night_bands

modis_250m_bands = ["1", "2"]
modis_500m_bands = ["3", "4", "5", "6", "7"]
modis_1000m_refl_bands = [
    "8",
    "9",
    "10",
    "11",
    "12",
    "13h",
    "13l",
    "14h",
    "14l",
    "15",
    "16",
    "17",
    "18",
    "19",
    "26",
]
modis_1000m_emiss_bands = [
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "27",
    "28",
    "29",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
]

modis_1000m_day_bands = modis_1000m_refl_bands + modis_1000m_emiss_bands
modis_1000m_night_bands = ["n_" + band for band in modis_1000m_emiss_bands]
modis_bands = (
    modis_1000m_day_bands
    + modis_1000m_night_bands
    + modis_500m_bands
    + modis_250m_bands
)

era5_bands = [
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "temperature_2m_max",
    "dewpoint_temperature_2m_min",
    "total_precipitation_sum",
    "volumetric_soil_water_layer_1_min",
    "surface_solar_radiation_downwards_max",
    "leaf_area_index_low_vegetation_max",
    "leaf_area_index_high_vegetation_max",
    "skin_temperature_max",
]

drought_bands = [
    "drought_kbdi",
]
