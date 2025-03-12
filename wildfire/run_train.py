import shutil
import time
from einops import reduce
import fire
import torch
from tqdm import tqdm
import wandb

from wildfire.data_types import *
from wildfire.data_utils import *
from wildfire.db import InfoSample, Run, init_db, insert
from wildfire.training_utils import *
from wildfire.dataset_utils import *
from wildfire.third.unet import Unet
import os
from glob import glob
from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint


def _normalization(dataset: WildfireDataset, dst_path: str):
    loader = get_data_loader(
        dataset, shuffle=False, pin_memory=False, batch_size=4
    )
    band_names = dataset.band_names

    mean = torch.zeros(len(band_names), dtype=torch.float64)
    mean_sq = torch.zeros(len(band_names), dtype=torch.float64)
    num_batches = 0
    for batch in tqdm(loader):
        keys = list(batch.keys())
        # Last idx is "idx"
        co = 0
        for key in keys:
            if key == "idx" or "tyx" in key or "fire_mask" in key:
                continue
            x = batch[key].to(torch.float64)
            length = x.shape[1]
            x_mean = reduce(x, "b c y x -> c", "mean")
            x_mean_sq = reduce(x**2, "b c y x -> c", "mean")
            mean[co : co + length] += x_mean
            mean_sq[co : co + length] += x_mean_sq
            co += length
        if co != len(band_names):
            raise ValueError(f"Expected {len(band_names)} bands, got {co}")
        num_batches += 1
    mean /= num_batches
    mean_sq /= num_batches
    std_dev = torch.sqrt(mean_sq - mean**2)
    content = {}
    for i, name in enumerate(band_names):
        content[name] = {
            "mean": mean[i].item(),
            "std_dev": std_dev[i].item(),
        }
    json_dump(content, dst_path)
    print(f"Saved normalization to {dst_path}")


def _load_cache(dataset: WildfireDataset):
    dataset.cache = {}
    loader = get_data_loader(
        dataset, shuffle=False, pin_memory=False, batch_size=4, num_workers=31
    )
    for batch in tqdm(loader):
        for i, idx in enumerate(batch["idx"]):
            dataset.cache[idx] = {k: batch[k][i] for k in batch.keys()}


losses = {
    "bce": BCELoss(config.pos_weight),
    "iou": wildfire_lovasz_hinge,
    "iou_mix": IOU_MixLoss(0.8, 0.2, config.pos_weight),
}


def pick_samples_save(**kwargs):
    init_config(**kwargs)
    sensors = ["viirs", "modis"]
    train_paths = [h5_get_path(sensor, False) for sensor in sensors]
    test_paths = [h5_get_path(sensor, True) for sensor in sensors]
    train = pick_samples(*train_paths)
    test = pick_samples(*test_paths)
    train, val = split_samples_by_xy(train)

    def save(samples, name):
        samples = [asdict(sample) for sample in samples]
        print(f"{name}: {len(samples)} samples")
        path = os.path.join(config.h5_dir, f"{name}.json")
        print(f"Saving {path}")
        json_dump(samples, path)

    save(train, "train")
    save(val, "val")
    save(test, "test")


def extract_datasets(out_dir: str, **kwargs):
    if kwargs:
        init_config(**kwargs)
    os.makedirs(out_dir, exist_ok=True)

    def load_samples(name):
        path = os.path.join(config.h5_dir, f"{name}.json")
        samples = json_load(path)
        samples = [Sample(**sample) for sample in samples]
        return samples

    sensors = ["viirs", "modis"]

    def get_dst_path(path: str) -> str:
        rel_path = os.path.relpath(path, config.h5_dir)
        return os.path.join(out_dir, rel_path)

    def copy_samples(src_path, dst_path, samples, dates=None):
        print(f"Copying {src_path} to {dst_path}")
        h5 = h5py.File(src_path, "r")
        dates_was_none = dates is None
        if dates_was_none:
            dates = list(h5["num_fire_pixels_by_day"].keys())
        if os.path.exists(dst_path):
            print(f"Skipping {dst_path} because it already exists")
            return dates
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        h5_out = h5py.File(dst_path, "w")
        if dates_was_none and "num_fire_pixels_by_day" not in h5_out:
            h5.copy("num_fire_pixels_by_day", h5_out)
        for sample in tqdm(samples):
            xy = H5Grid.get_cell_path(sample.x, sample.y)
            path_full = "/".join(["cells", xy, dates[sample.t]])
            if path_full not in h5:
                continue
            path_parent = "/".join(["cells", xy])
            if path_parent not in h5_out:
                h5_out.create_group(path_parent)
            dst = h5_out[path_parent]
            h5.copy(path_full, dst)
        return dates

    train = load_samples("train")
    val = load_samples("val")
    test = load_samples("test")
    for sensor in sensors:
        train_h5 = h5_get_path(sensor, False)
        dates_train = copy_samples(
            train_h5, get_dst_path(train_h5), train + val
        )
        test_h5 = h5_get_path(sensor, True)
        dates_test = copy_samples(test_h5, get_dst_path(test_h5), test)
        print(f"Copied {sensor}")

    for t in ["train", "val", "test"]:
        src = os.path.join(config.h5_dir, f"{t}.json")
        dst = os.path.join(out_dir, f"{t}.json")
        shutil.copy(src, dst)

    for aux in ["drought", "era5"]:
        train_h5 = h5_get_path(aux, False)
        copy_samples(
            train_h5, get_dst_path(train_h5), train + val, dates=dates_train
        )
        test_h5 = h5_get_path(aux, True)
        copy_samples(test_h5, get_dst_path(test_h5), test, dates=dates_test)
        print(f"Copied {aux}")


class FinetuneTrainer(LightningModule):
    def __init__(
        self, train_dataset: WildfireDataset, run_dir: str, run_id: str
    ):
        super().__init__()
        self.run_dir = run_dir
        self.run_id = run_id
        self.is_sanity: bool = True
        self.validation_metrics = []
        self.metrics_by_thr = {
            thr: WildfireMetrics(thr) for thr in config.prediction_thresholds
        }
        if (
            not os.path.exists(config.normalization_path)
            or config.redo_normalization
        ):
            print(f"Calculating {config.normalization_path}, hold on...")
            _normalization(train_dataset, config.normalization_path)

        # ======== Load normalization ========
        norm_dict = json_load(config.normalization_path)
        mean, std_dev = [], []
        for band in train_dataset.band_names:
            mean.append(norm_dict[band]["mean"])
            std_dev.append(norm_dict[band]["std_dev"])
        self.normalization = Normalization(mean, std_dev)
        in_channels = len(train_dataset.band_names)
        self.model = Unet(
            config.backbone, in_channels=in_channels, num_classes=1
        )
        self.criterion = losses[config.loss]

        json_dump(
            config.wandb_config(), os.path.join(self.run_dir, "config.json")
        )
        if config.save_all or config.test_save_all:
            self.prepare_h5()
        else:
            self.h5_tensors = None

    def prepare_h5(self):
        self.h5_tensors: h5py.File | None = h5py.File(
            self.run_dir + "/tensors.h5", "w"
        )
        print(f"Run dir: {self.run_dir}")

    def __del__(self):
        try:
            self.h5_tensors.close()
        except Exception as e:
            pass

    # ======== Done with training ========
    def on_validation_epoch_start(self) -> None:
        # ======== To cuda ========
        # if self.is_sanity:

        if self.is_sanity:
            return

        metrics = self.get_and_reset_metrics()
        for k, v in metrics.items():
            self.log("train/" + k, v, on_step=False, on_epoch=True)

    # ======== Done with validation ========
    def on_validation_epoch_end(self):
        metrics = self.get_and_reset_metrics()
        self.validation_metrics.append(
            {k: v.item() for k, v in metrics.items()}
        )
        ret = {}
        for k, v in metrics.items():
            k = "valid/" + k
            ret[k] = v
            self.log(k, v, on_step=False, on_epoch=True)
        self.is_sanity = False
        return ret

    def on_test_epoch_start(self):
        print("Starting test")

    def on_test_epoch_end(self) -> None:
        metrics = self.get_and_reset_metrics()
        print("Reset test")
        for k, v in metrics.items():
            self.log("test/" + k, v, on_step=False, on_epoch=True)

    def update_metrics(
        self, pred_prob: Tensor, gt_cls: Tensor, loss_mask: Tensor
    ):
        for metric in self.metrics_by_thr.values():
            metric.update(pred_prob, gt_cls, loss_mask)

    def get_and_reset_metrics(self):
        metrics = {}
        for thr, metric in self.metrics_by_thr.items():
            thr_metrics = metric.get_and_reset()
            thr_metrics = {f"{thr}/{k}": v for k, v in thr_metrics.items()}
            metrics.update(thr_metrics)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        return optimizer

    def save_tensor(self, data: dict, path: list[str]):
        group = h5_get_nested(self.h5_tensors, path, write=True)
        for key, val in data.items():
            val = val.detach().cpu().numpy()
            group.create_dataset(key, data=val)

    def forward(self, x, batch: FinetuningBatch, prefix: str = "train/"):
        cur_fire_cls, next_fire_cls = (
            batch["cur_fire_mask"],
            batch["next_fire_mask"],
        )

        def reduce_to_km(x):
            km_size = config.patch_size_km
            return reduce(
                x, "b c (y 3) (x 3) -> b c y x", "max", y=km_size, x=km_size
            )

        is_train = prefix == "train/"
        if config.get_target(not is_train) == "viirs":
            cur_fire_cls = reduce_to_km(cur_fire_cls)
            next_fire_cls = reduce_to_km(next_fire_cls)

        if config.include_fire_mask:
            cur_mask = cur_fire_cls
            cur_mask = (cur_mask - 5.0) / 4.0
            x = torch.cat([x, cur_mask], dim=-3)
        loss_mask = get_loss_mask(cur_fire_cls, next_fire_cls)

        if config.baseline:
            loss = torch.tensor(0.0, device=x.device, requires_grad=True)
            pred_prob = fire_cls_to_prob(cur_fire_cls)
        else:
            logits = self.model(x)
            logits = reduce_to_km(logits)
            loss = self.criterion(logits, next_fire_cls, loss_mask)
            pred_prob = torch.sigmoid(logits)
        save_all = config.save_all and not self.is_sanity
        save_test = prefix == "test/" and config.test_save_all
        if save_all or save_test:
            self._save_batch(
                batch,
                pred_prob.detach().cpu(),
                cur_fire_cls.detach().cpu(),
                next_fire_cls.detach().cpu(),
                loss_mask.detach().cpu(),
                prefix,
            )
        self.update_metrics(
            pred_prob.detach(), next_fire_cls.detach(), loss_mask.detach()
        )
        self.log(
            prefix + "loss",
            loss,
            prog_bar=is_train,
            on_step=is_train,
            on_epoch=True,
        )
        return loss, pred_prob

    def _save_batch(
        self,
        batch,
        pred_prob,
        cur_fire_cls,
        next_fire_cls,
        loss_mask,
        prefix,
    ):
        cur_num_fire = reduce(
            fire_cls_to_prob(cur_fire_cls), "b ... -> b", "sum"
        )
        next_num_fire = reduce(
            fire_cls_to_prob(next_fire_cls), "b ... -> b", "sum"
        )
        idxs = batch["idx"].cpu().tolist()
        tyx = batch["tyx"].cpu()
        for i, idx in enumerate(idxs):
            m = WildfireMetrics()
            dataset = prefix.replace("/", "")
            self.save_tensor(
                {
                    "pred_prob": pred_prob[i],
                    "cur_fire_cls": cur_fire_cls[i],
                    "next_fire_cls": next_fire_cls[i],
                    "loss_mask": loss_mask[i],
                    "tyx": tyx[i],
                },
                path=[dataset, self.current_epoch, idx],
            )
            m.update(pred_prob[i], next_fire_cls[i], loss_mask[i])
            t, y, x = [x.item() for x in tyx[i]]
            info: InfoSample = {
                "run_id": self.run_id,
                "idx": idx,
                "epoch": self.current_epoch,
                "dataset": dataset,
                "cur_num_fire": cur_num_fire[i],
                "next_num_fire": next_num_fire[i],
                "tn": m.TN,
                "fn": m.FN,
                "fp": m.FP,
                "tp": m.TP,
                "t": t,
                "y": y,
                "x": x,
                **m.get(),
            }
            while True:
                try:
                    insert(InfoSample, info)
                    break
                except Exception as e:
                    print(e)
                    print(info)

    def training_step(self, batch: FinetuningBatch, batch_idx: int):
        x = upsample_and_cat(batch)
        if config.train_rotation:
            angle = torch.randint(-180, 180, (1,)).item()
            x = rotate(
                x, angle, interpolation=InterpolationMode.BILINEAR, fill=0
            )
            for key in ["cur_fire_mask", "next_fire_mask"]:
                val = batch[key]
                val = rotate(val, angle, InterpolationMode.NEAREST, fill=0)
                batch[key] = val

        x = self.normalization(x)
        loss, _ = self.forward(x, batch)
        return loss

    def validation_step(self, batch: FinetuningBatch, batch_idx: int):
        x = upsample_and_cat(batch)
        x = self.normalization(x)
        _, pred_prob = self.forward(x, batch, prefix="valid/")

    def test_step(self, batch: FinetuningBatch, batch_idx: int):
        x = upsample_and_cat(batch)
        x = self.normalization(x)
        _, pred_prob = self.forward(x, batch, prefix="test/")


def finetune(**kwargs):
    """(2nd) step of training pipeline
    Finetune a model on next day fire prediction. Weighted sampling so we have
    mostly fire occurences.
    """
    # For ffcv, we still need a Dataset class. So first we can try finetuning on
    # that straight up without using ffcv.
    if kwargs:
        init_config(**kwargs)
    init_db(config.db_path)

    def load_samples(name):
        path = os.path.join(config.h5_dir, f"{name}.json")
        samples = json_load(path)
        samples = [Sample(**sample) for sample in samples]
        return samples

    def get_other_path(test: bool, test_dir: bool):
        return h5_get_path(config.get_target(test), test_dir)

    train_dataset = WildfireDataset(
        load_samples("train"),
        h5_get_path(config.sensor, False),
        get_other_path(False, False),
    )
    val_dataset = WildfireDataset(
        load_samples("val"),
        h5_get_path(config.sensor, False),
        get_other_path(True, False),
    )
    test_dataset = WildfireDataset(
        load_samples("test"),
        h5_get_path(config.sensor, True),
        get_other_path(True, True),
    )
    if config.in_memory:
        for ds in [train_dataset, val_dataset]:
            _load_cache(ds)

    def _save_paths(module: FinetuneTrainer):
        if module.h5_tensors is None:
            return
        train = h5_get_nested(module.h5_tensors, ["train"], write=True)
        train.attrs.create("sat_path", train_dataset.sat_path)
        train.attrs.create("target_path", train_dataset.target_path)
        val = h5_get_nested(module.h5_tensors, ["val"], write=True)
        val.attrs.create("sat_path", val_dataset.sat_path)
        val.attrs.create("target_path", val_dataset.target_path)
        test = h5_get_nested(module.h5_tensors, ["test"], write=True)
        test.attrs.create("sat_path", test_dataset.sat_path)
        test.attrs.create("target_path", test_dataset.target_path)

    print(f"train_dataset: {len(train_dataset)}")
    print(f"val_dataset: {len(val_dataset)}")
    train = get_data_loader(train_dataset, shuffle=True)
    val = get_data_loader(val_dataset, shuffle=False)
    test = get_data_loader(test_dataset, shuffle=False)
    # ======== Num runs ========
    single = config.num_runs == 1
    test_results = []
    run_id = insert(Run, {"name": config.name}).lastrowid
    for i in range(config.num_runs):
        suffix = "" if single else f"_{i}"
        if not config.debug and config.wandb:
            # wandb.login()
            wandb_config = config.wandb_config()
            wandb_config["run_id"] = run_id
            logger = WandbLogger(
                project="wildfire",
                name=config.name + suffix,
                config=wandb_config,
            )
        else:
            logger = None
        torch.set_float32_matmul_precision(
            ["medium", "high", "high"][config.precision]
        )

        run_dir = os.path.join(config.runs_path, str(run_id) + suffix)
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoint dir: {checkpoint_dir}")
        module = FinetuneTrainer(train_dataset, run_dir, str(run_id) + suffix)
        _save_paths(module)
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor="valid/0.5/iou",
            mode="max",
            save_top_k=1,
            save_last=False,
        )
        trainer = Trainer(
            max_epochs=config.num_epochs,
            num_sanity_val_steps=2,
            precision=["bf16-true", "bf16-mixed", "32-true"][config.precision],
            check_val_every_n_epoch=config.check_val_every_n_epoch,
            logger=logger,
            # fast_dev_run=config.debug,
            log_every_n_steps=10,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(module, train, val)
        json_dump(
            module.validation_metrics,
            os.path.join(run_dir, "validation_metrics.json"),
        )
        res = trainer.test(module, test, "best", verbose=True)[0]
        test_results.append(res)
        json_dump(res, os.path.join(run_dir, "test_results.json"))
        wandb.finish()

    if single:
        return
    # ======== Save test samples ========
    results_by_metric = defaultdict(list)
    for res in test_results:
        for k, v in res.items():
            results_by_metric[k].append(v)

    for k, v in results_by_metric.items():
        if "0.25/" in k or "loss" in k:
            continue
        std_dev = np.std(v)
        mean = np.mean(v)
        print(f"{k}: {mean:.4f} ± {std_dev:.4f}")

    best_run = np.argmax(results_by_metric["test/0.5/iou"])
    print(f"Best run: {config.runs_path}/{run_id}_{best_run}")


fire.Fire()
