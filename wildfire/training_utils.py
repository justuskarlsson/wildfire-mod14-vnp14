from collections import defaultdict
from einops import rearrange
import torch
from torch import Tensor

from torch import nn
import torch.nn.functional as F
from torchvision.transforms.functional import rotate
import h5py
from torchvision.transforms.functional import InterpolationMode

from wildfire.third.lovasz_losses import lovasz_hinge
from wildfire.data_types import *


class Normalization(nn.Module):
    def __init__(self, mean: list[float], std_dev: list[float]):
        super().__init__()
        # b c h w
        self.mean = torch.Tensor(mean).reshape(1, -1, 1, 1)
        self.std_dev = torch.Tensor(std_dev).reshape(1, -1, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        if x.device != self.mean.device:
            self.mean = self.mean.to(x.device)
            self.std_dev = self.std_dev.to(x.device)
        return (x - self.mean) / self.std_dev


class Rotation(nn.Module):
    def __init__(self, angle: float):
        super().__init__()
        self.angle = angle

    def forward(self, x: Tensor) -> Tensor:
        return F.rotate(x, self.angle)


def upsample_and_cat(batch: FinetuningBatch) -> Tensor:

    def upsample(data):
        return nn.functional.interpolate(
            data,
            size=(config.patch_size, config.patch_size),
            mode="bilinear",
        )

    if config.is_viirs:
        stack = [
            batch["hi"],
            upsample(batch["mid"]),
        ]
    else:
        stack = [
            upsample(batch["modis_1000"]),
            upsample(batch["modis_500"]),
            batch["modis_250"],
        ]

    stack += [
        upsample(batch["era5"]),
        upsample(batch["drought"]),
    ]
    channel_dim = -3
    return torch.cat(stack, dim=channel_dim)


def fire_cls_to_prob(fire_cls: Tensor) -> Tensor:
    return (fire_cls >= config.min_confidence).to(fire_cls.dtype)


def feq(x, y):
    return torch.abs(x - y) < 1e-6


def fneq(x, y):
    return torch.abs(x - y) < 1e-6


def get_loss_mask(cur_fire_cls: Tensor, next_fire_cls: Tensor) -> Tensor:
    """
    Strict equality, small ints exact for floats.
    """
    mask = torch.zeros_like(cur_fire_cls, dtype=torch.bool)
    o = config.loss_pixel_padding
    if o > 0:
        mask[..., o:-o, o:-o] = True
    else:
        mask[...] = True

    mask &= next_fire_cls != FireCls.CLOUD
    mask &= next_fire_cls != FireCls.NO_DATA
    mask &= next_fire_cls != FireCls.LOW
    if config.min_confidence == FireCls.HIGH:
        mask &= next_fire_cls != FireCls.MEDIUM
    mask &= cur_fire_cls != FireCls.CLOUD
    mask &= cur_fire_cls != FireCls.NO_DATA
    return mask


class BCELoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight]).reshape(1, 1, 1)

    def forward(
        self, pred: Tensor, gt_cls: Tensor, loss_mask: Tensor
    ) -> Tensor:
        gt_prob = fire_cls_to_prob(gt_cls)
        if self.pos_weight.device != pred.device:
            self.pos_weight = self.pos_weight.to(pred.device)
        h, w = pred.shape[-2:]
        flatten = lambda x: x.reshape(-1, 1, h, w)
        loss = F.binary_cross_entropy_with_logits(
            flatten(pred),
            flatten(gt_prob),
            reduction="none",
            pos_weight=self.pos_weight,
        )
        loss = loss.reshape(gt_cls.shape)
        mean_loss = (loss * loss_mask).sum() / loss_mask.sum()
        return mean_loss


def wildfire_lovasz_hinge(pred_logits, gt_cls, loss_mask: Tensor):
    ignore = -1
    gt_labels = fire_cls_to_prob(gt_cls).long()
    gt_labels[~loss_mask] = ignore
    # B 1 H W -> B H W
    h, w = gt_labels.shape[-2:]
    pred_logits = pred_logits.reshape(-1, h, w)
    gt_labels = gt_labels.reshape(-1, h, w)
    return lovasz_hinge(pred_logits, gt_labels, per_image=False, ignore=ignore)


class IOU_MixLoss(nn.Module):
    def __init__(self, a, b, pos_weight: float = 1.0):
        super().__init__()
        self.bce_loss = BCELoss(pos_weight)
        self.lovasz_hinge_loss = wildfire_lovasz_hinge
        self.a = a
        self.b = b

    def forward(self, pred_logits, gt_cls, loss_mask: Tensor) -> Tensor:
        bce_loss = self.bce_loss(pred_logits, gt_cls, loss_mask)
        lovasz_hinge_loss = self.lovasz_hinge_loss(
            pred_logits, gt_cls, loss_mask
        )
        return self.a * bce_loss + self.b * lovasz_hinge_loss


class WildfireMetrics:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.FP = None
        self.TP = None
        self.FN = None
        self.TN = None

    def update(self, pred_prob: Tensor, gt_cls: Tensor, loss_mask: Tensor):
        gt_prob = fire_cls_to_prob(gt_cls).to(torch.float32)
        pred_binary = torch.where(pred_prob > self.threshold, 1.0, 0.0).to(
            torch.float32
        )
        mask = loss_mask
        gt_binary = gt_prob[mask]
        pred_binary = pred_binary[mask]

        def update(cur_val, new_val):
            if cur_val is None:
                return new_val
            return cur_val + new_val

        # True Negatives: Predicted negative (0) and actually negative (0)
        # False Negatives: Predicted negative (0) but actually positive (1)
        # False Positives: Predicted positive (1) but actually negative (0)
        # True Positives: Predicted positive (1) and actually positive (1)
        self.TN = update(self.TN, ((1 - pred_binary) * (1 - gt_binary)).sum())
        self.FN = update(self.FN, ((1 - pred_binary) * gt_binary).sum())
        self.FP = update(self.FP, (pred_binary * (1 - gt_binary)).sum())
        self.TP = update(self.TP, (pred_binary * gt_binary).sum())

    def get(self):
        return dict(
            iou=100 * self.TP / (self.TP + self.FP + self.FN),
            f1=100 * 2 * self.TP / (2 * self.TP + self.FP + self.FN),
            precision=100 * self.TP / (self.TP + self.FP),
            recall=100 * self.TP / (self.TP + self.FN),
        )

    def get_and_reset(self):
        metrics = self.get()
        self.reset()
        return metrics


def save_batch(epoch, batch_idx, **data):
    data = {k: v.detach().cpu() for k, v in data.items()}
    dir_path = os.path.join(config.root_path, "runs", config.name)
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, f"{epoch:02d}_{batch_idx:03d}.pt")
    torch.save(
        data,
        path,
    )
