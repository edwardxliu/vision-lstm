from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# ----------------- Mixup / CutMix -----------------
def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels, num_classes=num_classes).float()


def smooth_one_hot(onehot: torch.Tensor, label_smoothing: float) -> torch.Tensor:
    if label_smoothing <= 0:
        return onehot
    k = onehot.size(-1)
    return onehot * (1.0 - label_smoothing) + label_smoothing / k


def rand_bbox(size, lam):
    w = size[3]
    h = size[2]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    return bbx1, bby1, bbx2, bby2


def mixup_cutmix(
    x: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    mixup_alpha: float,
    cutmix_alpha: float,
    prob: float,
    switch_prob: float,
    label_smoothing: float,
):
    """
    Returns: mixed_x, soft_targets
    """
    if prob <= 0 or (mixup_alpha <= 0 and cutmix_alpha <= 0):
        oh = smooth_one_hot(one_hot(y, num_classes), label_smoothing)
        return x, oh

    r = np.random.rand()
    if r > prob:
        oh = smooth_one_hot(one_hot(y, num_classes), label_smoothing)
        return x, oh

    use_cutmix = (np.random.rand() < switch_prob) and (cutmix_alpha > 0)
    if use_cutmix:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x2 = x.flip(0)
        x_out = x.clone()
        x_out[:, :, bby1:bby2, bbx1:bbx2] = x2[:, :, bby1:bby2, bbx1:bbx2]
        lam_adj = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
        y1 = smooth_one_hot(one_hot(y, num_classes), label_smoothing)
        y2 = smooth_one_hot(one_hot(y.flip(0), num_classes), label_smoothing)
        return x_out, y1 * lam_adj + y2 * (1.0 - lam_adj)


    lam = np.random.beta(mixup_alpha, mixup_alpha)
    x2 = x.flip(0)
    x = x * lam + x2 * (1.0 - lam)
    y1 = smooth_one_hot(one_hot(y, num_classes), label_smoothing)
    y2 = smooth_one_hot(one_hot(y.flip(0), num_classes), label_smoothing)
    return x, y1 * lam + y2 * (1.0 - lam)


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    return -(soft_targets * logp).sum(dim=-1).mean()


# ----------------- EMA -----------------
def deepcopy_model(model: nn.Module) -> nn.Module:
    import copy

    return copy.deepcopy(model)


def create_ema_model(model: nn.Module) -> nn.Module:
    ema = deepcopy_model(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


@torch.no_grad()
def update_ema(model: nn.Module, ema: nn.Module, decay: float) -> None:
    # state_dict() returns references to the live tensors, so in-place updates
    # on `v` mutate `ema`'s parameters/buffers directly. No load_state_dict needed.
    msd = model.state_dict()
    esd = ema.state_dict()
    for k, v in esd.items():
        if k not in msd:
            continue
        m = msd[k]
        if not torch.is_tensor(v) or not torch.is_tensor(m):
            continue

        # Non-floating buffers (e.g. num_batches_tracked) are copied directly.
        if not torch.is_floating_point(v):
            v.copy_(m)
            continue

        # Floating tensors do EMA (align dtype to be bf16/fp16-safe).
        v.mul_(decay).add_(m.to(dtype=v.dtype), alpha=1.0 - decay)


# ----------------- Branch alpha helpers (backward compat) -----------------
def get_branch_alpha(epoch: int, start: int, ramp: int, alpha_max: float) -> float:
    if epoch < start:
        return 0.0
    if ramp <= 0:
        return alpha_max
    t = min(1.0, (epoch - start) / float(ramp))
    return alpha_max * t


def try_set_head_alpha(model: nn.Module, alpha: float) -> None:
    # VisionLSTM2 has a branch alpha setter; other models ignore.
    if hasattr(model, "set_head_alpha"):
        try:
            model.set_head_alpha(alpha)
        except Exception:
            pass

