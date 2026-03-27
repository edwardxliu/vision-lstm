# lstm5_stage1_pretrain_192_sample_ablation_paper.py
# PSWF / VisionLSTM5 - paper-grade training harness (DDP)
#
# What this script adds vs your original:
# - unified DATASET: ImageNet-1K / Tiny-ImageNet
# - optional ImageNet-C evaluation (eval-only or post-train)
# - per-epoch wall-clock + throughput logging (images/s), JSONL metrics dump
# - auto plots (acc vs epoch/step/time) on rank0
# - ablation IDs extended: W3, W3_POOL_ONLY, etc.
# - optional MODEL_KIND: vil (VisionLSTM2) / vit_tiny (minimal ViT-T), plus a stub hook for mambavision
#
# Usage (example, ImageNet-1K stage1 @192):
#   export DATASET=imagenet DATA_ROOT=/path/to/imagenet
#   export ABLATION=W3 DWT_FUSE=add DISABLE_BRANCH=1
#   export IMG_SIZE=192 EPOCHS=200 PER_GPU_BATCH=32 ACCUM_STEPS=1 AMP_DTYPE=bf16
#   torchrun --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py
#
# Tiny-ImageNet (train+val):
#   export DATASET=tiny_imagenet DATA_ROOT=/path/to/tiny-imagenet-200
#   export IMG_SIZE=64 EPOCHS=300 PER_GPU_BATCH=128
#
# ImageNet-C evaluation (eval-only):
#   export MODE=eval_imagenetc IMAGENETC_ROOT=/path/to/imagenet-c
#   export CKPT=/path/to/ema_best.pth
#   torchrun --nproc_per_node=1 lstm5_stage1_pretrain_192_sample_ablation_paper.py

import os
import math
import time
import json
import random
import hashlib
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

import torch.backends.cudnn as cudnn
from torch.amp import autocast, GradScaler


# ----------------- Perf defaults -----------------
cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True


# ----------------- Env helpers -----------------
def env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, None)
    if v is None:
        return default
    v = str(v).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def env_int(name: str, default: int) -> int:
    v = os.environ.get(name, None)
    if v is None or str(v).strip() == "":
        return default
    return int(v)


def env_float(name: str, default: float) -> float:
    v = os.environ.get(name, None)
    if v is None or str(v).strip() == "":
        return default
    return float(v)


def env_str(name: str, default: str) -> str:
    v = os.environ.get(name, None)
    if v is None:
        return default
    return str(v)


def env_list_int(name: str, default: Optional[List[int]] = None, sep: str = ",") -> Optional[List[int]]:
    v = os.environ.get(name, None)
    if v is None or str(v).strip() == "":
        return default
    return [int(x.strip()) for x in str(v).split(sep) if x.strip()]


# ----------------- Reproducibility -----------------
def set_global_seed(seed: int):
    """Set torch / numpy / random seeds for reproducibility. Call once after DDP init."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Avoid nondeterministic algorithms when reproducibility is needed (optional; can slow training)
    # torch.use_deterministic_algorithms(True)


def _worker_init_fn(worker_id: int, base_seed: int):
    """Per-worker RNG seed so that augmentations (RandomCrop, etc.) are deterministic.
    Call with base_seed = data_seed + get_rank() * 10000 so each rank's workers get distinct but reproducible seeds.
    """
    seed = base_seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


# ----------------- DDP helpers -----------------
def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def ddp_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs, flush=True)


# ----------------- Simple datasets -----------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".JPEG")

class TinyImageNetCDataset(Dataset):
    """
    Tiny-ImageNet-C: root/<corruption>/<severity>/<wnid>/*.JPEG
    Labels are mapped using wnids.txt order from tiny-imagenet-200.
    """
    def __init__(self, severity_root: str, wnids_path: str, transform=None):
        self.severity_root = Path(severity_root)
        self.transform = transform

        wnids = [x.strip() for x in open(wnids_path, "r").read().splitlines() if x.strip()]
        self.wnid_to_idx = {w: i for i, w in enumerate(wnids)}

        self.samples = []
        # iterate by wnids.txt order to guarantee deterministic mapping
        for wnid in wnids:
            d = self.severity_root / wnid
            if not d.is_dir():
                continue
            for p in d.rglob("*"):
                if p.suffix.lower() in [".jpeg", ".jpg", ".png"]:
                    self.samples.append((str(p), self.wnid_to_idx[wnid]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

class PathLabelDataset(torch.utils.data.Dataset):
    """A minimal ImageFolder-like dataset that holds (path, label) samples."""
    def __init__(self, samples: List[Tuple[str, int]], classes: List[str], transform=None):
        self.samples = samples
        self.targets = [y for _, y in samples]
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, y


def load_tiny_imagenet(root: str, split: str, transform=None) -> PathLabelDataset:
    """
    root: tiny-imagenet-200 directory (contains train/, val/, wnids.txt)
    """
    import os
    wnids_path = os.path.join(root, "wnids.txt")
    if not os.path.isfile(wnids_path):
        raise RuntimeError(f"Tiny-ImageNet root missing wnids.txt: {wnids_path}")
    with open(wnids_path, "r") as f:
        wnids = [ln.strip() for ln in f if ln.strip()]
    class_to_idx = {w: i for i, w in enumerate(wnids)}

    samples: List[Tuple[str, int]] = []
    if split == "train":
        for w in wnids:
            img_dir = os.path.join(root, "train", w, "images")
            if not os.path.isdir(img_dir):
                continue
            for fn in os.listdir(img_dir):
                if fn.endswith(IMG_EXTS):
                    samples.append((os.path.join(img_dir, fn), class_to_idx[w]))
    elif split == "val":
        ann_path = os.path.join(root, "val", "val_annotations.txt")
        img_dir = os.path.join(root, "val", "images")
        if not os.path.isfile(ann_path):
            raise RuntimeError(f"Tiny-ImageNet val_annotations.txt missing: {ann_path}")
        mapping = {}
        with open(ann_path, "r") as f:
            for ln in f:
                parts = ln.strip().split("\t")
                if len(parts) >= 2:
                    mapping[parts[0]] = parts[1]
        for fn, w in mapping.items():
            p = os.path.join(img_dir, fn)
            if os.path.isfile(p) and w in class_to_idx:
                samples.append((p, class_to_idx[w]))
    else:
        raise ValueError("split must be 'train' or 'val'")
    return PathLabelDataset(samples=samples, classes=wnids, transform=transform)


def stable_hash_u64(s: str, seed: int) -> int:
    """Deterministic per-path hashing for per-class sampling (stable across ranks/machines)."""
    h = hashlib.blake2b(digest_size=8)
    h.update(str(seed).encode("utf-8"))
    h.update(b"::")
    h.update(s.encode("utf-8"))
    return int.from_bytes(h.digest(), byteorder="little", signed=False)


def filter_samples(
    samples: List[Tuple[str, int]],
    keep_class_indices: List[int],
    per_class_cap: int = 0,
    cap_seed: int = 0,
) -> Tuple[List[Tuple[str, int]], int]:
    """
    Filter + remap labels to 0..K-1. Optional per-class cap with stable hashing.
    """
    keep_set = set(keep_class_indices)
    remap = {old_i: new_i for new_i, old_i in enumerate(keep_class_indices)}

    tmp: List[Tuple[str, int]] = []
    for path, y in samples:
        if y in keep_set:
            tmp.append((path, remap[y]))

    if per_class_cap and per_class_cap > 0:
        buckets: List[List[Tuple[int, Tuple[str, int]]]] = [[] for _ in range(len(keep_class_indices))]
        for path, y in tmp:
            hv = stable_hash_u64(path, cap_seed)
            buckets[y].append((hv, (path, y)))
        tmp2: List[Tuple[str, int]] = []
        for b in buckets:
            b.sort(key=lambda t: t[0])
            tmp2.extend([s for _, s in b[:per_class_cap]])
        tmp = tmp2

    return tmp, len(keep_class_indices)


def select_subset_classes(total_classes: int, subset_k: int, seed: int, device: torch.device) -> List[int]:
    """
    Pick K class indices on rank0 then broadcast to all ranks (DDP-safe).
    """
    if subset_k <= 0 or subset_k >= total_classes:
        return list(range(total_classes))

    if is_main_process():
        rng = random.Random(seed)
        keep = list(range(total_classes))
        rng.shuffle(keep)
        keep = keep[:subset_k]
        keep.sort()
        keep_tensor = torch.tensor(keep, device=device, dtype=torch.int64)
    else:
        keep_tensor = torch.empty((subset_k,), device=device, dtype=torch.int64)

    if is_dist():
        dist.broadcast(keep_tensor, src=0)

    keep = keep_tensor.tolist()
    keep.sort()
    return keep


# ----------------- Mixup / CutMix -----------------
def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels, num_classes=num_classes).float()


def smooth_one_hot(onehot: torch.Tensor, label_smoothing: float) -> torch.Tensor:
    if label_smoothing <= 0:
        return onehot
    K = onehot.size(-1)
    return onehot * (1.0 - label_smoothing) + label_smoothing / K


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

    r = random.random()
    if r > prob:
        oh = smooth_one_hot(one_hot(y, num_classes), label_smoothing)
        return x, oh

    use_cutmix = (random.random() < switch_prob) and (cutmix_alpha > 0)
    if use_cutmix:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x2 = x.flip(0)
        x[:, :, bby1:bby2, bbx1:bbx2] = x2[:, :, bby1:bby2, bbx1:bbx2]
        lam_adj = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
        y1 = smooth_one_hot(one_hot(y, num_classes), label_smoothing)
        y2 = smooth_one_hot(one_hot(y.flip(0), num_classes), label_smoothing)
        return x, y1 * lam_adj + y2 * (1.0 - lam_adj)
    else:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        x2 = x.flip(0)
        x = x * lam + x2 * (1.0 - lam)
        y1 = smooth_one_hot(one_hot(y, num_classes), label_smoothing)
        y2 = smooth_one_hot(one_hot(y.flip(0), num_classes), label_smoothing)
        return x, y1 * lam + y2 * (1.0 - lam)


def rand_bbox(size, lam):
    W = size[3]
    H = size[2]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    return -(soft_targets * logp).sum(dim=-1).mean()


# ----------------- EMA -----------------
def create_ema_model(model: nn.Module) -> nn.Module:
    ema = deepcopy_model(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


def deepcopy_model(model: nn.Module) -> nn.Module:
    import copy
    return copy.deepcopy(model)


@torch.no_grad()
def update_ema(model: nn.Module, ema: nn.Module, decay: float):
    msd = model.state_dict()
    esd = ema.state_dict()
    for k, v in esd.items():
        if k not in msd:
            continue
        m = msd[k]
        if not torch.is_tensor(v) or not torch.is_tensor(m):
            continue

        # Long / Bool buffer 直接 copy（比如 num_batches_tracked）
        if not torch.is_floating_point(v):
            v.copy_(m)
            continue

        # 浮点张量做 EMA（对齐 dtype，兼容 bf16/fp16）
        v.mul_(decay).add_(m.to(dtype=v.dtype), alpha=1.0 - decay)

    ema.load_state_dict(esd, strict=False)


# ----------------- Minimal ViT-T (for plug-and-play demos) -----------------
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, attn_drop=0.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViTTiny(nn.Module):
    """
    A minimal ViT-T/16 style model:
      dim=192, depth=12, heads=3, patch=16
    If pswf_embed is provided, it should be a module that returns a feature map (B,C,H',W') before patch embedding.
    """
    def __init__(self, img_size: int, patch_size: int, num_classes: int,
                 dim: int = 192, depth: int = 12, heads: int = 3,
                 drop: float = 0.0, attn_drop: float = 0.0,
                 pswf_embed: Optional[nn.Module] = None,
                 patch_embed: Optional[nn.Module] = None,
                 pswf_gate: Optional[nn.Module] = None,
                 wavelet_warmup_steps: int = 0,
                 wavelet_fuse_mode: str = "add",
                 wavelet_scale_init: float = 0.0):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.dim = dim
        self.pswf_embed = pswf_embed
        # 仅在 ViT + PSWF 下启用的轻量 gate，用 wavelet 特征对 cls token 做微调
        self.pswf_gate = pswf_gate
        # 与 VIL 保持一致：通过可学习的 scale 和可选 warmup 控制小波残差强度
        if pswf_gate is not None:
            self.wavelet_scale = nn.Parameter(torch.tensor(float(wavelet_scale_init)))
            self.wavelet_warmup_steps = int(wavelet_warmup_steps) if wavelet_warmup_steps > 0 else 0
            self.wavelet_fuse_mode = str(wavelet_fuse_mode)
            self.register_buffer("_wavelet_step", torch.tensor(0, dtype=torch.long))
            self.register_buffer("_current_global_step", torch.tensor(-1, dtype=torch.long))
        else:
            self.wavelet_scale = None
            self.wavelet_warmup_steps = 0
            self.wavelet_fuse_mode = "add"

        if patch_embed is None:
            # standard patch embedding from RGB
            self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size, bias=True)
            self.pre_ch = 3
            self.ds_factor = 1
            pe_img = img_size
            pe_patch = patch_size
        else:
            # external patch embedding expects feature map already produced
            self.patch_embed = patch_embed
            self.pre_ch = None
            self.ds_factor = 2  # typical PSWF uses /2 (post-stem)
            pe_img = img_size // self.ds_factor
            pe_patch = patch_size // self.ds_factor

        grid = pe_img // pe_patch
        self.num_patches = grid * grid

        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, 1 + self.num_patches, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

        self.blocks = nn.Sequential(*[ViTBlock(dim, heads, mlp_ratio=4.0, attn_drop=attn_drop, drop=drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def set_wavelet_global_step(self, global_step: int):
        """
        为 ViT + PSWF 提供与 VIL 一致的接口，用于控制小波 warmup 的当前 global_step。
        仅在启用了 pswf_gate 且 wavelet_warmup_steps>0 时生效。
        """
        if self.wavelet_scale is not None and self.wavelet_warmup_steps > 0:
            self._current_global_step.fill_(global_step)

    def forward(self, x):
        gate_vec = None
        if self.pswf_embed is None:
            x = self.patch_embed(x)  # (B, D, H/p, W/p)
        else:
            out = self.pswf_embed(x)
            if isinstance(out, tuple):
                # (main_feat, wav_feat)：主路径 pool-only，gate 只读小波
                feat, wav_feat = out
                if self.pswf_gate is not None:
                    gate_vec = self.pswf_gate(wav_feat)
            else:
                feat = out
                if self.pswf_gate is not None:
                    gate_vec = self.pswf_gate(feat)
            x = self.patch_embed(feat)  # (B, D, H/p, W/p) with p_eff
        # VitPatchEmbed returns (B, H, W, D); Conv2d returns (B, D, H, W)
        if x.ndim == 4 and x.shape[-1] == self.dim and x.shape[1] != self.dim:
            x = x.permute(0, 3, 1, 2).contiguous()  # -> (B, D, H, W)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)

        B = x.size(0)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos[:, : x.size(1)]
        x = self.blocks(x)
        x = self.norm(x)
        cls = x[:, 0]
        # 轻量小波调制：支持可选 warmup 与加性/乘性融合
        if gate_vec is not None and self.wavelet_scale is not None:
            if self.wavelet_warmup_steps > 0 and self.training:
                if self._current_global_step.item() >= 0:
                    global_step = self._current_global_step.item()
                else:
                    global_step = self._wavelet_step.item()
                    self._wavelet_step += 1
                warmup_factor = min(1.0, max(0.0, float(global_step) / self.wavelet_warmup_steps))
            else:
                warmup_factor = 1.0

            effective_scale = self.wavelet_scale * warmup_factor
            gate_vec_t = torch.tanh(gate_vec.to(cls.dtype))
            if self.wavelet_fuse_mode == "multiply":
                cls = cls * (1.0 + effective_scale * gate_vec_t)
            else:
                cls = cls + effective_scale * gate_vec_t
        return self.head(cls)


# ----------------- Ablation config -----------------
def get_ablation_cfg(ablation_id: str, baseline_ablation: str = "A0") -> dict:
    """
    Map an ABLATION id to VisionLSTM2 toggles.
    You can treat W3 as your "PSWF" mainline.

    - W3_POOL_ONLY: post-stem downsample path only, no wavelet in token path, no head residual.
    - W3_TOKENONLY: post-stem concat+1×1 mix (wavelet in tokens), head wavelet residual off.
    - W3_RESIDUALONLY: main path pool-only, head wavelet residual on (DWT only for CLS modulation).
    - W3_BOTH: both token wavelet and head residual (same as W3).
    """
    ablation_id = (ablation_id or "A0").strip().upper()
    baseline_ablation = (baseline_ablation or "A0").strip().upper()

    A = {
        "A0": dict(use_conv_stem=True,  use_dwt=True,  pre_patch_dwt=False, disable_branch=True,  pooling="bilateral_flatten", head_inject_gated=True),
        "A1": dict(use_conv_stem=True,  use_dwt=False, pre_patch_dwt=False, disable_branch=True,  pooling="bilateral_flatten", head_inject_gated=True),
        "A2": dict(use_conv_stem=False, use_dwt=False, pre_patch_dwt=True,  disable_branch=True,  pooling="bilateral_flatten", head_inject_gated=True),
        "A3": dict(use_conv_stem=False, use_dwt=False, pre_patch_dwt=False, disable_branch=True,  pooling="bilateral_flatten", head_inject_gated=True),
    }

    W = {
        # W3: conv stem -> post-stem DWT + pooled conv concat -> 1x1 mix -> patchify + head wavelet residual（等价 W3_BOTH）
        "W3": {**A["A1"], "post_stem_dwt": True, "post_stem_merge": "concat", "disable_branch": True,
               "wavelet_warmup_steps": 0, "wavelet_fuse_mode": "add", "head_wavelet_residual": True},
        "W4": {**A["A1"], "post_stem_dwt": True, "post_stem_merge": "concat", "disable_branch": False},
        "W3_POOL_ONLY": {**A["A1"], "post_stem_dwt": True, "post_stem_merge": "concat", "disable_branch": True, "pool_only": True, "head_wavelet_residual": False},
        # 解耦消融：只开 post-stem concat+1×1，关掉 head residual
        "W3_TOKENONLY": {**A["A1"], "post_stem_dwt": True, "post_stem_merge": "concat", "disable_branch": True,
                         "wavelet_warmup_steps": 0, "wavelet_fuse_mode": "add", "head_wavelet_residual": False},
        # 主路 pool-only，单独开 head wavelet residual（类似 ViT 的 W3_RESIDUAL）
        "W3_RESIDUALONLY": {**A["A1"], "post_stem_dwt": True, "post_stem_merge": "concat", "disable_branch": True,
                            "pool_only": True, "head_wavelet_residual": True, "wavelet_warmup_steps": 0, "wavelet_fuse_mode": "add"},
        # W3_IMPROVED_WARMUP: 改进版 + warmup（前5000步）+ 加性融合
        "W3_IMPROVED_WARMUP": {**A["A1"], "post_stem_dwt": True, "post_stem_merge": "concat", "disable_branch": True,
                                "wavelet_warmup_steps": 5000, "wavelet_fuse_mode": "add", "head_wavelet_residual": True},
    }

    def resolve_base(base_id: str) -> dict:
        base_id = (base_id or "A0").strip().upper()
        if base_id in A:
            cfg = dict(A[base_id])
            cfg.setdefault("post_stem_dwt", False)
            cfg.setdefault("post_stem_merge", "replace")
            cfg.setdefault("pool_only", False)
            return cfg
        if base_id in W:
            cfg = dict(W[base_id])
            cfg.setdefault("head_inject_gated", True)
            cfg.setdefault("pool_only", cfg.get("pool_only", False))
            cfg.setdefault("head_wavelet_residual", True)
            return cfg
        raise KeyError(f"Unknown baseline ablation: {base_id}")

    # Group B/C could be extended similarly; for paper we keep the basics stable.
    if ablation_id in A:
        return resolve_base(ablation_id)
    if ablation_id in W:
        return resolve_base(ablation_id)

    # Fallback: treat as baseline
    return resolve_base(baseline_ablation)


def _infer_num_classes_for_builder() -> int:
    """Infer num_classes from env when building without loading dataset (e.g. for model_compute_lstm5_paper)."""
    v = os.environ.get("NUM_CLASSES")
    if v is not None and str(v).strip() != "":
        try:
            return int(v)
        except ValueError:
            pass
    v = os.environ.get("SUBSET_CLASSES")
    if v is not None and str(v).strip() != "":
        try:
            k = int(v)
            if k > 0:
                return k
        except ValueError:
            pass
    ds = (os.environ.get("DATASET") or "").strip().lower()
    if ds in ("tiny_imagenet", "tiny-imagenet", "tinyimagenet", "tiny_imagenet_200"):
        return 200
    return 1000


def build_model_from_env(num_classes: Optional[int] = None, img_size: Optional[int] = None):
    """
    Build model from env vars (no DDP, no data loaders). Used by model_compute_lstm5_paper.py
    and model_analyse_lstm5_paper.py. Returns (model, cfg_dict); cfg_dict is a small dict
    with keys like model_kind, ablation_id, img_size, num_classes for reporting.
    """
    model_kind = env_str("MODEL_KIND", "vil").lower()
    _img_size = img_size if img_size is not None else env_int("IMG_SIZE", 192)
    _num_classes = num_classes if num_classes is not None else _infer_num_classes_for_builder()
    dim = env_int("DIM", 192)
    depth = env_int("DEPTH", 12)
    feat_ch = env_list_int("FEAT_CH", default=[32, 64, 64]) or [32, 64, 64]
    patch_size = env_int("PATCH_SIZE", 16)
    stride = env_int("STRIDE", patch_size)
    auto_patch_dwt = env_bool("AUTO_PATCH_DWT", True)
    ablation_id = env_str("ABLATION", "W3")
    dwt_fuse = env_str("DWT_FUSE", "add")
    disable_branch_env = env_bool("DISABLE_BRANCH", True)
    drop_path = env_float("DROP_PATH", 0.0)
    drop_path_decay = env_bool("DROP_PATH_DECAY", False)
    legacy_norm = env_bool("LEGACY_NORM", False)
    conv_kind = env_str("CONV_KIND", "2d")
    conv_kernel = env_int("CONV_KERNEL", 3)
    proj_bias = env_bool("PROJ_BIAS", True)
    norm_bias = env_bool("NORM_BIAS", True)

    cfg = dict(model_kind=model_kind, ablation_id=ablation_id, img_size=_img_size, num_classes=_num_classes)

    if model_kind == "vil":
        from vision_lstm5_mod4_paper import VisionLSTM2
        abl_cfg = get_ablation_cfg(ablation_id)
        if "DISABLE_BRANCH" in os.environ:
            abl_cfg["disable_branch"] = bool(disable_branch_env)
        pool_only = bool(abl_cfg.get("pool_only", False))
        dwt_fuse_eff = "none" if pool_only else dwt_fuse
        model = VisionLSTM2(
            dim=dim,
            depth=depth,
            input_shape=(3, _img_size, _img_size),
            output_shape=(_num_classes,),
            mode="classifier",
            pooling=abl_cfg.get("pooling", "bilateral_flatten"),
            drop_path_rate=drop_path,
            drop_path_decay=drop_path_decay,
            legacy_norm=legacy_norm,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel,
            proj_bias=proj_bias,
            norm_bias=norm_bias,
            patch_size=patch_size,
            stride=stride,
            feature_extractor_channels=feat_ch,
            use_conv_stem=abl_cfg.get("use_conv_stem", True),
            use_dwt=abl_cfg.get("use_dwt", False),
            pre_patch_dwt=abl_cfg.get("pre_patch_dwt", False),
            post_stem_dwt=abl_cfg.get("post_stem_dwt", False),
            post_stem_merge=abl_cfg.get("post_stem_merge", "replace"),
            disable_branch=abl_cfg.get("disable_branch", True),
            auto_patch_dwt=auto_patch_dwt,
            dwt_fuse=dwt_fuse_eff,
            wavelet_warmup_steps=int(os.environ["WAVELET_WARMUP_STEPS"]) if os.environ.get("WAVELET_WARMUP_STEPS") else abl_cfg.get("wavelet_warmup_steps", 0),
            wavelet_fuse_mode=os.environ.get("WAVELET_FUSE_MODE") or abl_cfg.get("wavelet_fuse_mode", "multiply"),
            head_wavelet_residual=abl_cfg.get("head_wavelet_residual", True),
            wavelet_scale_init=env_float("WAVELET_SCALE_INIT", 0.0),
        )
        return model, cfg

    if model_kind == "vit_tiny":
        ab_u = (ablation_id or "").strip().upper()
        use_pswf = ab_u.startswith("W3") or ab_u.startswith("W4")
        pool_only = ("POOL_ONLY" in ab_u) or ("POOLONLY" in ab_u)
        if use_pswf:
            from vision_lstm5_mod4_paper import (
                FeatureExtractor, PostStemWaveletMerge, VitPatchEmbed, WaveletGlobalGate,
                StemWithWaveletResidual, DWTPreprocessor,
            )
            vit_wavelet_warmup_steps = 0
            vit_wavelet_fuse_mode = "add"
            vit_wavelet_scale_init = env_float("WAVELET_SCALE_INIT", 0.0)
            if ab_u == "W3_IMPROVED_WARMUP":
                vit_wavelet_warmup_steps = int(os.environ["WAVELET_WARMUP_STEPS"]) if os.environ.get("WAVELET_WARMUP_STEPS") else 5000
                vit_wavelet_fuse_mode = os.environ.get("WAVELET_FUSE_MODE") or "multiply"
            stem = FeatureExtractor(input_channels=3, conv_channels=feat_ch, use_dwt=False, dwt_fuse="none")
            token_only = "TOKENONLY" in ab_u or ab_u == "W3_TOKENONLY"
            use_residual = ("RESIDUAL" in ab_u) or (ab_u == "W3_RESIDUAL")
            if use_residual:
                post_pool_only = PostStemWaveletMerge(channels=stem.final_channels, dwt_fuse="none", merge="concat")
                dwt_module = DWTPreprocessor(channels=stem.final_channels, dwt_fuse="add")
                pswf_embed = StemWithWaveletResidual(stem, post_pool_only, dwt_module)
                main_ch = post_pool_only.out_channels
                pswf_gate = None if token_only else WaveletGlobalGate(in_channels=dwt_module.out_channels, dim=dim)
            else:
                dwt_fuse_eff = "none" if pool_only else dwt_fuse
                post = PostStemWaveletMerge(channels=stem.final_channels, dwt_fuse=dwt_fuse_eff, merge="concat")
                pswf_embed = nn.Sequential(stem, post)
                main_ch = post.out_channels
                pswf_gate = None if (token_only or pool_only) else WaveletGlobalGate(in_channels=main_ch, dim=dim)
            pe_res = (_img_size // 2, _img_size // 2)
            if bool(auto_patch_dwt):
                patch_eff = patch_size // 2
                stride_eff = stride // 2
            else:
                patch_eff = patch_size
                stride_eff = stride
            patch_embed = VitPatchEmbed(
                dim=dim, num_channels=main_ch, resolution=pe_res,
                patch_size=(patch_eff, patch_eff), stride=(stride_eff, stride_eff),
                init_weights="xavier_uniform",
            )
            model = ViTTiny(
                img_size=_img_size, patch_size=patch_size, num_classes=_num_classes,
                dim=dim, depth=depth, heads=max(1, dim // 64),
                pswf_embed=pswf_embed, patch_embed=patch_embed, pswf_gate=pswf_gate,
                wavelet_warmup_steps=vit_wavelet_warmup_steps,
                wavelet_fuse_mode=vit_wavelet_fuse_mode,
                wavelet_scale_init=vit_wavelet_scale_init,
            )
        else:
            model = ViTTiny(
                img_size=_img_size, patch_size=patch_size, num_classes=_num_classes,
                dim=dim, depth=depth, heads=max(1, dim // 64),
            )
        return model, cfg

    if model_kind == "mambavision":
        raise RuntimeError("MODEL_KIND=mambavision is a stub. Provide a builder in-code or import your local implementation.")
    raise ValueError("MODEL_KIND must be vil | vit_tiny | mambavision")


# ----------------- Branch alpha (kept for backward compatibility) -----------------
def get_branch_alpha(epoch: int, start: int, ramp: int, alpha_max: float) -> float:
    if epoch < start:
        return 0.0
    if ramp <= 0:
        return alpha_max
    t = min(1.0, (epoch - start) / float(ramp))
    return alpha_max * t


def try_set_head_alpha(model: nn.Module, alpha: float):
    # VisionLSTM2 has a branch alpha setter; other models ignore
    if hasattr(model, "set_head_alpha"):
        try:
            model.set_head_alpha(alpha)
        except Exception:
            pass


# ----------------- Eval helpers -----------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp_dtype) -> Tuple[float, float]:
    model.eval()
    total, correct = 0, 0
    loss_sum = 0.0
    with torch.inference_mode(), autocast(device_type=device.type, dtype=amp_dtype):
        for imgs, target in loader:
            imgs = imgs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = model(imgs)
            loss = F.cross_entropy(logits, target)
            loss_sum += loss.item() * target.size(0)
            pred = logits.argmax(1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return loss_sum / max(1, total), correct / max(1, total)


@torch.no_grad()
def evaluate_imagenet_c(model: nn.Module, imagenetc_root: str, img_size: int, device: torch.device, amp_dtype,
                        batch_size: int, num_workers: int,
                        dataset_name: str = "imagenet",
                        data_root: str = "",
                        corruptions: Optional[List[str]] = None,
                        severities: Optional[List[int]] = None) -> Dict[str, float]:
    """
    Evaluate ImageNet-C folder layout:
      IMAGENETC_ROOT/<corruption>/<severity>/<class>/*.JPEG
    For tiny_imagenet, labels are mapped using wnids.txt in DATA_ROOT.
    """
    # 1) auto-detect corruptions if not provided
    if corruptions is None:
        corruptions = sorted([
            d for d in os.listdir(imagenetc_root)
            if os.path.isdir(os.path.join(imagenetc_root, d)) and not d.startswith(".")
        ])

    # 2) default severities
    if severities is None:
        severities = [1, 2, 3, 4, 5]

    resize_size = 224 if img_size >= 128 else img_size
    val_tf = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    wnids_path = ""
    if dataset_name == "tiny_imagenet":
        wnids_path = os.path.join(data_root, "wnids.txt")
        if not os.path.isfile(wnids_path):
            raise RuntimeError(f"tiny_imagenet requires wnids.txt at: {wnids_path}")

    results: Dict[str, float] = {}
    all_acc = []

    for c in corruptions:
        accs = []
        for s in severities:
            d = os.path.join(imagenetc_root, c, str(s))
            if not os.path.isdir(d):
                continue

            # 3) critical: correct label mapping for Tiny-ImageNet-C
            if dataset_name == "tiny_imagenet":
                ds = TinyImageNetCDataset(severity_root=d, wnids_path=wnids_path, transform=val_tf)
            else:
                ds = datasets.ImageFolder(root=d, transform=val_tf)

            loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=False)
            _, acc = evaluate(model, loader, device, amp_dtype)
            accs.append(acc)

        if accs:
            m = float(sum(accs) / len(accs))
            results[c] = m
            all_acc.append(m)

    results["mean"] = float(sum(all_acc) / max(1, len(all_acc)))
    return results



# ----------------- Plotting -----------------
def save_json(path: str, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def append_jsonl(path: str, rec: dict):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def plot_metrics(metrics_jsonl: str, out_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        ddp_print(f"[Plot] matplotlib not available, skip plots: {e}")
        return

    # load
    rows = []
    with open(metrics_jsonl, "r") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    if not rows:
        return

    def _plot(xkey, ykey, title, fname):
        xs = [r.get(xkey, None) for r in rows]
        ys = [r.get(ykey, None) for r in rows]
        xs, ys = zip(*[(x, y) for x, y in zip(xs, ys) if (x is not None and y is not None)])
        plt.figure()
        plt.plot(xs, ys)
        plt.xlabel(xkey)
        plt.ylabel(ykey)
        plt.title(title)
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=160)
        plt.close()

    _plot("epoch", "val_acc", "Val Acc vs Epoch", "curve_val_acc_epoch.png")
    _plot("global_step", "val_acc", "Val Acc vs Step", "curve_val_acc_step.png")
    _plot("elapsed_sec", "val_acc", "Val Acc vs Time (s)", "curve_val_acc_time.png")
    _plot("epoch", "train_loss", "Train Loss vs Epoch", "curve_train_loss_epoch.png")
    _plot("epoch", "val_loss", "Val Loss vs Epoch", "curve_val_loss_epoch.png")


# ----------------- Main -----------------
def main():
    # ----- DDP init -----
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if torch.cuda.is_available() else torch.device("cpu")

    # ----- Global seed (torch / numpy / random). DATA_SEED also used for subset/cap later. -----
    data_seed = env_int("DATA_SEED", 1234)
    set_global_seed(data_seed)

    # ----- Mode -----
    mode = env_str("MODE", "train").lower()
    dataset_name = env_str("DATASET", "imagenet").lower()
    data_root = env_str("DATA_ROOT", env_str("IMAGENET_ROOT", ""))
    if not data_root:
        raise RuntimeError("Set DATA_ROOT (or IMAGENET_ROOT for backward compat).")

    # ----- Training hyperparams -----
    img_size = env_int("IMG_SIZE", 192)
    num_epochs = env_int("EPOCHS", 200)
    per_gpu_batch = env_int("PER_GPU_BATCH", 32)
    accum_steps = env_int("ACCUM_STEPS", 1)
    num_workers = env_int("NUM_WORKERS", 8)

    base_lr = env_float("BASE_LR", 2e-4)
    warmup_epochs = env_int("WARMUP_EPOCHS", 5)
    weight_decay = env_float("WEIGHT_DECAY", 0.05)
    clip_grad = env_float("CLIP_GRAD", 1.0)

    # ---- Model/scheduler knobs (kept consistent with your original script) ----
    drop_path = env_float("DROP_PATH", 0.0)
    drop_path_decay = env_bool("DROP_PATH_DECAY", False)
    legacy_norm = env_bool("LEGACY_NORM", False)
    conv_kind = env_str("CONV_KIND", "2d")
    conv_kernel = env_int("CONV_KERNEL", 3)
    proj_bias = env_bool("PROJ_BIAS", True)
    norm_bias = env_bool("NORM_BIAS", True)

    ema_decay = env_float("EMA_DECAY", 0.9997)

    mixup_alpha = env_float("MIXUP_ALPHA", 0.1)
    cutmix_alpha = env_float("CUTMIX_ALPHA", 0.0)
    mixup_prob = env_float("MIXUP_PROB", 0.0)
    switch_prob = env_float("SWITCH_PROB", 0.5)
    label_smooth = env_float("LABEL_SMOOTH", 0.0)

    # ----- Model config -----
    model_kind = env_str("MODEL_KIND", "vil").lower()  # vil | vit_tiny | mambavision
    dim = env_int("DIM", 192)
    depth = env_int("DEPTH", 12)
    feat_ch = env_list_int("FEAT_CH", default=[32, 64, 64]) or [32, 64, 64]
    patch_size = env_int("PATCH_SIZE", 16)
    stride = env_int("STRIDE", patch_size)
    auto_patch_dwt = env_bool("AUTO_PATCH_DWT", True)

    ablation_id = env_str("ABLATION", "W3")
    dwt_fuse = env_str("DWT_FUSE", "add")  # add | gated | LL | concat | none
    disable_branch_env = env_bool("DISABLE_BRANCH", True)  # explicitly override if set

    # Branch schedule (kept)
    BRANCH_START = env_int("BRANCH_START", 1)
    BRANCH_RAMP = env_int("BRANCH_RAMP", 0)
    BRANCH_MAX = env_float("BRANCH_ALPHA_MAX", 0.0)

    # AMP
    amp_dtype_name = env_str("AMP_DTYPE", "bf16").lower()
    amp_autocast_dtype = torch.bfloat16 if amp_dtype_name == "bf16" else torch.float16
    scaler = GradScaler(enabled=(amp_autocast_dtype == torch.float16))

    # ----- Output -----
    out_dir = env_str("OUT_DIR", "./outputs_pswf_paper")
    os.makedirs(out_dir, exist_ok=True)
    tag = env_str("RUN_TAG", f"{dataset_name}_{model_kind}_{ablation_id}_img{img_size}_dim{dim}_d{depth}_ch{'-'.join(map(str,feat_ch))}_{dwt_fuse}")
    run_dir = os.path.join(out_dir, tag)
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    config_path = os.path.join(run_dir, "config.json")
    ckpt_path = os.path.join(run_dir, "ema_best.pth")

    # ----- Build transforms -----
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    if dataset_name == "imagenet":
        # ImageNet train pipeline with 3-Augment (ColorJitter + GaussianBlur)
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0), ratio=(3/4, 4/3)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.0,
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        ])
    else:
        # Tiny-ImageNet style (64x64): keep spatial structure, avoid aggressive random-resize
        train_tf = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    val_tf = transforms.Compose([
        transforms.Resize(224 if dataset_name == "imagenet" else img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # ----- Build datasets -----
    if dataset_name == "imagenet":
        train_base = datasets.ImageFolder(root=os.path.join(data_root, "train"), transform=train_tf)
        val_base   = datasets.ImageFolder(root=os.path.join(data_root, "val"),   transform=val_tf)
        classes = train_base.classes
        train_samples = list(train_base.samples)
        val_samples = list(val_base.samples)
    elif dataset_name in ("tiny_imagenet", "tiny-imagenet"):
        train_base = load_tiny_imagenet(data_root, "train", transform=train_tf)
        val_base   = load_tiny_imagenet(data_root, "val",   transform=val_tf)
        classes = train_base.classes
        train_samples = list(train_base.samples)
        val_samples = list(val_base.samples)
    else:
        raise ValueError("DATASET must be imagenet or tiny_imagenet")

    num_classes = len(classes)

    # ----- Subset class sampling (data_seed set above for global RNG) -----
    subset_k = env_int("SUBSET_CLASSES", 0)
    subset_seed = env_int("SUBSET_SEED", data_seed)

    keep_classes = select_subset_classes(num_classes, subset_k, subset_seed, device=device) if subset_k > 0 else list(range(num_classes))

    train_cap = env_int("TRAIN_SAMPLES_PER_CLASS", 0)
    val_cap   = env_int("VAL_SAMPLES_PER_CLASS", 0)
    cap_seed  = env_int("CAP_SEED", subset_seed)

    train_samples_f, new_k = filter_samples(train_samples, keep_classes, per_class_cap=train_cap, cap_seed=cap_seed)
    val_samples_f, _ = filter_samples(val_samples, keep_classes, per_class_cap=val_cap, cap_seed=cap_seed)

    # remapped class names
    classes_f = [classes[i] for i in keep_classes]
    train_dataset = PathLabelDataset(train_samples_f, classes_f, transform=train_tf)
    val_dataset   = PathLabelDataset(val_samples_f,   classes_f, transform=val_tf)
    num_classes = len(classes_f)

    if is_main_process():
        ddp_print(f"[Data] dataset={dataset_name} train={len(train_dataset)} val={len(val_dataset)} classes={num_classes}")
        ddp_print(f"[Run] dir={run_dir}")

    # ----- Dataloaders -----
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_dist() else None
    # 验证集不再使用 DistributedSampler，避免在仅 rank0 上评估时只看到 1/world_size 的子集
    val_sampler = None
    # Per-worker seed for reproducible augmentations (base_seed per rank so workers are deterministic)
    worker_seed_base = data_seed + (get_rank() * 10000) if is_dist() else data_seed
    train_worker_init = lambda wid: _worker_init_fn(wid, worker_seed_base)
    val_worker_init = lambda wid: _worker_init_fn(wid, worker_seed_base + 5000)  # distinct from train

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_gpu_batch,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=train_worker_init if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=per_gpu_batch,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        worker_init_fn=val_worker_init if num_workers > 0 else None,
    )

    # ----- Build model -----
    if model_kind == "vil":
        from vision_lstm5_mod4_paper import VisionLSTM2
        cfg = get_ablation_cfg(ablation_id)
        # explicit override if user forces
        if "DISABLE_BRANCH" in os.environ:
            cfg["disable_branch"] = bool(disable_branch_env)
        pool_only = bool(cfg.get("pool_only", False))
        dwt_fuse_eff = "none" if pool_only else dwt_fuse

        model = VisionLSTM2(
            dim=dim,
            depth=depth,
            input_shape=(3, img_size, img_size),
            output_shape=(num_classes,),
            mode="classifier",
            pooling=cfg.get("pooling", "bilateral_flatten"),
            drop_path_rate=drop_path,
            drop_path_decay=drop_path_decay,
            legacy_norm=legacy_norm,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel,
            proj_bias=proj_bias,
            norm_bias=norm_bias,
            patch_size=patch_size,
            stride=stride,
            feature_extractor_channels=feat_ch,
            use_conv_stem=cfg.get("use_conv_stem", True),
            use_dwt=cfg.get("use_dwt", False),
            pre_patch_dwt=cfg.get("pre_patch_dwt", False),
            post_stem_dwt=cfg.get("post_stem_dwt", False),
            post_stem_merge=cfg.get("post_stem_merge", "replace"),
            disable_branch=cfg.get("disable_branch", True),
            auto_patch_dwt=auto_patch_dwt,
            dwt_fuse=dwt_fuse_eff,
            wavelet_warmup_steps=int(os.environ["WAVELET_WARMUP_STEPS"]) if os.environ.get("WAVELET_WARMUP_STEPS") else cfg.get("wavelet_warmup_steps", 0),
            wavelet_fuse_mode=os.environ.get("WAVELET_FUSE_MODE") or cfg.get("wavelet_fuse_mode", "multiply"),  # 可用 WAVELET_FUSE_MODE=add|multiply 覆盖
            head_wavelet_residual=cfg.get("head_wavelet_residual", True),
            wavelet_scale_init=env_float("WAVELET_SCALE_INIT", 0.0),  # 非 0 时 add/multiply 才有实际差异，否则 effective_scale=0 两者等价
        )
    elif model_kind == "vit_tiny":
        ab_u = (ablation_id or "").strip().upper()
        use_pswf = ab_u.startswith("W3") or ab_u.startswith("W4")
        pool_only = ("POOL_ONLY" in ab_u) or ("POOLONLY" in ab_u)

        if use_pswf:
            from vision_lstm5_mod4_paper import (
                FeatureExtractor, PostStemWaveletMerge, VitPatchEmbed, WaveletGlobalGate,
                StemWithWaveletResidual, DWTPreprocessor,
            )

            # ViT 小波 warmup / 融合模式（仅在 W3_IMPROVED_WARMUP 等需要时启用）
            vit_wavelet_warmup_steps = 0
            vit_wavelet_fuse_mode = "add"
            vit_wavelet_scale_init = env_float("WAVELET_SCALE_INIT", 0.0)
            if ab_u == "W3_IMPROVED_WARMUP":
                vit_wavelet_warmup_steps = int(os.environ["WAVELET_WARMUP_STEPS"]) if os.environ.get("WAVELET_WARMUP_STEPS") else 5000
                vit_wavelet_fuse_mode = os.environ.get("WAVELET_FUSE_MODE") or "multiply"

            # stem 不做 DWT（避免误读：dwt_fuse 在 use_dwt=False 时无意义）
            stem = FeatureExtractor(input_channels=3, conv_channels=feat_ch, use_dwt=False, dwt_fuse="none")

            # ViT 也支持 tokenization-only：仅改 tokenizer，不做 head-level modulation，便于写清「可插拔 tokenization」收益
            token_only = "TOKENONLY" in ab_u or ab_u == "W3_TOKENONLY"
            # W3_RESIDUAL：主路径 pool-only，小波单独一路 -> gate 调制 CLS，梯度直通
            use_residual = ("RESIDUAL" in ab_u) or (ab_u == "W3_RESIDUAL")
            if use_residual:
                post_pool_only = PostStemWaveletMerge(channels=stem.final_channels, dwt_fuse="none", merge="concat")
                dwt_module = DWTPreprocessor(channels=stem.final_channels, dwt_fuse="add")
                pswf_embed = StemWithWaveletResidual(stem, post_pool_only, dwt_module)
                main_ch = post_pool_only.out_channels
                pswf_gate = None if token_only else WaveletGlobalGate(in_channels=dwt_module.out_channels, dim=dim)
            else:
                # W3_POOL_ONLY => 关闭 wavelet 分支，仅保留 pooled downsample 路径；同时不再对 CLS 做 head-level gate
                dwt_fuse_eff = "none" if pool_only else dwt_fuse
                post = PostStemWaveletMerge(channels=stem.final_channels, dwt_fuse=dwt_fuse_eff, merge="concat")
                pswf_embed = nn.Sequential(stem, post)
                main_ch = post.out_channels
                # 对于 TOKENONLY 或 POOL_ONLY，都不创建 pswf_gate，保证其为「纯 tokenizer」配置
                pswf_gate = None if (token_only or pool_only) else WaveletGlobalGate(in_channels=main_ch, dim=dim)

            pe_res = (img_size // 2, img_size // 2)
            if bool(auto_patch_dwt):
                patch_eff = patch_size // 2
                stride_eff = stride // 2
            else:
                patch_eff = patch_size
                stride_eff = stride

            patch_embed = VitPatchEmbed(
                dim=dim, num_channels=main_ch, resolution=pe_res,
                patch_size=(patch_eff, patch_eff), stride=(stride_eff, stride_eff),
                init_weights="xavier_uniform",
            )
            model = ViTTiny(
                img_size=img_size, patch_size=patch_size, num_classes=num_classes,
                dim=dim, depth=depth, heads=max(1, dim // 64),
                pswf_embed=pswf_embed, patch_embed=patch_embed, pswf_gate=pswf_gate,
                wavelet_warmup_steps=vit_wavelet_warmup_steps,
                wavelet_fuse_mode=vit_wavelet_fuse_mode,
                wavelet_scale_init=vit_wavelet_scale_init,
            )
        else:
            if ab_u not in ("", "A3"):
                ddp_print(f"[Warn] MODEL_KIND=vit_tiny treats ABLATION={ablation_id} as vanilla ViT baseline. "
                        f"Use A3 for baseline, W3/W3_POOL_ONLY for PSWF.")
            model = ViTTiny(img_size=img_size, patch_size=patch_size, num_classes=num_classes,
                            dim=dim, depth=depth, heads=max(1, dim // 64))
    elif model_kind == "mambavision":
        # Stub hook: you can drop your own builder here.
        # Expected: a model that takes (B,3,H,W) and returns logits (B,num_classes)
        raise RuntimeError("MODEL_KIND=mambavision is a stub in this script. Provide a builder in-code or import your local implementation.")
    else:
        raise ValueError("MODEL_KIND must be vil | vit_tiny | mambavision")

    model.to(device)

    # Optional: report params on rank0
    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters())
        ddp_print(f"[Model] kind={model_kind} params={n_params/1e6:.3f}M | dwt_fuse={dwt_fuse} | ablation={ablation_id}")

    # Optional: warm-start training from an existing checkpoint (weights only).
    # 用法示例（继续训练 ImageNet-1K）：
    #   export MODE=train EPOCHS=100
    #   export RESUME_CKPT=outputs_pswf_paper/in1k192_vil_W3_poolonly_ch32_reg/ema_best.pth
    #   torchrun --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py
    #
    # 注意：这里只加载权重到当前 model，不恢复 optimizer / scheduler / epoch 号，
    # 相当于以 RESUME_CKPT 作为新的初始化再跑一套 schedule。
    resume_ckpt = env_str("RESUME_CKPT", "").strip()
    if mode == "train" and resume_ckpt:
        if os.path.isfile(resume_ckpt):
            ddp_print(f"[Resume] Loading weights from RESUME_CKPT={resume_ckpt}")
            sd = torch.load(resume_ckpt, map_location="cpu")
            # 兼容纯 state_dict 或 {model: state_dict} 这两种常见格式
            if isinstance(sd, dict) and not any(torch.is_tensor(v) for v in sd.values()):
                state = sd
            elif isinstance(sd, dict) and any(torch.is_tensor(v) for v in sd.values()):
                # 如果字典里本身就是 tensor map，当作 state_dict 用
                state = sd
            else:
                state = sd
            missing, unexpected = model.load_state_dict(state, strict=False)
            ddp_print(f"[Resume] loaded with missing={len(missing)} unexpected={len(unexpected)}")
        else:
            ddp_print(f"[Resume] RESUME_CKPT not found: {resume_ckpt} (train from scratch)")

    # ----- Load checkpoint for eval modes (no training) -----
    if mode.startswith("eval"):
        ckpt = env_str("CKPT", "")
        if not ckpt:
            # 如果没有显式指定 CKPT，则优先尝试当前 run_dir 下的 ema_best.pth
            if os.path.isfile(ckpt_path):
                ckpt = ckpt_path
                ddp_print(f"[Eval] CKPT not set, fallback to {ckpt}")
            else:
                raise RuntimeError("MODE=eval* 需要设置 CKPT=/path/to/checkpoint，或者保证当前运行目录下存在 ema_best.pth。")
        sd = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(sd, strict=False)
        model.eval()
        if mode == "eval":
            loss, acc = evaluate(model, val_loader, device, amp_autocast_dtype)
            ddp_print(f"[Eval] val_loss={loss:.4f} acc={acc:.4f}")
            # 同时把一次性评估结果写到 JSON 文件里，方便之后查看
            if is_main_process():
                eval_path = os.path.join(run_dir, "eval_val.json")
                save_json(eval_path, {"val_loss": float(loss), "val_acc": float(acc)})
        elif mode == "eval_imagenetc":
            imagenetc_root = env_str("IMAGENETC_ROOT", "")
            if not imagenetc_root:
                raise RuntimeError("Set IMAGENETC_ROOT for ImageNet-C evaluation.")
            res = evaluate_imagenet_c(
                model, imagenetc_root, img_size, device, amp_autocast_dtype, 
                batch_size=per_gpu_batch, num_workers=num_workers, 
                dataset_name=dataset_name, data_root=data_root)
            ddp_print(f"[Eval] ImageNet-C mean acc={res.get('mean', 0.0):.4f}")
            if is_main_process():
                save_json(os.path.join(run_dir, "imagenet_c.json"), res)
        if is_dist():
            dist.barrier()
            dist.destroy_process_group()
        return

    # ----- DDP wrap -----
    if is_dist():
        model = DDP(model, device_ids=[device.index], find_unused_parameters=False)

    # ----- Optimizer -----
    # Simple AdamW; if you want branch-specific LR/WD, extend here.
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    updates_per_epoch = max(1, math.ceil(len(train_loader) / max(1, accum_steps)))
    num_training_steps = num_epochs * updates_per_epoch
    warmup_steps = warmup_epochs * updates_per_epoch

    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    sch1 = LinearLR(optimizer, start_factor=0.1, total_iters=max(1, warmup_steps))
    sch2 = CosineAnnealingLR(optimizer, T_max=max(1, num_training_steps - warmup_steps), eta_min=base_lr * 3e-2)
    scheduler = SequentialLR(optimizer, schedulers=[sch1, sch2], milestones=[warmup_steps])

    # EMA (keep a non-DDP copy)
    if model_kind == "vil" and is_dist():
        ema_model = create_ema_model(model.module).to(device)
    else:
        ema_model = create_ema_model(model.module if isinstance(model, DDP) else model).to(device)

    # Save config
    if is_main_process():
        cfg_dump = dict(
            mode=mode, dataset=dataset_name, data_root=data_root,
            model_kind=model_kind, ablation=ablation_id, dwt_fuse=dwt_fuse,
            img_size=img_size, epochs=num_epochs, per_gpu_batch=per_gpu_batch, accum_steps=accum_steps,
            global_batch=per_gpu_batch * get_world_size() * accum_steps,
            dim=dim, depth=depth, feat_ch=feat_ch, patch_size=patch_size, stride=stride, auto_patch_dwt=auto_patch_dwt,
            base_lr=base_lr, warmup_epochs=warmup_epochs, weight_decay=weight_decay,
            mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, mixup_prob=mixup_prob, label_smooth=label_smooth,
            ema_decay=ema_decay,
        )
        save_json(config_path, cfg_dump)

        # reset metrics file
        if os.path.isfile(metrics_path):
            os.remove(metrics_path)

    best_acc = 0.0
    global_step = 0
    train_start = time.time()

    # Optional: stop when reaching target acc (paper: time-to-acc)
    stop_at_acc = env_float("STOP_AT_ACC", 0.0)
    min_epochs = env_int("MIN_EPOCHS", 0)

    for epoch in range(1, num_epochs + 1):
        epoch_t0 = time.time()

        # branch alpha schedule
        if isinstance(model, DDP):
            base_model = model.module
        else:
            base_model = model
        if getattr(base_model, "disable_branch", False):
            alpha = 0.0
        else:
            alpha = get_branch_alpha(epoch, BRANCH_START, BRANCH_RAMP, BRANCH_MAX)

        try_set_head_alpha(base_model, alpha)
        try_set_head_alpha(ema_model, alpha)

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        soft_acc_hist = []
        it_t0 = time.time()

        if is_main_process():
            ddp_print(f"\n[Train] Epoch {epoch}/{num_epochs} | branch_alpha={alpha:.3e}")

        for it, (imgs, target) in enumerate(train_loader, 1):
            imgs = imgs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            imgs_mixed, soft_targets = mixup_cutmix(
                imgs, target,
                num_classes=num_classes,
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                prob=mixup_prob,
                switch_prob=switch_prob,
                label_smoothing=label_smooth,
            )

            with autocast(device_type=device.type, dtype=amp_autocast_dtype):
                logits = model(imgs_mixed)
                loss = soft_cross_entropy(logits, soft_targets) / accum_steps

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if it % accum_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                if clip_grad and clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                update_ema(base_model, ema_model, ema_decay)
                scheduler.step()
                global_step += 1
                
                # 改进1: 设置 wavelet warmup 的 global_step（如果启用了 warmup）
                if hasattr(base_model, 'set_wavelet_global_step'):
                    base_model.set_wavelet_global_step(global_step)
                if hasattr(ema_model, 'set_wavelet_global_step'):
                    ema_model.set_wavelet_global_step(global_step)

            running_loss += loss.item() * accum_steps

            with torch.no_grad():
                pred = logits.argmax(1)
                soft_acc = soft_targets.gather(1, pred.unsqueeze(1)).squeeze(1).mean().item()
                soft_acc_hist.append(soft_acc)

            log_every = env_int("LOG_EVERY", 100)
            if is_main_process() and (it % log_every == 0):
                dt = max(1e-6, time.time() - it_t0)
                it_t0 = time.time()
                avg_soft = float(sum(soft_acc_hist) / max(1, len(soft_acc_hist)))
                lr0 = scheduler.get_last_lr()[0]
                # approximate global throughput
                imgs_seen = per_gpu_batch * get_world_size() * log_every
                ips = imgs_seen / dt
                ddp_print(f"  iter {it:5d}/{len(train_loader)} | loss {loss.item()*accum_steps:.4f} | soft_acc {avg_soft:.3f} | lr {lr0:.2e} | {ips:.0f} img/s")

        # Validation on rank0 (EMA), then broadcast
        val_loss_g, val_acc_g = 0.0, 0.0
        if is_main_process():
            val_loss_g, val_acc_g = evaluate(ema_model, val_loader, device, amp_autocast_dtype)

        if is_dist():
            metrics = torch.tensor([val_loss_g, val_acc_g], device=device, dtype=torch.float32)
            dist.broadcast(metrics, src=0)
            val_loss_g, val_acc_g = float(metrics[0].item()), float(metrics[1].item())

        epoch_sec = time.time() - epoch_t0
        elapsed = time.time() - train_start
        train_loss_epoch = running_loss / max(1, len(train_loader))
        train_soft_acc = float(sum(soft_acc_hist) / max(1, len(soft_acc_hist)))
        lr0 = scheduler.get_last_lr()[0]

        # record
        if is_main_process():
            rec = dict(
                epoch=epoch,
                global_step=global_step,
                lr=lr0,
                train_loss=train_loss_epoch,
                train_soft_acc=train_soft_acc,
                val_loss=val_loss_g,
                val_acc=val_acc_g,
                epoch_sec=epoch_sec,
                elapsed_sec=elapsed,
                world_size=get_world_size(),
                per_gpu_batch=per_gpu_batch,
                accum_steps=accum_steps,
            )
            append_jsonl(metrics_path, rec)

            ddp_print(f"[Epoch {epoch}] Train loss={train_loss_epoch:.4f}, soft_acc={train_soft_acc:.4f}")
            ddp_print(f"[Epoch {epoch}] Val   loss={val_loss_g:.4f}, acc={val_acc_g:.4f} | epoch_sec={epoch_sec:.1f} | elapsed={elapsed/3600:.2f}h")

            if val_acc_g > best_acc:
                best_acc = val_acc_g
                torch.save(ema_model.state_dict(), ckpt_path)
                ddp_print(f"  🌟 New best saved: {ckpt_path} (acc={best_acc:.4f})")

        # early stop on acc
        if stop_at_acc > 0 and epoch >= max(1, min_epochs) and val_acc_g >= stop_at_acc:
            if is_main_process():
                ddp_print(f"[Stop] Reached STOP_AT_ACC={stop_at_acc:.4f} at epoch={epoch}.")
            break

    # Optional: ImageNet-C post-train evaluation on best ckpt (rank0)
    if env_bool("EVAL_IMAGENETC", False) and is_main_process():
        imagenetc_root = env_str("IMAGENETC_ROOT", "")
        if imagenetc_root and os.path.isdir(imagenetc_root):
            best_sd = torch.load(ckpt_path, map_location="cpu")
            if isinstance(model, DDP):
                base_model = model.module
            else:
                base_model = model
            base_model.load_state_dict(best_sd, strict=False)
            base_model.eval()
            res = evaluate_imagenet_c(
                model, imagenetc_root, img_size, device, amp_autocast_dtype, 
                batch_size=per_gpu_batch, num_workers=num_workers, 
                dataset_name=dataset_name, data_root=data_root)
            save_json(os.path.join(run_dir, "imagenet_c.json"), res)
            ddp_print(f"[ImageNet-C] mean acc={res.get('mean', 0.0):.4f}")

    # Plots
    if is_main_process() and env_bool("PLOT", True):
        plot_metrics(metrics_path, run_dir)

    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
