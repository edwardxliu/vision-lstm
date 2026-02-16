# lstm5_stage1_pretrain_192_sample_ablation.py
# VisionLSTM5 (vision_lstm5_mod2.py) - ImageNet-1K @192 (DDP) with class-subset sampling + ablations
#
# Key features (adapted from your lstm6_stage1_pretrain_192_sample.py):
# - torch.distributed DDP (nccl), rank0 validation, AMP(bf16/fp16) stable (GradScaler only for fp16)
# - DDP-safe class-subset sampling (broadcasted from rank0)
# - Optional per-class image cap using deterministic hashing (no broadcast needed)
# - Mixup/CutMix + label smoothing, EMA, warmup+cosine LR
# - Ablation switch via env var ABLATION (A0..A3 / B0..B2 / C0..C1)
#
# Usage (example):
#   export IMAGENET_ROOT=/path/to/imagenet
#   export ABLATION=A0 SUBSET_CLASSES=150 SUBSET_SEED=1234 EPOCHS=100 PER_GPU_BATCH=128
#   torchrun --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation.py
#
# Notes:
# - This script expects "train" and "val" folders under IMAGENET_ROOT, ImageFolder layout.
# - Put vision_lstm5_mod2.py in the same directory (or in PYTHONPATH).

import os
import math
import random
import hashlib
from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
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
    if v is None:
        return default
    items = [s.strip() for s in str(v).split(sep)]
    items = [s for s in items if s]
    if not items:
        return default
    return [int(x) for x in items]


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def is_main_process() -> bool:
    return get_rank() == 0


# ----------------- DDP setup -----------------
def ddp_setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ================= Subset: random classes (DDP-safe) =================
def select_subset_classes(num_total_classes: int, subset_k: int, seed: int, device) -> List[int]:
    """Rank0 chooses a sorted list of class indices, broadcasts to all ranks."""
    if subset_k <= 0 or subset_k >= num_total_classes:
        return list(range(num_total_classes))

    rank = get_rank()

    if rank == 0:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        perm = torch.randperm(num_total_classes, generator=g)
        keep = perm[:subset_k].tolist()
        keep.sort()
        keep_len = torch.tensor([subset_k], dtype=torch.int64, device=device)
        keep_tensor = torch.tensor(keep, dtype=torch.int64, device=device)
    else:
        keep_len = torch.zeros(1, dtype=torch.int64, device=device)
        keep_tensor = torch.empty(subset_k, dtype=torch.int64, device=device)

    if is_dist():
        dist.broadcast(keep_len, src=0)
        k = int(keep_len.item())
        if rank != 0:
            keep_tensor = torch.empty(k, dtype=torch.int64, device=device)
        dist.broadcast(keep_tensor, src=0)

    keep = keep_tensor.tolist()
    keep.sort()
    return keep


def stable_hash_u64(s: str, seed: int) -> int:
    """Deterministic per-path hashing for per-class sampling, stable across ranks/machines."""
    h = hashlib.blake2b(digest_size=8)
    h.update(str(seed).encode("utf-8"))
    h.update(b"::")
    h.update(s.encode("utf-8"))
    return int.from_bytes(h.digest(), byteorder="little", signed=False)


def filter_imagefolder_inplace(
    ds: datasets.ImageFolder,
    keep_class_indices: List[int],
    per_class_cap: int = 0,
    cap_seed: int = 0,
) -> int:
    """
    In-place filter ImageFolder by class indices, remap labels to 0..K-1.
    Optional: per_class_cap>0 keeps at most N samples per class using deterministic hashing.
    Returns: new number of classes K.
    """
    keep_set = set(keep_class_indices)
    remap = {old_i: new_i for new_i, old_i in enumerate(keep_class_indices)}

    # 1) filter & remap
    tmp_samples: List[Tuple[str, int]] = []
    for path, y in ds.samples:
        if y in keep_set:
            tmp_samples.append((path, remap[y]))

    # 2) optional per-class cap
    if per_class_cap and per_class_cap > 0:
        buckets: List[List[Tuple[int, Tuple[str, int]]]] = [[] for _ in range(len(keep_class_indices))]
        for path, y in tmp_samples:
            hv = stable_hash_u64(path, cap_seed)
            buckets[y].append((hv, (path, y)))
        new_samples: List[Tuple[str, int]] = []
        for y in range(len(buckets)):
            items = buckets[y]
            items.sort(key=lambda x: x[0])
            take = items[:per_class_cap]
            new_samples.extend([t[1] for t in take])
        tmp_samples = new_samples

    ds.samples = tmp_samples
    ds.imgs = tmp_samples
    ds.targets = [y for _, y in tmp_samples]

    old_classes = ds.classes
    new_classes = [old_classes[i] for i in keep_class_indices]
    ds.classes = new_classes
    ds.class_to_idx = {cls_name: i for i, cls_name in enumerate(new_classes)}

    return len(new_classes)


# ----------------- EMA utils -----------------
def create_ema_model(model: torch.nn.Module) -> torch.nn.Module:
    ema = deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    ema.eval()
    return ema


@torch.no_grad()
def update_ema(model: torch.nn.Module, ema_model: torch.nn.Module, decay: float):
    msd = model.state_dict()
    esd = ema_model.state_dict()
    for k, v in esd.items():
        if k not in msd:
            continue
        src = msd[k]
        if not torch.is_floating_point(v) or not torch.is_floating_point(src):
            v.copy_(src)
        else:
            v.copy_(v * decay + src.detach() * (1.0 - decay))


# ----------------- Mixup / CutMix / LS -----------------
def rand_bbox(W, H, lam):
    cut_rat = (1.0 - lam) ** 0.5
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def one_hot_with_label_smoothing(targets, num_classes, smoothing):
    bs = targets.size(0)
    with torch.no_grad():
        y = torch.zeros(bs, num_classes, device=targets.device)
        y.scatter_(1, targets.unsqueeze(1), 1.0)
        if smoothing > 0.0:
            y = y * (1.0 - smoothing) + smoothing / num_classes
    return y


def mixup_cutmix(
    x, targets,
    num_classes,
    mixup_alpha=0.2,
    cutmix_alpha=0.8,
    prob=0.8,
    switch_prob=0.5,
    label_smoothing=0.1,
):
    if prob <= 0.0 or (mixup_alpha <= 0.0 and cutmix_alpha <= 0.0):
        soft_targets = one_hot_with_label_smoothing(targets, num_classes, label_smoothing)
        return x, soft_targets

    bs = x.size(0)
    device = x.device
    y = one_hot_with_label_smoothing(targets, num_classes, label_smoothing)

    if np.random.rand() > prob:
        return x, y

    use_cutmix = (np.random.rand() < switch_prob) and (cutmix_alpha > 0.0)
    perm = torch.randperm(bs, device=device)

    if use_cutmix:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        _, _, H, W = x.size()
        x1, y1, x2, y2 = rand_bbox(W, H, lam)

        x_mixed = x.clone()
        x_mixed[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]

        area = (x2 - x1) * (y2 - y1)
        lam_adj = 1.0 - float(area) / float(W * H)
        y_mixed = y * lam_adj + y[perm] * (1.0 - lam_adj)
        return x_mixed, y_mixed
    else:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        x_mixed = x * lam + x[perm] * (1.0 - lam)
        y_mixed = y * lam + y[perm] * (1.0 - lam)
        return x_mixed, y_mixed


def soft_cross_entropy(pred, soft_targets):
    log_probs = F.log_softmax(pred, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


# ----------------- Branch alpha schedule -----------------
def get_branch_alpha(epoch: int, start_epoch: int, ramp_epochs: int, alpha_max: float) -> float:
    if alpha_max <= 0.0:
        return 0.0
    if epoch < start_epoch:
        return 0.0
    t = min(1.0, (epoch - start_epoch) / max(1, ramp_epochs))
    return alpha_max * t


def try_set_head_alpha(model: torch.nn.Module, alpha: float):
    try:
        ha = getattr(model, "head_adapter", None)
        if ha is None:
            return
        if hasattr(ha, "alpha"):
            with torch.no_grad():
                ha.alpha.fill_(alpha)
            ha.alpha.requires_grad_(False)
    except Exception:
        pass


# ----------------- Optim param grouping -----------------
def build_param_groups(model: torch.nn.Module, base_lr: float, weight_decay: float,
                       branch_lr_scale: float = 1.0):
    """4 groups: main_decay/main_no_decay/branch_decay/branch_no_decay."""
    no_wd = set()
    if hasattr(model, "no_weight_decay"):
        try:
            no_wd = set(model.no_weight_decay())
        except Exception:
            no_wd = set()

    for n, _p in model.named_parameters():
        if n.endswith(".bias"):
            no_wd.add(n)

    main_decay, main_no_decay = [], []
    branch_decay, branch_no_decay = [], []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_branch = (n.startswith("feature_extractor_branch") or n.startswith("head_adapter"))
        is_no_wd = (n in no_wd)

        if is_branch:
            (branch_no_decay if is_no_wd else branch_decay).append(p)
        else:
            (main_no_decay if is_no_wd else main_decay).append(p)

    groups = []
    if main_decay:
        groups.append({"params": main_decay, "lr": base_lr, "weight_decay": weight_decay, "is_branch": False, "is_no_wd": False})
    if main_no_decay:
        groups.append({"params": main_no_decay, "lr": base_lr, "weight_decay": 0.0, "is_branch": False, "is_no_wd": True})
    if branch_decay:
        groups.append({"params": branch_decay, "lr": base_lr * branch_lr_scale, "weight_decay": 0.0, "is_branch": True, "is_no_wd": False})
    if branch_no_decay:
        groups.append({"params": branch_no_decay, "lr": base_lr * branch_lr_scale, "weight_decay": 0.0, "is_branch": True, "is_no_wd": True})
    return groups


# ----------------- Ablation config -----------------
def get_ablation_cfg(ablation_id: str, baseline_ablation: str = "A0") -> dict:
    """
    Map an ABLATION id to model toggles.

    Supports:
      - A-group: A0..A3 (original stem/patch-DWT topology ablations)
      - W-group: W1/W2 (new: post-stem DWT downsample after conv stem, before PatchEmbed)
      - B-group: B0/B1/B2 (branch + gated injection on top of a chosen BASELINE_ABLATION)
      - C-group: C0/C1 (pooling choices on top of BASELINE_ABLATION; does NOT force branch on/off)

    Key improvement vs earlier version:
      - BASELINE_ABLATION can be A* or W*.
      - C-group no longer silently enables the branch (so pooling can be isolated if desired).
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
        # W1/W2 are defined as variants of A1 (RGB conv stem) + post-stem DWT downsample.
        "W1": {**A["A1"], "post_stem_dwt": True, "disable_branch": True},
        "W2": {**A["A1"], "post_stem_dwt": True, "disable_branch": False},
        # W3/W4: keep pooled conv features and concat with DWT(x), then 1x1 mix -> C
        "W3": {**A["A1"], "post_stem_dwt": True, "post_stem_merge": "concat", "disable_branch": True},
        "W4": {**A["A1"], "post_stem_dwt": True, "post_stem_merge": "concat", "disable_branch": False},
    }

    def resolve_base(base_id: str) -> dict:
        base_id = (base_id or "A0").strip().upper()
        if base_id in A:
            cfg = dict(A[base_id])
            cfg.setdefault("post_stem_dwt", False)
            return cfg
        if base_id in W:
            cfg = dict(W[base_id])
            cfg.setdefault("head_inject_gated", True)
            return cfg
        # fallback
        cfg = dict(A["A0"])
        cfg.setdefault("post_stem_dwt", False)
        return cfg

    # Base config that B/C will be applied on top of
    base = resolve_base(baseline_ablation)

    # Direct mappings
    if ablation_id in A:
        cfg = dict(A[ablation_id])
        cfg.setdefault("post_stem_dwt", False)
        return cfg
    if ablation_id in W:
        cfg = dict(W[ablation_id])
        cfg.setdefault("head_inject_gated", True)
        return cfg

    # B-group: branch / gated injection (applied on base)
    if ablation_id == "B0":
        cfg = dict(base)
        cfg.update(disable_branch=False, head_inject_gated=True)
        return cfg
    if ablation_id == "B1":
        cfg = dict(base)
        cfg.update(disable_branch=False, head_inject_gated=False)
        return cfg
    if ablation_id == "B2":
        cfg = dict(base)
        cfg.update(disable_branch=True)
        return cfg

    # C-group: pooling choices (applied on base; keep branch state unchanged)
    if ablation_id == "C0":
        cfg = dict(base)
        cfg.update(pooling="bilateral_flatten")
        return cfg
    if ablation_id == "C1":
        cfg = dict(base)
        cfg.update(pooling="attn")
        return cfg

    # Optional: pooling tests while FORCING branch on/off (if you want to separate interactions)
    if ablation_id == "C2":  # branch ON + bilateral_flatten
        cfg = dict(base)
        cfg.update(disable_branch=False, pooling="bilateral_flatten")
        return cfg
    if ablation_id == "C3":  # branch ON + attn
        cfg = dict(base)
        cfg.update(disable_branch=False, pooling="attn")
        return cfg
    if ablation_id == "C4":  # branch OFF + bilateral_flatten
        cfg = dict(base)
        cfg.update(disable_branch=True, pooling="bilateral_flatten")
        return cfg
    if ablation_id == "C5":  # branch OFF + attn
        cfg = dict(base)
        cfg.update(disable_branch=True, pooling="attn")
        return cfg

    return resolve_base("A0")





# ----------------- Main -----------------
def main():
    ddp_setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    # rank-independent seed for dataset sampling; rank-dependent for training randomness
    data_seed = env_int("DATA_SEED", 1234)
    train_seed = env_int("SEED", 42) + get_rank()
    setup_seed(train_seed)

    # AMP dtype
    amp_dtype = env_str("AMP_DTYPE", "bf16").lower()
    amp_autocast_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    scaler = GradScaler("cuda", enabled=(amp_autocast_dtype == torch.float16))

    # Data
    img_size = env_int("IMG_SIZE", 192)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    data_dir = env_str("IMAGENET_ROOT", "/home/omnisky/Public/edward/workspace/data/imagenet_dataset")
    if not data_dir:
        raise RuntimeError("Please set IMAGENET_ROOT=/path/to/imagenet (with train/val subfolders).")

    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=train_tf)
    val_dataset   = datasets.ImageFolder(root=os.path.join(data_dir, "val"),   transform=val_tf)

    if is_main_process():
        print(f"[Data] train={len(train_dataset)} val={len(val_dataset)} classes={len(train_dataset.classes)}", flush=True)

    # Subset classes
    subset_k = env_int("SUBSET_CLASSES", 150)
    subset_seed = env_int("SUBSET_SEED", data_seed)

    if subset_k > 0:
        total_classes = len(train_dataset.classes)
        keep_classes = select_subset_classes(total_classes, subset_k, subset_seed, device=device)

        # Optional per-class cap (train only by default)
        train_cap = env_int("TRAIN_SAMPLES_PER_CLASS", 0)
        val_cap   = env_int("VAL_SAMPLES_PER_CLASS", 0)
        cap_seed  = env_int("CAP_SEED", subset_seed)

        new_k_train = filter_imagefolder_inplace(train_dataset, keep_classes, per_class_cap=train_cap, cap_seed=cap_seed)
        new_k_val   = filter_imagefolder_inplace(val_dataset, keep_classes, per_class_cap=val_cap,   cap_seed=cap_seed)
        assert new_k_train == new_k_val, "train/val subset class count mismatch"
        num_classes = new_k_train

        if is_main_process():
            print(f"[Subset] keep {num_classes}/{total_classes} classes seed={subset_seed} | train={len(train_dataset)} val={len(val_dataset)}", flush=True)
            if train_cap > 0 or val_cap > 0:
                print(f"[Cap] train_cap={train_cap} val_cap={val_cap} cap_seed={cap_seed}", flush=True)
    else:
        num_classes = len(train_dataset.classes)

    # Loader
    per_gpu_bs = env_int("PER_GPU_BATCH", 128)
    num_workers = env_int("NUM_WORKERS", 8)

    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=per_gpu_bs,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    if is_main_process():
        val_loader = DataLoader(
            val_dataset,
            batch_size=per_gpu_bs,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        val_loader = None

    # Model
    from vision_lstm5_mod4 import VisionLSTM2

    ablation_id = env_str("ABLATION", "A0").strip().upper()
    baseline_ablation = env_str("BASELINE_ABLATION", "A0").strip().upper()
    cfg = get_ablation_cfg(ablation_id, baseline_ablation)

    # Allow env overrides for any cfg fields
    if os.environ.get("POOLING", ""):
        cfg["pooling"] = env_str("POOLING", cfg["pooling"])
    if os.environ.get("DISABLE_BRANCH", ""):
        cfg["disable_branch"] = env_bool("DISABLE_BRANCH", cfg["disable_branch"])
    if os.environ.get("USE_DWT", ""):
        cfg["use_dwt"] = env_bool("USE_DWT", cfg["use_dwt"])
    if os.environ.get("PRE_PATCH_DWT", ""):
        cfg["pre_patch_dwt"] = env_bool("PRE_PATCH_DWT", cfg["pre_patch_dwt"])
    if os.environ.get("USE_CONV_STEM", ""):
        cfg["use_conv_stem"] = env_bool("USE_CONV_STEM", cfg["use_conv_stem"])
    if os.environ.get("HEAD_INJECT_GATED", ""):
        cfg["head_inject_gated"] = env_bool("HEAD_INJECT_GATED", cfg["head_inject_gated"])
    if os.environ.get("POST_STEM_DWT", ""):
        cfg["post_stem_dwt"] = env_bool("POST_STEM_DWT", cfg.get("post_stem_dwt", False))

    dim = env_int("DIM", 192)
    depth = env_int("DEPTH", 12)
    feature_extractor_channels = env_list_int("FEAT_CH", default=[32, 64, 64])
    patch_base = env_int("PATCH_SIZE", 16)   # pass base patch size; if DWT enabled and auto_patch_dwt=True, model uses half internally
    stride_base = env_int("STRIDE", patch_base)
    auto_patch_dwt = env_bool("AUTO_PATCH_DWT", True)

    drop_path = env_float("DROP_PATH", 0.0)
    drop_path_decay = env_bool("DROP_PATH_DECAY", False)
    legacy_norm = env_bool("LEGACY_NORM", False)

    conv_kind = env_str("CONV_KIND", "2d")
    conv_kernel = env_int("CONV_KERNEL", 3)
    proj_bias = env_bool("PROJ_BIAS", True)
    norm_bias = env_bool("NORM_BIAS", True)

    dwt_fuse = env_str("DWT_FUSE", "gated")
    post_stem_merge_env = env_str("POST_STEM_MERGE", "replace")
    head_gate_hidden_ratio = env_float("HEAD_GATE_HIDDEN_RATIO", 0.0)
    head_gate_init_bias = env_float("HEAD_GATE_INIT_BIAS", -2.0)
    attn_pool_heads = env_int("ATTN_POOL_HEADS", 4)

    model = VisionLSTM2(
        dim=dim,
        input_shape=(3, img_size, img_size),
        patch_size=patch_base,
        depth=depth,
        output_shape=(num_classes,),
        mode="classifier",
        pooling=cfg["pooling"],
        drop_path_rate=drop_path,
        drop_path_decay=drop_path_decay,
        stride=stride_base,
        legacy_norm=legacy_norm,
        conv_kind=conv_kind,
        conv_kernel_size=conv_kernel,
        proj_bias=proj_bias,
        norm_bias=norm_bias,
        feature_extractor_channels=feature_extractor_channels,
        use_dwt=cfg["use_dwt"],
        dwt_fuse=dwt_fuse,
        auto_patch_dwt=auto_patch_dwt,
        use_conv_stem=cfg["use_conv_stem"],
        pre_patch_dwt=cfg["pre_patch_dwt"],
        disable_branch=cfg["disable_branch"],
        head_inject_gated=cfg["head_inject_gated"],
        head_gate_hidden_ratio=head_gate_hidden_ratio,
        head_gate_init_bias=head_gate_init_bias,
        post_stem_dwt=cfg.get("post_stem_dwt", False),
        post_stem_merge=cfg.get("post_stem_merge", post_stem_merge_env),
        attn_pool_heads=attn_pool_heads,
    ).to(device)

    # Optionally resume
    resume_ckpt = env_str("RESUME_CKPT", "").strip()
    if resume_ckpt:
        map_location = {"cuda:%d" % 0: "cuda:%d" % local_rank}
        if os.path.isfile(resume_ckpt):
            state = torch.load(resume_ckpt, map_location=map_location)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if is_main_process():
                print(f"[Resume] {resume_ckpt} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        else:
            if is_main_process():
                print(f"[Resume] not found: {resume_ckpt}", flush=True)

    # DDP wrap
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # Hyperparams
    accum_steps   = env_int("ACCUM_STEPS", 1)
    num_epochs    = env_int("EPOCHS", 100)
    warmup_epochs = env_int("WARMUP_EPOCHS", 7)

    mixup_alpha   = env_float("MIXUP_ALPHA", 0.2)
    cutmix_alpha  = env_float("CUTMIX_ALPHA", 0.8)
    mixup_prob    = env_float("MIXUP_PROB", 0.8)
    switch_prob   = env_float("SWITCH_PROB", 0.5)
    label_smooth  = env_float("LABEL_SMOOTH", 0.1)

    ema_decay     = env_float("EMA_DECAY", 0.9995)
    base_lr       = env_float("BASE_LR", 5e-4)
    weight_decay  = env_float("WEIGHT_DECAY", 0.05)
    clip_grad     = env_float("CLIP_GRAD", 1.0)
    branch_lr_scale = env_float("BRANCH_LR_SCALE", 1.0)

    BRANCH_START = env_int("BRANCH_START", warmup_epochs)
    BRANCH_RAMP  = env_int("BRANCH_RAMP", 15)
    BRANCH_MAX   = env_float("BRANCH_ALPHA_MAX", 1e-2)

    world_size = dist.get_world_size()
    global_batch = per_gpu_bs * world_size * accum_steps

    if is_main_process():
        print(
            f"[Config] ablation={ablation_id} baseline={baseline_ablation} | img={img_size} | "
            f"epochs={num_epochs} warmup={warmup_epochs} per_gpu_bs={per_gpu_bs} global_bs={global_batch} accum={accum_steps}",
            flush=True
        )
        print(
            f"[Model] dim={dim} depth={depth} feat_ch={feature_extractor_channels} | patch_base={patch_base} stride_base={stride_base} auto_patch_dwt={auto_patch_dwt}",
            flush=True
        )
        print(
            f"        use_stem={cfg['use_conv_stem']} use_dwt={cfg['use_dwt']} pre_patch_dwt={cfg['pre_patch_dwt']} | pooling={cfg['pooling']} | disable_branch={cfg['disable_branch']} gated={cfg['head_inject_gated']}",
            flush=True
        )
        print(
            f"[Opt] base_lr={base_lr:.2e} wd={weight_decay} clip={clip_grad} ema={ema_decay} amp={amp_dtype}",
            flush=True
        )

    # Optimizer & scheduler
    param_groups = build_param_groups(model.module, base_lr, weight_decay, branch_lr_scale)
    optimizer = torch.optim.AdamW(param_groups)

    updates_per_epoch = math.ceil(len(train_loader) / accum_steps)
    num_training_steps = num_epochs * updates_per_epoch
    warmup_steps = warmup_epochs * updates_per_epoch

    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    sch1 = LinearLR(optimizer, start_factor=0.1, total_iters=max(1, warmup_steps))
    sch2 = CosineAnnealingLR(
        optimizer,
        T_max=max(1, num_training_steps - warmup_steps),
        eta_min=base_lr * 3e-2
    )
    scheduler = SequentialLR(optimizer, schedulers=[sch1, sch2], milestones=[warmup_steps])

    # EMA
    ema_model = create_ema_model(model.module).to(device)

    # Checkpoint
    out_dir = env_str("OUT_DIR", "./outputs_lstm5_stage1")
    os.makedirs(out_dir, exist_ok=True)
    tag = env_str("RUN_TAG", f"{ablation_id}_sub{subset_k}_img{img_size}_dim{dim}_d{depth}")
    ckpt_path = os.path.join(out_dir, f"{tag}_ema_best.pth")

    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        if cfg.get("disable_branch", False):
            alpha = 0.0
        else:
            alpha = get_branch_alpha(epoch, BRANCH_START, BRANCH_RAMP, BRANCH_MAX)

        # Open branch weight decay after BRANCH_START (optional)
        for g in optimizer.param_groups:
            if g.get("is_branch", False) and (not g.get("is_no_wd", False)):
                g["weight_decay"] = (weight_decay if epoch >= BRANCH_START else 0.0)

        try_set_head_alpha(model.module, alpha)
        try_set_head_alpha(ema_model, alpha)

        train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        soft_acc_hist = []

        if is_main_process():
            print(f"\n[Train] Epoch {epoch}/{num_epochs} | branch_alpha={alpha:.3e}", flush=True)

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

            with autocast("cuda", dtype=amp_autocast_dtype):
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

                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                update_ema(model.module, ema_model, ema_decay)
                scheduler.step()

            running_loss += loss.item() * accum_steps

            with torch.no_grad():
                pred = logits.argmax(1)
                soft_acc = soft_targets.gather(1, pred.unsqueeze(1)).squeeze(1).mean().item()
                soft_acc_hist.append(soft_acc)

            log_every = env_int("LOG_EVERY", 100)
            if is_main_process() and (it % log_every == 0):
                avg_acc = sum(soft_acc_hist) / max(1, len(soft_acc_hist))
                print(
                    f"  iter {it:5d}/{len(train_loader)} | loss {loss.item()*accum_steps:.4f} | soft_acc {avg_acc:.3f} | lr {scheduler.get_last_lr()[0]:.2e}",
                    flush=True
                )

        # Validation (EMA, rank0 only)
        val_loss_g, val_acc_g = 0.0, 0.0
        if is_main_process():
            ema_model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0

            with torch.inference_mode(), autocast("cuda", dtype=amp_autocast_dtype):
                for imgs, target in val_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    logits = ema_model(imgs)
                    loss = F.cross_entropy(logits, target)
                    val_loss += loss.item() * target.size(0)
                    pred = logits.argmax(1)
                    val_correct += (pred == target).sum().item()
                    val_total += target.size(0)

            val_loss_g = val_loss / max(1, val_total)
            val_acc_g = val_correct / max(1, val_total)

        if is_dist():
            metrics = torch.tensor([val_loss_g, val_acc_g], device=device, dtype=torch.float32)
            dist.broadcast(metrics, src=0)
            val_loss_g, val_acc_g = metrics[0].item(), metrics[1].item()

        if is_main_process():
            train_loss_epoch = running_loss / max(1, len(train_loader))
            train_soft_acc = sum(soft_acc_hist) / max(1, len(soft_acc_hist))
            print(f"[Epoch {epoch}] Train loss={train_loss_epoch:.4f}, soft_acc={train_soft_acc:.4f}", flush=True)
            print(f"[Epoch {epoch}] Val   loss={val_loss_g:.4f}, acc={val_acc_g:.4f}", flush=True)

            if val_acc_g > best_acc:
                best_acc = val_acc_g
                torch.save(ema_model.state_dict(), ckpt_path)
                print(f"  🌟 New best saved: {ckpt_path} (acc={best_acc:.4f})", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
