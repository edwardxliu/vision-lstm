# VisionLSTM5 (mod2) - Tiny-ImageNet-200 Training (Single GPU/CPU, no DDP)
#
# This script is adapted for running structured ablations on Tiny-ImageNet with vision_lstm5_mod2.py:
#   Stage A (path ablations): A0/A1/A2/A3
#   Stage B (branch injection): B0/B1/B2
#   Stage C (pooling): C0/C1
#
# Usage examples (Windows/Linux):
#   set TINYIMAGENET_ROOT=...\tiny-imagenet-200
#   set ABLATION=A0
#   set EPOCHS=100
#   python lstm5_train_tinyimagenet_ablation.py
#
# Notes:
# - PATCH_SIZE / STRIDE are treated as the *base* patch config (pre-DWT). If auto_patch_dwt=True (recommended),
#   the model will automatically halve patch/stride when DWT downsampling is enabled, keeping token grid fair.
# - For Tiny-ImageNet (IMG_SIZE=64), a good base default is PATCH_SIZE=4 (no DWT) and PATCH_SIZE_eff=2 (with DWT).

import os
import math
import random
import numpy as np
from copy import deepcopy
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
import torch.backends.cudnn as cudnn
from torch.amp import autocast, GradScaler

# ----------------- 基础配置 -----------------
NUM_CLASSES = int(os.environ.get("NUM_CLASSES", "200"))

cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------- EMA 工具 -----------------
def create_ema_model(model):
    ema = deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    ema.eval()
    return ema


@torch.no_grad()
def update_ema(model, ema_model, decay: float):
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


# ----------------- Mixup / CutMix -----------------
def rand_bbox(W, H, lam):
    cut_rat = (1.0 - lam) ** 0.5
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = int(np.clip(cx - cut_w // 2, 0, W))
    y1 = int(np.clip(cy - cut_h // 2, 0, H))
    x2 = int(np.clip(cx + cut_w // 2, 0, W))
    y2 = int(np.clip(cy + cut_h // 2, 0, H))
    return x1, y1, x2, y2


def apply_mixup_cutmix(
    x: torch.Tensor,
    y: torch.Tensor,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    prob: float = 0.0,
    switch_prob: float = 0.5,
):
    """
    Returns:
      x_mixed, y_a, y_b, lam, mixed
    - If not mixed: y_b == y_a, lam=1.0, mixed=False
    """
    if prob <= 0.0 or (mixup_alpha <= 0.0 and cutmix_alpha <= 0.0):
        return x, y, y, 1.0, False

    if np.random.rand() > prob:
        return x, y, y, 1.0, False

    bs = x.size(0)
    device = x.device
    perm = torch.randperm(bs, device=device)
    y_a = y
    y_b = y[perm]

    use_cutmix = (np.random.rand() < switch_prob) and (cutmix_alpha > 0.0)
    if use_cutmix:
        lam = float(np.random.beta(cutmix_alpha, cutmix_alpha))
        _, _, H, W = x.size()
        x1, y1, x2, y2 = rand_bbox(W, H, lam)

        x_mixed = x.clone()
        x_mixed[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]

        area = (x2 - x1) * (y2 - y1)
        lam_adj = 1.0 - float(area) / float(W * H)
        return x_mixed, y_a, y_b, lam_adj, True
    else:
        lam = float(np.random.beta(mixup_alpha, mixup_alpha))
        x_mixed = x * lam + x[perm] * (1.0 - lam)
        return x_mixed, y_a, y_b, lam, True


def soft_target_loss(
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
    label_smooth: float = 0.0,
    mixed: bool = False,
):
    def ce(pred, target):
        try:
            return F.cross_entropy(pred, target, label_smoothing=label_smooth)
        except TypeError:
            if label_smooth <= 0:
                return F.cross_entropy(pred, target)
            num_classes = pred.size(1)
            log_probs = F.log_softmax(pred, dim=1)
            nll = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
            smooth = -log_probs.mean(dim=1)
            return ((1.0 - label_smooth) * nll + label_smooth * smooth).mean()

    if not mixed:
        return ce(logits, y_a)
    return lam * ce(logits, y_a) + (1.0 - lam) * ce(logits, y_b)


# ----------------- TinyImageNet Val Dataset -----------------
class TinyImageNetVal(Dataset):
    """
    tiny-imagenet-200/
      train/<wnid>/images/*.JPEG
      val/images/*.JPEG
      val/val_annotations.txt
    """
    def __init__(self, val_root: str, class_to_idx: dict, transform=None):
        self.val_root = val_root
        self.transform = transform
        self.class_to_idx = class_to_idx

        ann_path = os.path.join(val_root, "val_annotations.txt")
        img_dir = os.path.join(val_root, "images")

        if not os.path.isfile(ann_path):
            raise FileNotFoundError(f"val_annotations.txt not found at: {ann_path}")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"val/images not found at: {img_dir}")

        samples = []
        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                img_name, wnid = parts[0], parts[1]
                if wnid not in class_to_idx:
                    continue
                path = os.path.join(img_dir, img_name)
                target = class_to_idx[wnid]
                samples.append((path, target))

        if len(samples) == 0:
            raise RuntimeError(
                "TinyImageNetVal found 0 samples. "
                "请确认 TINYIMAGENET_ROOT 指向 tiny-imagenet-200，且 train/val 目录完整。"
            )

        self.samples = samples
        self.loader = default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, target = self.samples[idx]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


# ----------------- Optim param grouping -----------------
def build_param_groups(model, base_lr, weight_decay, branch_lr_scale=1.0):
    """
    分四组：main_decay / main_no_decay / branch_decay / branch_no_decay
    branch 判定：feature_extractor_branch / head_adapter（包含门控注入）
    """
    no_wd = set()
    if hasattr(model, "no_weight_decay"):
        try:
            no_wd = set(model.no_weight_decay())
        except Exception:
            no_wd = set()

    for n, _p in model.named_parameters():
        if n.endswith(".bias"):
            no_wd.add(n)
    for n, _p in model.named_parameters():
        if ".bn" in n and n.endswith(".weight"):
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


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key, "")
    if v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key, "")
    return default if v == "" else int(v)


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key, "")
    return default if v == "" else float(v)


# ----------------- 主入口（单机单卡） -----------------
def main():
    # device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")

    setup_seed(int(os.environ.get("SEED", "42")))

    # AMP dtype probe (kept from your script)
    amp_dtype_req = os.environ.get("AMP_DTYPE", "bf16").lower().strip()

    def _make_amp_ctx(_dtype):
        if device.type != "cuda" or _dtype is None:
            return nullcontext()
        return autocast("cuda", dtype=_dtype)

    def _probe_depthwise_conv(_dtype) -> bool:
        if device.type != "cuda" or _dtype is None:
            return True
        try:
            for ch in (384, 768):
                x = torch.randn(2, ch, 8, 8, device=device, dtype=torch.float16)
                conv = torch.nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch, bias=False).to(device)
                with autocast("cuda", dtype=_dtype):
                    y = conv(x)
                _ = y.mean().item()
            return True
        except RuntimeError as e:
            msg = str(e)
            if ("unable to find an engine" in msg.lower()) or ("find was unable to find an engine" in msg.lower()):
                return False
            return False

    amp_autocast_dtype = None
    if amp_dtype_req in ("none", "no", "off", "fp32"):
        amp_autocast_dtype = None
    elif amp_dtype_req == "fp16":
        amp_autocast_dtype = torch.float16
    else:
        if device.type == "cuda" and hasattr(torch.cuda, "is_bf16_supported") and (not torch.cuda.is_bf16_supported()):
            print("⚠️  AMP_DTYPE=bf16 但当前 CUDA 不支持 bf16，自动回退到 fp16。", flush=True)
            amp_autocast_dtype = torch.float16
        else:
            amp_autocast_dtype = torch.bfloat16
            if not _probe_depthwise_conv(amp_autocast_dtype):
                print("⚠️  bf16 下 depthwise conv cuDNN 无可用引擎，自动回退到 fp16。", flush=True)
                amp_autocast_dtype = torch.float16

    amp_ctx = _make_amp_ctx(amp_autocast_dtype)
    if amp_autocast_dtype is None:
        amp_name = "fp32"
    elif amp_autocast_dtype == torch.bfloat16:
        amp_name = "bf16"
    elif amp_autocast_dtype == torch.float16:
        amp_name = "fp16"
    else:
        amp_name = str(amp_autocast_dtype)

    scaler = GradScaler(enabled=(device.type == "cuda" and amp_autocast_dtype == torch.float16))

    # ---- TinyImageNet: 64×64 by default ----
    img_size = _env_int("IMG_SIZE", 64)
    val_resize = _env_int("VAL_RESIZE", int(img_size * 1.15))

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(val_resize),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    data_root = os.environ.get("TINYIMAGENET_ROOT", "./tiny-imagenet-200")
    train_root = os.path.join(data_root, "train")
    val_root = os.path.join(data_root, "val")

    train_dataset = datasets.ImageFolder(root=train_root, transform=train_tf)
    class_to_idx = train_dataset.class_to_idx
    val_dataset = TinyImageNetVal(val_root=val_root, class_to_idx=class_to_idx, transform=val_tf)

    print(f"[Tiny] train: {len(train_dataset)}, val: {len(val_dataset)} | classes={len(train_dataset.classes)}", flush=True)
    if len(train_dataset.classes) != NUM_CLASSES:
        print(f"⚠️  WARNING: train classes={len(train_dataset.classes)}，但 NUM_CLASSES={NUM_CLASSES}", flush=True)

    per_gpu_bs = _env_int("PER_GPU_BATCH", 32)
    num_workers = _env_int("NUM_WORKERS", 8)

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_gpu_bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=per_gpu_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )

    # ---- Ablation presets ----
    # A-stage: disable_branch=True (focus on path)
    # B-stage: compare gated vs ungated branch injection
    # C-stage: compare pooling (requires a fixed choice for branch injection)
    ABL = os.environ.get("ABLATION", "A0").strip().upper()

    presets = {
        "A0": dict(use_conv_stem=True,  use_dwt=True,  pre_patch_dwt=False, disable_branch=True,  head_inject_gated=True,  pooling="bilateral_flatten"),
        "A1": dict(use_conv_stem=True,  use_dwt=False, pre_patch_dwt=False, disable_branch=True,  head_inject_gated=True,  pooling="bilateral_flatten"),
        "A2": dict(use_conv_stem=False, use_dwt=False, pre_patch_dwt=True,  disable_branch=True,  head_inject_gated=True,  pooling="bilateral_flatten"),
        "A3": dict(use_conv_stem=False, use_dwt=False, pre_patch_dwt=False, disable_branch=True,  head_inject_gated=True,  pooling="bilateral_flatten"),
        "B0": dict(use_conv_stem=True,  use_dwt=True,  pre_patch_dwt=False, disable_branch=False, head_inject_gated=True,  pooling="bilateral_flatten"),
        "B1": dict(use_conv_stem=True,  use_dwt=True,  pre_patch_dwt=False, disable_branch=False, head_inject_gated=False, pooling="bilateral_flatten"),
        "B2": dict(use_conv_stem=True,  use_dwt=True,  pre_patch_dwt=False, disable_branch=True,  head_inject_gated=True,  pooling="bilateral_flatten"),
        "C0": dict(use_conv_stem=True,  use_dwt=True,  pre_patch_dwt=False, disable_branch=False, head_inject_gated=True,  pooling="bilateral_flatten"),
        "C1": dict(use_conv_stem=True,  use_dwt=True,  pre_patch_dwt=False, disable_branch=False, head_inject_gated=True,  pooling="attn"),
    }
    preset = presets.get(ABL, presets["A0"])

    # Allow env overrides
    use_conv_stem = _env_bool("USE_CONV_STEM", preset["use_conv_stem"])
    use_dwt = _env_bool("USE_DWT", preset["use_dwt"])
    pre_patch_dwt = _env_bool("PRE_PATCH_DWT", preset["pre_patch_dwt"])
    disable_branch = _env_bool("DISABLE_BRANCH", preset["disable_branch"])
    head_inject_gated = _env_bool("HEAD_INJECT_GATED", preset["head_inject_gated"])
    pooling = os.environ.get("POOLING", preset["pooling"]).strip()

    auto_patch_dwt = _env_bool("AUTO_PATCH_DWT", True)

    # Base patch config (pre-DWT)
    patch_size = _env_int("PATCH_SIZE", 4)
    stride = _env_int("STRIDE", patch_size)

    # Conv stem channels (used only when use_conv_stem=True)
    feat_ch_str = os.environ.get("FEAT_CH", "32,64,64")
    feature_extractor_channels = [int(x) for x in feat_ch_str.split(",") if x.strip()]

    # Model core
    dim = _env_int("DIM", 192)
    depth = _env_int("DEPTH", 8)

    conv_kind = os.environ.get("CONV_KIND", "2d")
    conv_kernel = _env_int("CONV_KERNEL", 3)
    legacy_norm = _env_bool("LEGACY_NORM", False)
    proj_bias = _env_bool("PROJ_BIAS", True)
    norm_bias = _env_bool("NORM_BIAS", True)
    drop_path_rate = _env_float("DROP_PATH", 0.05)
    drop_path_decay = _env_bool("DROP_PATH_DECAY", True)

    # DWT fuse (for stem DWT or pre-patch DWT, depending on config)
    dwt_fuse = os.environ.get("DWT_FUSE", "gated").strip()

    # ---- Model ----
    from vision_lstm5_mod2 import VisionLSTM2

    model = VisionLSTM2(
        dim=dim,
        input_shape=(3, img_size, img_size),
        patch_size=patch_size,
        stride=stride,
        depth=depth,
        output_shape=(NUM_CLASSES,),
        mode="classifier",
        pooling=pooling,
        drop_path_rate=drop_path_rate,
        drop_path_decay=drop_path_decay,
        legacy_norm=legacy_norm,
        conv_kind=conv_kind,
        conv_kernel_size=conv_kernel,
        proj_bias=proj_bias,
        norm_bias=norm_bias,

        feature_extractor_channels=feature_extractor_channels,
        use_conv_stem=use_conv_stem,

        # DWT options
        use_dwt=use_dwt,
        dwt_fuse=dwt_fuse,
        pre_patch_dwt=pre_patch_dwt,
        auto_patch_dwt=auto_patch_dwt,

        # Branch options
        disable_branch=disable_branch,
        head_inject_gated=head_inject_gated,
    ).to(device)

    # ---- Print ablation summary ----
    # Effective patch/stride is handled inside model when auto_patch_dwt=True.
    print(
        f"[Ablation] {ABL} | stem={use_conv_stem} | use_dwt={use_dwt} | pre_patch_dwt={pre_patch_dwt} | "
        f"auto_patch_dwt={auto_patch_dwt} | disable_branch={disable_branch} | gated_inject={head_inject_gated} | pooling={pooling}",
        flush=True
    )
    print(f"[Model] dim={dim}, depth={depth}, base_patch={patch_size}, base_stride={stride}, feat_ch={feature_extractor_channels}, dwt_fuse={dwt_fuse}", flush=True)

    # Optional resume
    resume_ckpt = os.environ.get("RESUME_CKPT", "").strip()
    if resume_ckpt and os.path.isfile(resume_ckpt):
        state = torch.load(resume_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[Resume] {resume_ckpt} | missing={len(missing)}, unexpected={len(unexpected)}", flush=True)
    elif resume_ckpt:
        print(f"[Resume] RESUME_CKPT {resume_ckpt} not found, train from scratch", flush=True)
    else:
        print("[Resume] Train from scratch", flush=True)

    # ---- Hyperparams ----
    num_epochs = _env_int("EPOCHS", 100)
    warmup_epochs = _env_int("WARMUP_EPOCHS", 5)
    accum_steps = _env_int("ACCUM_STEPS", 1)

    mix_prob = _env_float("MIX_PROB", 0.0)  # default off for architecture iteration
    mixup_alpha = _env_float("MIXUP", 0.2)
    cutmix_alpha = _env_float("CUTMIX", 1.0)
    switch_prob = _env_float("SWITCH_PROB", 0.5)
    label_smooth = _env_float("LABEL_SMOOTH", 0.1)

    ema_decay = _env_float("EMA_DECAY", 0.9999)

    global_batch = per_gpu_bs * accum_steps
    base_lr = _env_float("BASE_LR", 2e-4)
    weight_decay = _env_float("WEIGHT_DECAY", 0.05)
    clip_grad = _env_float("CLIP_GRAD", 1.0)
    branch_lr_scale = _env_float("BRANCH_LR_SCALE", 1.0)

    print(
        f"[Config] img={img_size}, epochs={num_epochs}, warmup={warmup_epochs}, "
        f"bs={per_gpu_bs}, accum={accum_steps}, global_bs={global_batch}, "
        f"lr={base_lr:.2e}, wd={weight_decay}, clip={clip_grad}, amp={amp_name} | "
        f"mix_prob={mix_prob}, mixup={mixup_alpha}, cutmix={cutmix_alpha}, ls={label_smooth}",
        flush=True
    )

    param_groups = build_param_groups(model, base_lr, weight_decay, branch_lr_scale)
    optimizer = torch.optim.AdamW(param_groups, lr=base_lr)

    # LR schedule: linear warmup + cosine (per optimizer step)
    updates_per_epoch = max(1, math.ceil(len(train_loader) / accum_steps))
    num_training_steps = num_epochs * updates_per_epoch
    warmup_steps = warmup_epochs * updates_per_epoch

    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    sch1 = LinearLR(optimizer, start_factor=0.1, total_iters=max(1, warmup_steps))
    sch2 = CosineAnnealingLR(optimizer, T_max=max(1, num_training_steps - warmup_steps), eta_min=base_lr * 3e-2)
    scheduler = SequentialLR(optimizer, schedulers=[sch1, sch2], milestones=[warmup_steps])

    ema_model = create_ema_model(model).to(device)

    # Branch alpha schedule (only meaningful when branch is enabled and head_adapter exposes alpha)
    BRANCH_START = _env_int("BRANCH_ALPHA_START", 10)
    BRANCH_RAMP = _env_int("BRANCH_RAMP", 10)
    BRANCH_MAX = _env_float("BRANCH_ALPHA_MAX", 1e-2)

    def get_branch_alpha(epoch: int, start_epoch=30, ramp_epochs=30, alpha_max=1e-2):
        if epoch < start_epoch:
            return 0.0
        t = min(1.0, (epoch - start_epoch) / max(1, ramp_epochs))
        return alpha_max * t

    best_acc = 0.0
    pretrain_ckpt = os.environ.get("OUT_CKPT", f"tiny_{ABL}_ema_best.pth")
    log_every = _env_int("LOG_EVERY", 50)

    # ---- Train ----
    for epoch in range(1, num_epochs + 1):
        # Dynamic branch alpha & branch weight decay: only if branch enabled
        if (not disable_branch) and hasattr(model, "head_adapter") and hasattr(model.head_adapter, "alpha"):
            a = get_branch_alpha(epoch, BRANCH_START, BRANCH_RAMP, BRANCH_MAX)
            with torch.no_grad():
                model.head_adapter.alpha.fill_(a)
                if hasattr(ema_model, "head_adapter") and hasattr(ema_model.head_adapter, "alpha"):
                    ema_model.head_adapter.alpha.fill_(a)

            for g in optimizer.param_groups:
                if g.get("is_branch", False) and (not g.get("is_no_wd", False)):
                    g["weight_decay"] = (weight_decay if epoch >= BRANCH_START else 0.0)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        acc_hist = []
        opt_steps = 0

        for i, (imgs, target) in enumerate(train_loader, start=1):
            imgs = imgs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            imgs_m, y_a, y_b, lam, is_mixed = apply_mixup_cutmix(
                imgs, target,
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                prob=mix_prob,
                switch_prob=switch_prob
            )

            with amp_ctx:
                logits = model(imgs_m)
                loss = soft_target_loss(
                    logits, y_a, y_b, lam,
                    label_smooth=label_smooth,
                    mixed=is_mixed
                ) / accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item() * accum_steps

            with torch.no_grad():
                pred = logits.argmax(1)
                if not is_mixed:
                    acc = (pred == y_a).float().mean().item()
                else:
                    bs = logits.size(0)
                    soft_targets = torch.zeros((bs, logits.size(1)), device=logits.device, dtype=logits.dtype)
                    soft_targets.scatter_(1, y_a.unsqueeze(1), lam)
                    soft_targets.scatter_(1, y_b.unsqueeze(1), 1.0 - lam)
                    acc = soft_targets.gather(1, pred.unsqueeze(1)).squeeze(1).mean().item()
                acc_hist.append(acc)

            do_step = (i % accum_steps == 0)
            if do_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    if clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                opt_steps += 1
                update_ema(model, ema_model, ema_decay)

            if (log_every > 0) and (i % log_every == 0):
                avg_acc = float(sum(acc_hist) / max(1, len(acc_hist)))
                print(
                    f"  iter {i:5d}/{len(train_loader)} | loss {loss.item()*accum_steps:.4f} | "
                    f"acc {avg_acc:.3f} | lr {scheduler.get_last_lr()[0]:.2e}",
                    flush=True
                )

        # Tail step for grad accumulation
        if (len(train_loader) % accum_steps) != 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            opt_steps += 1
            update_ema(model, ema_model, ema_decay)

        # ---- Validate (EMA) ----
        ema_model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.inference_mode():
            for imgs, target in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                with amp_ctx:
                    logits = ema_model(imgs)
                    loss_v = F.cross_entropy(logits, target)
                val_loss += loss_v.item() * target.size(0)
                pred = logits.argmax(1)
                val_correct += (pred == target).sum().item()
                val_total += target.size(0)

        val_loss_g = val_loss / max(val_total, 1)
        val_acc_g = val_correct / max(val_total, 1)

        train_loss_epoch = running_loss / max(1, len(train_loader))
        train_acc_epoch = float(sum(acc_hist) / max(1, len(acc_hist)))

        print(f"[Epoch {epoch:03d}] Train loss={train_loss_epoch:.4f}, acc={train_acc_epoch:.4f} | opt_steps={opt_steps}", flush=True)
        print(f"[Epoch {epoch:03d}] Val   loss={val_loss_g:.4f}, acc={val_acc_g:.4f}, lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

        if val_acc_g > best_acc:
            best_acc = val_acc_g
            torch.save(ema_model.state_dict(), pretrain_ckpt)
            print(f"  🌟 New best saved @ {pretrain_ckpt} (acc={best_acc:.4f})", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
