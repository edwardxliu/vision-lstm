
# lstm6_stage1_    print(f"[Model] pooling={pooling}, use_dwt={use_dwt}, feat_ch={feature_extractor_channels}, patch={patch_size}, stride={stride}, pyramid={pyramid}, conv={conv_kind}{conv_kernel}", flush=True)

pretrain_tinyimagenet_single.py
# ViCxLSTM - Stage 1 Pre-Training on Tiny-ImageNet-200 (Single GPU/CPU, no DDP)
#
# Changes vs your pasted version:
# - Define apply_mixup_cutmix() and soft_target_loss()
# - Add MIX_PROB to easily disable mixup/cutmix for architecture iteration (default OFF)
# - Fix gradient-accumulation tail step (when len(loader) % accum_steps != 0)
# - Make training accuracy metric robust (hard acc if not mixed, soft acc if mixed)

import os
import math
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
import torch.backends.cudnn as cudnn
from torch.amp import autocast, GradScaler
from copy import deepcopy
from contextlib import nullcontext

# ----------------- 基础配置 -----------------
# Tiny-ImageNet 固定 200 类（可用环境变量覆盖）
NUM_CLASSES = int(os.environ.get("NUM_CLASSES", "200"))

cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True


def is_main_process():
    return True


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
    # 更稳：跳过非浮点参数/缓冲；支持 bf16/fp16/fp32 混合
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
        lam_adj = 1.0 - float(area) / float(W * H)  # adjust by true mixed area
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
    """
    Cross-entropy with optional label smoothing, supports mixup/cutmix by linear combination.
    """
    # PyTorch supports label_smoothing for cross_entropy in recent versions.
    # Provide a safe fallback if unavailable.
    def ce(pred, target):
        try:
            return F.cross_entropy(pred, target, label_smoothing=label_smooth)
        except TypeError:
            # manual label smoothing
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


def get_branch_alpha(epoch, start_epoch=30, ramp_epochs=30, alpha_max=1e-2):
    if epoch < start_epoch:
        return 0.0
    t = min(1.0, (epoch - start_epoch) / max(1, ramp_epochs))
    return alpha_max * t


# ----------------- TinyImageNet Val Dataset -----------------
class TinyImageNetVal(Dataset):
    """
    兼容 tiny-imagenet-200 原始目录结构：

      tiny-imagenet-200/
        train/<wnid>/images/*.JPEG
        val/images/*.JPEG
        val/val_annotations.txt   # 每行：img\twnid\tx1\ty1\tx2\ty2
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
    并在 param_group 里打标签，方便你后续动态改 branch 的 weight_decay。
    """
    no_wd = set()
    if hasattr(model, "no_weight_decay"):
        try:
            no_wd = set(model.no_weight_decay())
        except Exception:
            no_wd = set()

    # bias 一律不做 wd
    for n, _p in model.named_parameters():
        if n.endswith(".bias"):
            no_wd.add(n)
    # 额外建议：BatchNorm 的 weight 通常也不做 weight decay（更稳一些）
    for n, _p in model.named_parameters():
        if ".bn" in n and n.endswith(".weight"):
            no_wd.add(n)


    main_decay, main_no_decay = [], []
    branch_decay, branch_no_decay = [], []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        is_branch = (
            n.startswith("feature_extractor_branch")
            or n.startswith("head_adapter.proj")
            or n.startswith("gate_layer")
        )
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


# ----------------- 主入口（单机单卡） -----------------
def main():
    # 单机单卡（或 CPU）训练：不使用 DDP
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")

    setup_seed(int(os.environ.get("SEED", "42")))

    # AMP dtype（bf16 更稳；fp16 速度更快但更敏感）
    # 经验：部分 Windows + 消费级 GPU 在 bf16 depthwise conv 上会触发 cuDNN “FIND was unable to find an engine”
    # 因此这里做一次小探针：若 bf16 下 depthwise conv 不可用，则自动回退到 fp16（或用户显式关闭 AMP）。
    amp_dtype_req = os.environ.get("AMP_DTYPE", "bf16").lower().strip()

    def _make_amp_ctx(_dtype):
        if device.type != "cuda" or _dtype is None:
            return nullcontext()
        return autocast("cuda", dtype=_dtype)

    def _probe_depthwise_conv(_dtype) -> bool:
        if device.type != "cuda" or _dtype is None:
            return True
        try:
            # 尽量复现你的 ViLLayer 里 depthwise conv 的典型形态：groups=channels，H=W≈8
            for ch in (384, 768):
                x = torch.randn(2, ch, 8, 8, device=device, dtype=torch.float16)
                conv = torch.nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch, bias=False).to(device)
                with autocast("cuda", dtype=_dtype):
                    y = conv(x)
                _ = y.mean().item()
            return True
        except RuntimeError as e:
            msg = str(e)
            if ("FIND was unable to find an engine" in msg) or ("unable to find an engine" in msg.lower()):
                return False
            # 其他错误也视为不可用
            return False

    amp_autocast_dtype = None
    if amp_dtype_req in ("none", "no", "off", "fp32"):
        amp_autocast_dtype = None
    elif amp_dtype_req == "fp16":
        amp_autocast_dtype = torch.float16
    else:
        # 默认走 bf16，但要做两层检测：CUDA 支持 + cuDNN depthwise conv 可用
        if device.type == "cuda" and hasattr(torch.cuda, "is_bf16_supported") and (not torch.cuda.is_bf16_supported()):
            print("⚠️  AMP_DTYPE=bf16 但当前 CUDA 不支持 bf16，自动回退到 fp16。", flush=True)
            amp_autocast_dtype = torch.float16
        else:
            amp_autocast_dtype = torch.bfloat16
            if not _probe_depthwise_conv(amp_autocast_dtype):
                print("⚠️  bf16 下 depthwise conv cuDNN 无可用引擎（常见于 Windows/部分消费级 GPU/驱动）。自动回退到 fp16。", flush=True)
                amp_autocast_dtype = torch.float16

    amp_ctx = _make_amp_ctx(amp_autocast_dtype)
    scaler = GradScaler(enabled=(device.type == "cuda" and amp_autocast_dtype == torch.float16))

    # ---- TinyImageNet：默认 64×64（可配）----
    img_size = int(os.environ.get("IMG_SIZE", "64"))
    val_resize = int(os.environ.get("VAL_RESIZE", str(int(img_size * 1.15))))  # 64 -> 73

    # 仍用 ImageNet 归一化，方便后续迁移
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    # 训练增强：Tiny 上建议轻量一点
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

    # 数据根目录：TINYIMAGENET_ROOT 指向 tiny-imagenet-200
    data_root = os.environ.get("TINYIMAGENET_ROOT", "./tiny-imagenet-200")
    train_root = os.path.join(data_root, "train")
    val_root = os.path.join(data_root, "val")

    train_dataset = datasets.ImageFolder(root=train_root, transform=train_tf)
    class_to_idx = train_dataset.class_to_idx  # wnid->idx
    val_dataset = TinyImageNetVal(val_root=val_root, class_to_idx=class_to_idx, transform=val_tf)

    print(f"[Stage1-Tiny Single] train: {len(train_dataset)}, val: {len(val_dataset)} | classes={len(train_dataset.classes)}", flush=True)
    if len(train_dataset.classes) != NUM_CLASSES:
        print(
            f"⚠️  WARNING: train classes={len(train_dataset.classes)}，但 NUM_CLASSES={NUM_CLASSES}。\n"
            f"    如果你确认用的是 Tiny-ImageNet-200，请检查 TINYIMAGENET_ROOT；或将 NUM_CLASSES 设为实际类别数。",
            flush=True
        )

    per_gpu_bs = int(os.environ.get("PER_GPU_BATCH", "32"))
    num_workers = int(os.environ.get("NUM_WORKERS", "8"))

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

    # ---- 模型 ----
    from vision_lstm6 import VisionLSTM2

    pyramid = os.environ.get("PYRAMID", "half")  # none/half/half2/full
    pair_fusion = os.environ.get("PAIR_FUSION", "parallel_gated")
    col_every = int(os.environ.get("COL_EVERY", "2"))
    gamma_init = float(os.environ.get("GAMMA_INIT", "1e-4"))
    mixer_every = int(os.environ.get("MIXER_EVERY", "2"))

    # TinyImageNet + use_dwt=True 时，FeatureExtractor 在 'LL' 模式下会把 H/W 变成 IMG_SIZE//2
    patch_size = int(os.environ.get("PATCH_SIZE", "4"))
    stride = int(os.environ.get("STRIDE", str(patch_size)))

    # FeatureExtractor 的卷积通道配置（逗号分隔）
    feat_ch_str = os.environ.get("FEAT_CH", "64,64")
    feature_extractor_channels = [int(x) for x in feat_ch_str.split(",") if x.strip()]

    pooling = os.environ.get("POOLING", "global")  # global | bilateral_avg | bilateral_flatten | (None)
    conv_kind = os.environ.get("CONV_KIND", "2d")  # 2d | causal1d
    conv_kernel = int(os.environ.get("CONV_KERNEL", "3"))
    legacy_norm = (os.environ.get("LEGACY_NORM", "0") == "1")
    proj_bias = (os.environ.get("PROJ_BIAS", "1") == "1")
    norm_bias = (os.environ.get("NORM_BIAS", "1") == "1")
    drop_path_rate = float(os.environ.get("DROP_PATH", "0.05"))
    drop_path_decay = (os.environ.get("DROP_PATH_DECAY", "1") == "1")
    use_dwt = (os.environ.get("USE_DWT", "1") == "1")

    model = VisionLSTM2(
        dim=int(os.environ.get("DIM", "384")),
        input_shape=(3, img_size, img_size),
        patch_size=patch_size,
        depth=int(os.environ.get("DEPTH", "8")),
        output_shape=(NUM_CLASSES,),
        mode="classifier",
        pooling=pooling,
        drop_path_rate=drop_path_rate,
        drop_path_decay=drop_path_decay,
        stride=stride,
        legacy_norm=legacy_norm,
        conv_kind=conv_kind,
        conv_kernel_size=conv_kernel,
        proj_bias=proj_bias,
        norm_bias=norm_bias,
        feature_extractor_channels=feature_extractor_channels,
        use_dwt=use_dwt,

        # 金字塔/融合/局部 mixer
        pyramid=pyramid,
        mixer_every=mixer_every,
        pair_fusion=pair_fusion,
        col_every=col_every,
        gamma_init=gamma_init,
    ).to(device)

    # ---- 恢复（可选）----
    resume_ckpt = os.environ.get("RESUME_CKPT", "").strip()
    if resume_ckpt and os.path.isfile(resume_ckpt):
        state = torch.load(resume_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[Stage1-Tiny Single] Resume from {resume_ckpt} | missing={len(missing)}, unexpected={len(unexpected)}", flush=True)
    elif resume_ckpt:
        print(f"[Stage1-Tiny Single] RESUME_CKPT {resume_ckpt} not found, train from scratch", flush=True)
    else:
        print("[Stage1-Tiny Single] Train from scratch", flush=True)

    # branch alpha：先关掉
    if hasattr(model, "head_adapter") and hasattr(model.head_adapter, "alpha"):
        with torch.no_grad():
            model.head_adapter.alpha.fill_(0.0)

    # ---- 超参数 ----
    num_epochs = int(os.environ.get("EPOCHS", "100"))
    warmup_epochs = int(os.environ.get("WARMUP_EPOCHS", "5"))
    accum_steps = int(os.environ.get("ACCUM_STEPS", "1"))

    # ✅ 默认关闭混合增强：更适合 Tiny 上快速迭代结构
    mix_prob = float(os.environ.get("MIX_PROB", "0.0"))  # 0.0 = off
    mixup_alpha = float(os.environ.get("MIXUP", "0.2"))
    cutmix_alpha = float(os.environ.get("CUTMIX", "1.0"))
    switch_prob = float(os.environ.get("SWITCH_PROB", "0.5"))
    label_smooth = float(os.environ.get("LABEL_SMOOTH", "0.1"))

    ema_decay = float(os.environ.get("EMA_DECAY", "0.9999"))

    global_batch = per_gpu_bs * accum_steps
    base_lr = float(os.environ.get("BASE_LR", "2e-4"))  # 更稳的默认值：适配笔记本/小 batch
    weight_decay = float(os.environ.get("WEIGHT_DECAY", "0.05"))
    clip_grad = float(os.environ.get("CLIP_GRAD", "1.0"))
    BRANCH_LR_SCALE = float(os.environ.get("BRANCH_LR_SCALE", "1.0"))

    print(
        f"[Config] img={img_size}, epochs={num_epochs}, warmup_epochs={warmup_epochs}, "
        f"bs={per_gpu_bs}, accum={accum_steps}, global_bs={global_batch}, "
        f"lr={base_lr:.2e}, wd={weight_decay}, clip={clip_grad}, amp={amp_dtype} | "
        f"mix_prob={mix_prob}, mixup={mixup_alpha}, cutmix={cutmix_alpha}, ls={label_smooth}",
        flush=True
    )

    param_groups = build_param_groups(model, base_lr, weight_decay, BRANCH_LR_SCALE)
    optimizer = torch.optim.AdamW(param_groups, lr=base_lr)

    # LR schedule：Linear warmup + cosine（按 optimizer step 更新）
    updates_per_epoch = max(1, math.ceil(len(train_loader) / accum_steps))
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

    ema_model = create_ema_model(model).to(device)

    # branch alpha 动态策略
    BRANCH_START = int(os.environ.get("BRANCH_ALPHA_START", "10"))
    BRANCH_RAMP = int(os.environ.get("BRANCH_RAMP", "10"))
    BRANCH_MAX = float(os.environ.get("BRANCH_ALPHA_MAX", "1e-2"))

    best_acc = 0.0
    pretrain_ckpt = os.environ.get("OUT_CKPT", "stage1_tiny_ema_best.pth")
    log_every = int(os.environ.get("LOG_EVERY", "50"))

    # ---- 训练循环 ----
    for epoch in range(1, num_epochs + 1):
        # 动态 branch alpha
        a = get_branch_alpha(epoch, BRANCH_START, BRANCH_RAMP, BRANCH_MAX)
        if hasattr(model, "head_adapter") and hasattr(model.head_adapter, "alpha"):
            with torch.no_grad():
                model.head_adapter.alpha.fill_(a)
                if hasattr(ema_model, "head_adapter") and hasattr(ema_model.head_adapter, "alpha"):
                    ema_model.head_adapter.alpha.fill_(a)

        # 动态打开 branch_decay 的 weight_decay
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

            # mixup/cutmix（默认关闭：MIX_PROB=0）
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

            # 训练指标：不混合就 hard acc；混合就 soft acc（更贴合 mixup/cutmix）
            with torch.no_grad():
                pred = logits.argmax(1)
                if not is_mixed:
                    acc = (pred == y_a).float().mean().item()
                else:
                    # soft acc: probability mass assigned to predicted class
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

        # ✅ 处理尾巴：len(loader) 不是 accum_steps 的整数倍时，最后一段梯度也要 step
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

        # ---- 验证（EMA 模型）----
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
