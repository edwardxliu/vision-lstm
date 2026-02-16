# lstm6_stage1_pretrain_192_sample_mergeflex.py
# ViCxLSTM - ViL-style Stage 1 Pre-Training (ImageNet-1K @192) with class-subset option
# - Supports merge_kind / merge_kinds / last_merge_kind for pyramid downsampling ablations
# - Keeps AMP(bf16) stable: GradScaler only enabled for fp16

import os
import math
import random
import numpy as np
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
from torch.amp import autocast, GradScaler

# ----------------- 基础配置 -----------------
NUM_CLASSES = 1000
cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True


def env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, None)
    if v is None:
        return default
    v = str(v).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def env_list(name: str, default=None, sep: str = ","):
    v = os.environ.get(name, None)
    if v is None:
        return default
    items = [s.strip() for s in v.split(sep)]
    items = [s for s in items if s]
    return items if items else default


def is_main_process():
    return int(os.environ.get("RANK", 0)) == 0


def ddp_setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ================= 子集：随机抽样类别（DDP-safe）=================
def select_subset_classes(num_total_classes: int, subset_k: int, seed: int, device):
    if subset_k <= 0 or subset_k >= num_total_classes:
        return list(range(num_total_classes))

    rank = dist.get_rank() if dist.is_initialized() else 0

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

    if dist.is_initialized():
        dist.broadcast(keep_len, src=0)
        k = int(keep_len.item())

        if rank != 0:
            keep_tensor = torch.empty(k, dtype=torch.int64, device=device)

        dist.broadcast(keep_tensor, src=0)

    keep = keep_tensor.tolist()
    keep.sort()
    return keep


def filter_imagefolder_inplace(ds: datasets.ImageFolder, keep_class_indices: list[int]):
    """
    就地过滤 ImageFolder：仅保留 keep_class_indices 对应的类，
    并把标签 remap 到 0..K-1，更新 classes/class_to_idx/samples/targets。
    返回新类数 K。
    """
    keep_set = set(keep_class_indices)
    remap = {old_i: new_i for new_i, old_i in enumerate(keep_class_indices)}

    new_samples = []
    new_targets = []

    for path, y in ds.samples:
        if y in keep_set:
            ny = remap[y]
            new_samples.append((path, ny))
            new_targets.append(ny)

    ds.samples = new_samples
    ds.imgs = new_samples
    ds.targets = new_targets

    old_classes = ds.classes
    new_classes = [old_classes[i] for i in keep_class_indices]
    ds.classes = new_classes
    ds.class_to_idx = {cls_name: i for i, cls_name in enumerate(new_classes)}

    return len(new_classes)


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
    cutmix_alpha=0.2,
    prob=1.0,
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
    loss = -(soft_targets * log_probs).sum(dim=1).mean()
    return loss


def get_branch_alpha(epoch, start_epoch=30, ramp_epochs=30, alpha_max=1e-2):
    if epoch < start_epoch:
        return 0.0
    t = min(1.0, (epoch - start_epoch) / max(1, ramp_epochs))
    return alpha_max * t


# ----------------- Optim param grouping -----------------
def build_param_groups(model, base_lr, weight_decay, branch_lr_scale=1.0):
    """
    分四组：main_decay / main_no_decay / branch_decay / branch_no_decay
    并在 param_group 里打标签，方便你后续动态改 branch 的 weight_decay。

    ✅ branch 范围：feature_extractor_branch / head_adapter.proj / gate_layer
    """
    no_wd = set()
    if hasattr(model, "no_weight_decay"):
        try:
            no_wd = set(model.no_weight_decay())
        except Exception:
            no_wd = set()

    # 保险：bias 一律不做 wd（即便 no_weight_decay() 没覆盖）
    for n, _p in model.named_parameters():
        if n.endswith(".bias"):
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
        groups.append({
            "params": main_decay,
            "lr": base_lr,
            "weight_decay": weight_decay,
            "is_branch": False,
            "is_no_wd": False,
        })
    if main_no_decay:
        groups.append({
            "params": main_no_decay,
            "lr": base_lr,
            "weight_decay": 0.0,
            "is_branch": False,
            "is_no_wd": True,
        })
    if branch_decay:
        groups.append({
            "params": branch_decay,
            "lr": base_lr * branch_lr_scale,
            "weight_decay": 0.0,   # 先 0，训练循环里按 BRANCH_START 再打开
            "is_branch": True,
            "is_no_wd": False,
        })
    if branch_no_decay:
        groups.append({
            "params": branch_no_decay,
            "lr": base_lr * branch_lr_scale,
            "weight_decay": 0.0,
            "is_branch": True,
            "is_no_wd": True,
        })
    return groups


# ----------------- 主入口 -----------------
def main():
    ddp_setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    setup_seed(42 + dist.get_rank())

    # AMP dtype（建议 bf16 更稳）
    amp_dtype = os.environ.get("AMP_DTYPE", "bf16").lower()
    if amp_dtype == "bf16":
        amp_autocast_dtype = torch.bfloat16
    else:
        amp_autocast_dtype = torch.float16

    # ✅ bf16 不需要 GradScaler；fp16 才开
    scaler = GradScaler("cuda", enabled=(amp_autocast_dtype == torch.float16))

    # ---- 数据增强（192×192）----
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(192, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(192),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    data_dir = os.environ.get("IMAGENET_ROOT", "/home/omnisky/Public/edward/workspace/data/imagenet_dataset")
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=train_tf)
    val_dataset   = datasets.ImageFolder(root=os.path.join(data_dir, "val"),   transform=val_tf)

    if is_main_process():
        print(f"[Stage1] Data → train: {len(train_dataset)}, val: {len(val_dataset)}", flush=True)

    # ---- 子集抽样 ----
    subset_k = int(os.environ.get("SUBSET_CLASSES", "150"))
    subset_seed = int(os.environ.get("SUBSET_SEED", "1234"))

    if subset_k > 0:
        total_classes = len(train_dataset.classes)
        keep_classes = select_subset_classes(total_classes, subset_k, subset_seed, device=device)

        new_k_train = filter_imagefolder_inplace(train_dataset, keep_classes)
        new_k_val   = filter_imagefolder_inplace(val_dataset, keep_classes)
        assert new_k_train == new_k_val, "train/val 子集类数不一致，检查数据目录结构"

        num_classes = new_k_train
        if is_main_process():
            print(
                f"[Subset] enabled: keep {num_classes}/{total_classes} classes, "
                f"seed={subset_seed} | train={len(train_dataset)} val={len(val_dataset)}",
                flush=True
            )
    else:
        num_classes = len(train_dataset.classes)

    # ---- DataLoader ----
    per_gpu_bs   = int(os.environ.get("PER_GPU_BATCH", "128"))
    num_workers  = int(os.environ.get("NUM_WORKERS", "8"))
    world_size   = dist.get_world_size()

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

    # ---- 模型 ----
    # ✅ 使用 mergeflex 版本（支持 patch/conv/conv_patch/haar + merge_kinds/last_merge_kind）
    # from vision_lstm6_mergeflex import VisionLSTM2
    from vision_lstm6 import VisionLSTM2


    pyramid      = os.environ.get("PYRAMID", "half").lower()  # none/half/half2/full
    pair_fusion  = os.environ.get("PAIR_FUSION", "parallel_gated")
    col_every    = int(os.environ.get("COL_EVERY", "2"))
    gamma_init   = float(os.environ.get("GAMMA_INIT", "1e-4"))
    mixer_every  = int(os.environ.get("MIXER_EVERY", "2"))

    # ✅ patch/stride 可配：默认 8/8；想对齐 ViL：PATCH_SIZE=16 STRIDE=16
    patch_size = int(os.environ.get("PATCH_SIZE", "8"))
    stride     = int(os.environ.get("STRIDE", str(patch_size)))

    # ✅ merge 配置
    #   MERGE_KIND:   单一策略（默认 patch）
    #   MERGE_KINDS:  逐 merge 指定（逗号分隔），优先级高于 MERGE_KIND
    #   LAST_MERGE_KIND: 强制覆盖最后一次 merge（默认 patch；设为 "none" 关闭覆盖）
    merge_kind = os.environ.get("MERGE_KIND", "patch").lower()
    merge_kinds = env_list("MERGE_KINDS", default=None)
    if merge_kinds is not None:
        merge_kinds = [s.lower() for s in merge_kinds]

    last_merge_kind = os.environ.get("LAST_MERGE_KIND", "patch").strip().lower()
    if last_merge_kind in ("", "none", "null"):
        last_merge_kind = None

    # 手工指定 stage_dims / stage_depths（逗号分隔）
    # 例：STAGE_DIMS="256,384"   STAGE_DEPTHS="2,6"
    stage_dims = env_list("STAGE_DIMS", default=None)
    if stage_dims is not None:
        stage_dims = [int(x) for x in stage_dims]

    stage_depths = env_list("STAGE_DEPTHS", default=None)
    if stage_depths is not None:
        stage_depths = [int(x) for x in stage_depths]

    # ✅ 输入端 DWT（与 merge_kind 解耦：merge 是 token 合并方式；use_dwt 是 feature_extractor 的输入端降采样/子带融合）
    use_dwt = env_bool("USE_DWT", default=True)

    model = VisionLSTM2(
        dim=384,
        input_shape=(3, 192, 192),
        patch_size=patch_size,
        depth=8,
        output_shape=(num_classes,),
        mode="classifier",
        pooling="global",
        stride=stride,
        legacy_norm=True,
        drop_path_rate=0.1,
        drop_path_decay=True,
        conv_kind="2d",
        conv_kernel_size=3,
        proj_bias=True,
        norm_bias=True,
        feature_extractor_channels=[32, 64],
        use_dwt=use_dwt,

        # ===== 金字塔/结构相关 =====
        pyramid=pyramid,
        mixer_every=mixer_every,
        pair_fusion=pair_fusion,
        col_every=col_every,
        gamma_init=gamma_init,

        # ===== merge ablation =====
        merge_kind=merge_kind,
        merge_kinds=merge_kinds,
        last_merge_kind=last_merge_kind,

        stage_dims=stage_dims,
        stage_depths=stage_depths,
    ).to(device)

    resume_ckpt = os.environ.get("RESUME_CKPT", "").strip()
    map_location = {"cuda:%d" % 0: "cuda:%d" % local_rank}
    if resume_ckpt:
        if os.path.isfile(resume_ckpt):
            state = torch.load(resume_ckpt, map_location=map_location)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if is_main_process():
                print(f"[Stage1] Resume from {resume_ckpt}, missing={len(missing)}, unexpected={len(unexpected)}", flush=True)
        else:
            if is_main_process():
                print(f"[Stage1] RESUME_CKPT {resume_ckpt} not found, train from scratch", flush=True)
    else:
        if is_main_process():
            print("[Stage1] Train from scratch", flush=True)

    # branch alpha：先关掉注入 + alpha 不参与优化
    with torch.no_grad():
        model.head_adapter.alpha.fill_(0.0)
    model.head_adapter.alpha.requires_grad_(False)

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    # ---- 超参 & 优化器 & Scheduler ----
    accum_steps   = int(os.environ.get("ACCUM_STEPS", "1"))
    num_epochs    = int(os.environ.get("EPOCHS", "100"))
    warmup_epochs = int(os.environ.get("WARMUP_EPOCHS", "7"))

    mixup_alpha   = float(os.environ.get("MIXUP_ALPHA", "0.2"))
    cutmix_alpha  = float(os.environ.get("CUTMIX_ALPHA", "0.8"))
    mixup_prob    = float(os.environ.get("MIXUP_PROB", "0.8"))
    switch_prob   = float(os.environ.get("SWITCH_PROB", "0.5"))
    label_smooth  = float(os.environ.get("LABEL_SMOOTH", "0.1"))
    ema_decay     = float(os.environ.get("EMA_DECAY", "0.9995"))

    global_batch = per_gpu_bs * world_size * accum_steps
    base_lr = float(os.environ.get("BASE_LR", "5e-4"))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", "0.05"))
    clip_grad = float(os.environ.get("CLIP_GRAD", "1.0"))
    BRANCH_LR_SCALE = float(os.environ.get("BRANCH_LR_SCALE", "1.0"))

    if is_main_process():
        print(
            f"[Stage1 Config] classes={num_classes}, epochs={num_epochs}, warmup_epochs={warmup_epochs}, "
            f"per_gpu_bs={per_gpu_bs}, global_bs={global_batch}, accum={accum_steps}, "
            f"base_lr={base_lr:.3e}, weight_decay={weight_decay}, clip_grad={clip_grad}, "
            f"mixup={mixup_alpha}, cutmix={cutmix_alpha}, prob={mixup_prob}, "
            f"label_smooth={label_smooth}, ema_decay={ema_decay}, amp={amp_dtype}",
            flush=True
        )
        print(
            f"[ModelCfg] patch={patch_size}, stride={stride}, use_dwt={use_dwt} | "
            f"pyramid={pyramid}, fusion={pair_fusion}, col_every={col_every}, gamma_init={gamma_init}, mixer_every={mixer_every}",
            flush=True
        )
        if merge_kinds is None:
            print(f"[MergeCfg] merge_kind={merge_kind}, last_merge_kind={last_merge_kind}", flush=True)
        else:
            print(f"[MergeCfg] merge_kinds={merge_kinds}, last_merge_kind={last_merge_kind}", flush=True)

    # ✅ 参数分组（支持 no_weight_decay；gate_layer 归 branch）
    param_groups = build_param_groups(model.module, base_lr, weight_decay, BRANCH_LR_SCALE)
    optimizer = torch.optim.AdamW(param_groups)

    updates_per_epoch  = math.ceil(len(train_loader) / accum_steps)
    num_training_steps = num_epochs * updates_per_epoch
    warmup_steps       = warmup_epochs * updates_per_epoch

    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    sch1 = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    sch2 = CosineAnnealingLR(
        optimizer,
        T_max=max(1, num_training_steps - warmup_steps),
        eta_min=base_lr * 3e-2
    )
    scheduler = SequentialLR(optimizer, schedulers=[sch1, sch2], milestones=[warmup_steps])

    # EMA
    ema_model = create_ema_model(model.module).to(device)

    # ---- 训练循环 ----
    best_acc = 0.0
    pretrain_ckpt = os.environ.get("PRETRAIN_CKPT", "lstm6_stage1_192.pth")
    global_step = 0

    BRANCH_START = int(os.environ.get("BRANCH_START", str(warmup_epochs)))
    BRANCH_RAMP  = int(os.environ.get("BRANCH_RAMP", "15"))
    BRANCH_MAX   = float(os.environ.get("BRANCH_ALPHA_MAX", "1e-2"))

    for epoch in range(1, num_epochs + 1):
        a = get_branch_alpha(epoch, BRANCH_START, BRANCH_RAMP, BRANCH_MAX)

        # ✅ 动态打开 branch_decay 的 weight_decay（gate_layer 也在 branch 里）
        for g in optimizer.param_groups:
            if g.get("is_branch", False) and (not g.get("is_no_wd", False)):
                g["weight_decay"] = (weight_decay if epoch >= BRANCH_START else 0.0)

        with torch.no_grad():
            model.module.head_adapter.alpha.fill_(a)
            ema_model.head_adapter.alpha.fill_(a)

        train_sampler.set_epoch(epoch)
        model.train()
        running_loss, acc_hist = 0.0, []

        if is_main_process():
            print(f"\n[Stage1-Train] Epoch {epoch}/{num_epochs}", flush=True)
            print(f"[Branch] alpha={a:.3e} (start={BRANCH_START}, ramp={BRANCH_RAMP}, max={BRANCH_MAX})", flush=True)

        optimizer.zero_grad(set_to_none=True)

        for i, (imgs, target) in enumerate(train_loader, 1):
            imgs   = imgs.to(device, non_blocking=True)
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

            if i % accum_steps == 0:
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
                global_step += 1

            running_loss += loss.item() * accum_steps

            with torch.no_grad():
                pred = logits.argmax(1)
                soft_acc = soft_targets.gather(1, pred.unsqueeze(1)).squeeze(1).mean().item()
                acc_hist.append(soft_acc)

            if is_main_process() and (i % 100 == 0):
                avg_acc = sum(acc_hist) / len(acc_hist)
                print(
                    f"  iter {i:5d}/{len(train_loader)} | "
                    f"loss {loss.item()*accum_steps:.4f} | "
                    f"soft_acc {avg_acc:.3f} | lr {scheduler.get_last_lr()[0]:.2e}",
                    flush=True
                )

        # ---- 验证（用 EMA 模型，rank0 跑全量）----
        val_loss_g, val_acc_g = 0.0, 0.0

        if is_main_process():
            ema_model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0

            with torch.inference_mode(), autocast("cuda", dtype=amp_autocast_dtype):
                for imgs, target in val_loader:
                    imgs   = imgs.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)

                    logits = ema_model(imgs)
                    loss   = F.cross_entropy(logits, target)

                    val_loss    += loss.item() * target.size(0)
                    pred        = logits.argmax(1)
                    val_correct += (pred == target).sum().item()
                    val_total   += target.size(0)

            val_loss_g = val_loss / max(val_total, 1)
            val_acc_g  = val_correct / max(val_total, 1)

        # rank0 指标广播给所有 rank
        if dist.is_initialized():
            metrics = torch.tensor([val_loss_g, val_acc_g], device=device, dtype=torch.float32)
            dist.broadcast(metrics, src=0)
            val_loss_g, val_acc_g = metrics[0].item(), metrics[1].item()

        if is_main_process():
            train_loss_epoch = running_loss / len(train_loader)
            train_acc_epoch  = sum(acc_hist) / len(acc_hist) if acc_hist else 0.0
            print(f"[Stage1 Epoch {epoch}] Train loss={train_loss_epoch:.4f}, soft_acc={train_acc_epoch:.4f}", flush=True)
            print(f"[Stage1 Epoch {epoch}] Val   loss={val_loss_g:.4f}, acc={val_acc_g:.4f}, lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

            if val_acc_g > best_acc:
                best_acc = val_acc_g
                torch.save(ema_model.state_dict(), pretrain_ckpt)
                print(f"  🌟 Stage1 New best saved @ {pretrain_ckpt} (acc={best_acc:.4f})", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
