# stage2_finetune_224.py
# ViCxLSTM - ViL-style Stage 2 Fine-Tuning on ImageNet-1K @224x224

import os
import math
import random
import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
from torchvision.transforms import ColorJitter, RandomErasing
from copy import deepcopy

from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

NUM_CLASSES = 1000
cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

class ImageFolderKD(datasets.ImageFolder):
    """
    返回 (img_student, img_teacher, target)
    - 两者共享同一套几何增强（RandomResizedCrop + HFlip）
    - student 额外做 ColorJitter + RandomErasing
    - teacher 不做 jitter/erasing，保证 KD target 稳定
    """
    def __init__(
        self,
        root,
        mean,
        std,
        crop_size=224,
        scale=(0.8, 1.0),
        ratio=(3/4, 4/3),
        hflip_p=0.5,
        student_jitter=None,
        student_erasing=None,
    ):
        super().__init__(root=root, transform=None)
        self.mean = mean
        self.std = std
        self.crop_size = crop_size
        self.scale = scale
        self.ratio = ratio
        self.hflip_p = hflip_p
        self.student_jitter = student_jitter
        self.student_erasing = student_erasing

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)  # PIL

        # ---- 共享几何增强：同一组 crop 参数 + flip 参数 ----
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            img, scale=self.scale, ratio=self.ratio
        )
        img = TF.resized_crop(
            img, i, j, h, w,
            size=[self.crop_size, self.crop_size],
            interpolation=InterpolationMode.BICUBIC
        )
        if random.random() < self.hflip_p:
            img = TF.hflip(img)

        # teacher view：不做颜色扰动/擦除
        img_t = img

        # student view：额外 jitter（在 PIL 上做更常见）
        img_s = img
        if self.student_jitter is not None:
            img_s = self.student_jitter(img_s)

        # to tensor + normalize
        img_t = TF.normalize(TF.to_tensor(img_t), self.mean, self.std)
        img_s = TF.normalize(TF.to_tensor(img_s), self.mean, self.std)

        # student only: random erasing（必须在 tensor 上）
        if self.student_erasing is not None:
            img_s = self.student_erasing(img_s)

        return img_s, img_t, target


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


def create_ema_model(model):
    ema = deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


@torch.no_grad()
def update_ema(model, ema_model, decay: float):
    msd = model.state_dict()
    for k, v in ema_model.state_dict().items():
        if k in msd:
            v.copy_(v * decay + msd[k].detach() * (1.0 - decay))

def evaluate_ddp(net, val_loader, device):
    """
    在 DDP 下评估 top1 acc / loss（全卡汇总）
    net: 普通 nn.Module（不要传 DDP wrapper，传 model.module 或 ema_model）
    """
    net.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.inference_mode(), torch.amp.autocast('cuda'):
        for imgs, target in val_loader:
            imgs   = imgs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            logits = net(imgs)
            loss   = F.cross_entropy(logits, target)

            bs = target.size(0)
            val_loss    += loss.item() * bs
            val_correct += (logits.argmax(1) == target).sum().item()
            val_total   += bs

    t = torch.tensor([val_loss, val_correct, val_total], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)

    loss_g = (t[0] / t[2]).item()
    acc_g  = (t[1] / t[2]).item()
    return loss_g, acc_g

def build_teacher(num_classes=1000, device="cuda"):
    """
    Teacher 使用 torchvision 预训练（ImageNet-1K）模型。
    用环境变量 TEACHER 控制模型类型：
      TEACHER=convnext_base | vit_b_16 | swin_b | resnet152 ...
    """
    import torchvision.models as M
    name = os.environ.get("TEACHER", "convnext_base").lower()
    weights_path = os.environ.get("TEACHER_WEIGHTS", "convnext_base-6075fbad.pth")  # 本地权重路径

    if name == "convnext_base":
        teacher = M.convnext_base(weights=None)
        if weights_path:
            sd = torch.load(weights_path, map_location="cpu", weights_only=True)
            teacher.load_state_dict(sd, strict=True)
        else:
            teacher = M.convnext_base(weights=M.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    elif name == "vit_b_16":
        teacher = M.vit_b_16(weights=M.ViT_B_16_Weights.IMAGENET1K_V1)
    elif name == "swin_b":
        teacher = M.swin_b(weights=M.Swin_B_Weights.IMAGENET1K_V1)
    elif name == "resnet152":
        teacher = M.resnet152(weights=M.ResNet152_Weights.IMAGENET1K_V2)
    else:
        raise ValueError(f"Unknown TEACHER={name}")

    teacher.eval().to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher

def get_branch_alpha(epoch, start_epoch=2, ramp_epochs=8, alpha_max=1e-2):
    if epoch < start_epoch:
        return 0.0
    t = min(1.0, (epoch - start_epoch) / max(1, ramp_epochs))
    return alpha_max * t

def main():
    ddp_setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    setup_seed(100 + dist.get_rank())

    # ---- 数据增强（224×224，弱化，不再 Mixup/CutMix）----
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    # train_tf = transforms.Compose([
    #     transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),  # 轻微颜色扰动
    #     transforms.ToTensor(),
    #     transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    #     transforms.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    # ])

    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    data_dir = os.environ.get(
        "IMAGENET_ROOT",
        "/home/omnisky/Public/edward/workspace/data/imagenet_dataset"
    )
    
    student_jitter  = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
    student_erasing = transforms.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3))

    train_dataset = ImageFolderKD(
        root=os.path.join(data_dir, "train"),
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        crop_size=224,
        #scale=(0.8, 1.0),
        scale=(0.9, 1.0),
        ratio=(3/4, 4/3),
        hflip_p=0.5,
        student_jitter=student_jitter,
        student_erasing=student_erasing,
    )

    val_dataset   = datasets.ImageFolder(root=os.path.join(data_dir, "val"),   transform=val_tf)

    if is_main_process():
        print(f"[Stage2] Data → train: {len(train_dataset)}, val: {len(val_dataset)}", flush=True)

    per_gpu_bs   = int(os.environ.get("PER_GPU_BATCH", "64"))
    num_workers  = int(os.environ.get("NUM_WORKERS", "8"))
    world_size   = dist.get_world_size()

    train_sampler = DistributedSampler(train_dataset, shuffle=True,  drop_last=False)
    val_sampler   = DistributedSampler(val_dataset,   shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_gpu_bs,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=per_gpu_bs,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # ---- 模型（224 输入）----
    from vision_lstm5 import VisionLSTM2
    model = VisionLSTM2(
        dim=384,
        input_shape=(3, 224, 224),
        patch_size=8,
        depth=8,
        output_shape=(NUM_CLASSES,),
        mode="classifier",
        pooling="global",
        stride=8,
        legacy_norm=True,
        drop_path_rate=0.1,
        drop_path_decay=True,
        conv_kind="2d",
        conv_kernel_size=3,
        proj_bias=True,
        norm_bias=True,
        feature_extractor_channels=[32, 64],
        use_dwt=True,
    ).to(device)

    # ---- 加载 Stage1 的预训练权重 ----
    pretrain_ckpt = os.environ.get("PRETRAIN_CKPT", "vil_stage1_192.pth")
    map_location  = {"cuda:%d" % 0: "cuda:%d" % local_rank}

    if os.path.isfile(pretrain_ckpt):
        state = torch.load(pretrain_ckpt, map_location=map_location)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if is_main_process():
            print("missing keys sample:", missing[:30])
            print("unexpected keys sample:", unexpected[:30])
            print(f"[Stage2] Loaded Stage1 ckpt {pretrain_ckpt}, missing={len(missing)}, unexpected={len(unexpected)}", flush=True)
    else:
        if is_main_process():
            print(f"[Stage2] WARN: Stage1 ckpt {pretrain_ckpt} not found, fine-tune from scratch!", flush=True)

    with torch.no_grad():
        model.head_adapter.alpha.fill_(0.0)         # ✅一开始完全不注入
    model.head_adapter.alpha.requires_grad_(False)  # ✅alpha 不参与优化器更新

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    def split_params(m):
        backbone, head, branch = [], [], []
        no_wd_names = set()
        if hasattr(m, "no_weight_decay"):
            no_wd_names |= set(m.no_weight_decay())

        for n, p in m.named_parameters():
            if not p.requires_grad:
                continue

            is_branch = n.startswith("feature_extractor_branch") or n.startswith("head_adapter.proj")
            is_head   = n.startswith("head.") or n.startswith("legacy_norm.")
            is_no_wd  = (p.ndim == 1) or n.endswith(".bias") or ("norm" in n.lower()) or (n in no_wd_names)

            if is_branch:
                branch.append((p, is_no_wd))
            elif is_head:
                head.append((p, is_no_wd))
            else:
                backbone.append((p, is_no_wd))

        return backbone, head, branch

    backbone, head, branch = split_params(model.module)

    # ---- KD 配置 ----
    kd_on = int(os.environ.get("KD_ON", "1")) == 1
    kd_hard_final = float(os.environ.get("KD_HARD_FINAL", "0.7"))  # hard CE 最终权重（越大越偏 hard）
    #kd_alpha = float(os.environ.get("KD_ALPHA", "0.1")) 
    kd_T = float(os.environ.get("KD_T", "2.0"))
    kd_warmup_epochs = int(os.environ.get("KD_WARMUP_EPOCHS", "2"))
    kd_ramp_epochs   = int(os.environ.get("KD_RAMP_EPOCHS", "4"))  # ✅KD 逐步加权
    label_smooth = float(os.environ.get("LABEL_SMOOTHING", "0.0")) # 可选：0.0/0.1

    def kd_hard_weight(epoch: int) -> float:
        """hard CE 权重：从 1.0 逐步降到 kd_hard_final"""
        if epoch < kd_warmup_epochs:
            return 1.0
        t = min(1.0, (epoch - kd_warmup_epochs) / max(1, kd_ramp_epochs))
        return 1.0 - t * (1.0 - kd_hard_final)

    # ---- Early Stop ----
    early_patience = int(os.environ.get("EARLY_STOP_PATIENCE", "2"))
    early_min_ep   = int(os.environ.get("EARLY_STOP_MIN_EPOCHS", "4"))
    bad_epochs = 0

    teacher = None
    if kd_on:
        teacher = build_teacher(NUM_CLASSES, device=device)
        if is_main_process():
            print(f"[KD] ON | teacher={os.environ.get('TEACHER','convnext_base')} "
                  f"| hard_final={kd_hard_final} | T={kd_T} | warmup={kd_warmup_epochs} | ramp={kd_ramp_epochs}",
                  flush=True)


    # ---- 超参（小 LR 短精调）----
    accum_steps   = 1
    num_epochs    = int(os.environ.get("EPOCHS", "15"))      # ViL: fine-tune 20 epochs
    warmup_epochs = int(os.environ.get("WARMUP_EPOCHS", "2"))

    base_lr       = float(os.environ.get("BASE_LR", "2e-5"))  
    #weight_decay  = float(os.environ.get("WEIGHT_DECAY", "0.02"))
    weight_decay  = float(os.environ.get("WEIGHT_DECAY", "0.05"))
    HEAD_LR_SCALE   = float(os.environ.get("HEAD_LR_SCALE", "2.0"))
    BRANCH_LR_SCALE = float(os.environ.get("BRANCH_LR_SCALE", "1.0"))
    ema_decay     = float(os.environ.get("EMA_DECAY", "0.9999"))

    global_batch = per_gpu_bs * world_size * accum_steps

    def pack(group, lr, wd):
        decay = [p for p, no_wd in group if not no_wd]
        nodecay = [p for p, no_wd in group if no_wd]
        out = []
        if decay:
            out.append({"params": decay, "lr": lr, "weight_decay": wd})
        if nodecay:
            out.append({"params": nodecay, "lr": lr, "weight_decay": 0.0})
        return out

    param_groups = []
    param_groups += pack(backbone, base_lr, weight_decay)
    param_groups += pack(head, base_lr * HEAD_LR_SCALE, weight_decay)
    param_groups += pack(branch, base_lr * BRANCH_LR_SCALE, 0.0)   # 分支先不 wd

    if is_main_process():
        print(
            f"[Stage2 Config] epochs={num_epochs}, warmup_epochs={warmup_epochs}, "
            f"per_gpu_bs={per_gpu_bs}, global_bs={global_batch}, "
            f"base_lr={base_lr:.3e}, weight_decay={weight_decay}, ema_decay={ema_decay}",
            flush=True
        )


    optimizer = torch.optim.AdamW(param_groups)

    updates_per_epoch  = math.ceil(len(train_loader) / accum_steps)
    num_training_steps = num_epochs * updates_per_epoch
    warmup_steps       = warmup_epochs * updates_per_epoch

    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    sch1 = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    sch2 = CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps - warmup_steps,
        eta_min=base_lr * 1e-2,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[sch1, sch2],
        milestones=[warmup_steps],
    )

    #scaler = torch.cuda.amp.GradScaler()
    scaler = torch.amp.GradScaler('cuda')
    ema_model = create_ema_model(model.module).to(device)

    # ---- 训练循环（无 Mixup/CutMix，直接 CE）----
    best_acc   = 0.0
    finetune_ckpt = os.environ.get("FINETUNE_CKPT", "vil_stage2_224.pth")
    global_step = 0

    BRANCH_START = max(warmup_epochs, kd_warmup_epochs)
    BRANCH_RAMP  = int(os.environ.get("BRANCH_RAMP", "6"))
    BRANCH_MAX   = float(os.environ.get("BRANCH_ALPHA_MAX", "3e-3"))

    for epoch in range(1, num_epochs + 1):
        a = get_branch_alpha(epoch, BRANCH_START, BRANCH_RAMP, BRANCH_MAX)
        with torch.no_grad():
            model.module.head_adapter.alpha.fill_(a)
            ema_model.head_adapter.alpha.fill_(a)

        train_sampler.set_epoch(epoch)
        model.train()
        running_loss, acc_hist = 0.0, []

        if is_main_process():
            print(f"[Branch] alpha={a:.3e} (start={BRANCH_START}, ramp={BRANCH_RAMP}, max={BRANCH_MAX})", flush=True)
            print(f"\n[Stage2-Train] Epoch {epoch}/{num_epochs}", flush=True)

        optimizer.zero_grad(set_to_none=True)

        for i, (imgs_s, imgs_t, target) in enumerate(train_loader, 1):
            imgs_s = imgs_s.to(device, non_blocking=True)
            imgs_t = imgs_t.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                logits_s = model(imgs_s)

                # hard label loss
                #loss_ce = F.cross_entropy(logits_s, target)
                loss_ce = F.cross_entropy(logits_s, target, label_smoothing=label_smooth)

            use_kd = (teacher is not None) and (epoch >= kd_warmup_epochs)
            if use_kd:
                with torch.no_grad():
                    with torch.amp.autocast('cuda', enabled=False):
                        logits_t = teacher(imgs_t.float())      # teacher 强制 fp32

                # 建议：KD 计算用 float32 更稳
                log_p_s = F.log_softmax((logits_s / kd_T).float(), dim=1)
                p_t     = F.softmax((logits_t / kd_T).float(), dim=1)
                loss_kd = F.kl_div(log_p_s, p_t, reduction="batchmean")

                #loss = (kd_alpha * loss_ce + (1.0 - kd_alpha) * (kd_T * kd_T) * loss_kd)
                hard_w = kd_hard_weight(epoch)
                soft_w = 1.0 - hard_w
                loss = hard_w * loss_ce + soft_w * (kd_T * kd_T) * loss_kd
            else:
                loss = loss_ce

            loss = loss / accum_steps

            scaler.scale(loss).backward()

            if i % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                update_ema(model.module, ema_model, ema_decay)

                scheduler.step()
                global_step += 1

            running_loss += loss.item() * accum_steps

            with torch.no_grad():
                pred = logits_s.argmax(1)
                acc_hist.append((pred == target).float().mean().item())

            if is_main_process() and (i % 100 == 0):
                avg_acc = sum(acc_hist) / len(acc_hist)
                print(
                    f"  iter {i:5d}/{len(train_loader)} | "
                    f"loss {loss.item()*accum_steps:.4f} | "
                    f"acc {avg_acc:.3f} | lr {scheduler.get_last_lr()[0]:.2e}",
                    flush=True
                )

        # ---- 验证：同时评估 raw 和 EMA ----
        raw_loss_g, raw_acc_g = evaluate_ddp(model.module, val_loader, device)
        ema_loss_g, ema_acc_g = evaluate_ddp(ema_model,    val_loader, device)

        # 当轮最优
        if ema_acc_g >= raw_acc_g:
            best_epoch_acc  = ema_acc_g
            best_epoch_loss = ema_loss_g
            best_tag = "EMA"
            best_sd  = ema_model.state_dict()
        else:
            best_epoch_acc  = raw_acc_g
            best_epoch_loss = raw_loss_g
            best_tag = "RAW"
            best_sd  = model.module.state_dict()

        if is_main_process():
            train_loss_epoch = running_loss / len(train_loader)
            train_acc_epoch  = sum(acc_hist) / len(acc_hist) if acc_hist else 0.0

            print(f"[Stage2 Epoch {epoch}] Train loss={train_loss_epoch:.4f}, acc={train_acc_epoch:.4f}", flush=True)
            print(f"[VAL] raw loss={raw_loss_g:.4f}, acc={raw_acc_g:.4f} | ema loss={ema_loss_g:.4f}, acc={ema_acc_g:.4f}", flush=True)
            print(f"[VAL] pick={best_tag} | loss={best_epoch_loss:.4f}, acc={best_epoch_acc:.4f} | lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

            if best_epoch_acc > best_acc + 1e-6:
                best_acc = best_epoch_acc
                bad_epochs = 0
                torch.save(ema_model.state_dict(), finetune_ckpt)
                print(f"  🌟 Stage2 New best saved @ {finetune_ckpt} (acc={best_acc:.4f})", flush=True)
            else:
                bad_epochs += 1

        #if epoch >= early_min_ep and bad_epochs >= early_patience:
        stop = 0
        if is_main_process() and (epoch >= early_min_ep) and (bad_epochs >= early_patience):
            stop = 1

        stop_t = torch.tensor([stop], device=device, dtype=torch.int32)
        dist.broadcast(stop_t, src=0)

        if stop_t.item() == 1:
            if is_main_process():
                print(f"[EarlyStop] stop at epoch={epoch}, best_acc={best_acc:.4f}", flush=True)
            break

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
