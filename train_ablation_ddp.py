# train_ablation_ddp.py
# PSWF / VisionLSTM / ViT training harness (DDP).
#
# Features:
# - unified DATASET: ImageNet-1K / Tiny-ImageNet
# - optional ImageNet-C or Tiny-ImageNet-C evaluation (eval-only or post-train)
# - per-epoch wall-clock + throughput logging (images/s), JSONL metrics dump
# - auto plots (acc vs epoch/step/time) on rank0
# - ablation IDs extended: W3, W3_POOL_ONLY, etc.
# - optional MODEL_KIND: vil (VisionLSTM2) / vit_tiny (minimal ViT-T), plus a stub hook for mambavision
#
# Usage (example, ImageNet-1K @192):
#   export DATASET=imagenet DATA_ROOT=/path/to/imagenet
#   export ABLATION=W3 DWT_FUSE=add DISABLE_BRANCH=1
#   export IMG_SIZE=192 EPOCHS=200 PER_GPU_BATCH=32 ACCUM_STEPS=1 AMP_DTYPE=bf16
#   torchrun --nproc_per_node=8 train_ablation_ddp.py
#
# Tiny-ImageNet (train+val):
#   export DATASET=tiny_imagenet DATA_ROOT=/path/to/tiny-imagenet-200
#   export IMG_SIZE=64 EPOCHS=300 PER_GPU_BATCH=128
#
# ImageNet-C / Tiny-ImageNet-C evaluation (eval-only):
#   export MODE=eval_imagenetc IMAGENETC_ROOT=/path/to/imagenet-c-or-tinyc
#   export CKPT=/path/to/ema_best.pth
#   torchrun --nproc_per_node=1 train_ablation_ddp.py

import os
import math
import time
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
from torch.amp import autocast, GradScaler

from env_ddp import (
    load_yaml_config_if_present,
    env_bool,
    env_int,
    env_float,
    env_str,
    env_list_int,
    set_global_seed,
    is_dist,
    get_rank,
    get_world_size,
    is_main_process,
    ddp_print,
)
from data_loader import build_datasets_and_loaders
from train_helpers import (
    mixup_cutmix,
    soft_cross_entropy,
    create_ema_model,
    update_ema,
    get_branch_alpha,
    try_set_head_alpha,
)
from eval_helpers import evaluate, evaluate_imagenet_c
from logging_helpers import save_json, append_jsonl, plot_metrics
from model_builder import build_model_from_env


# ----------------- Perf defaults -----------------
cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True


def get_wavelet_monitor_stats(model: nn.Module) -> Dict[str, float]:
    getter = getattr(model, "get_wavelet_monitor_stats", None)
    if not callable(getter):
        return {}
    try:
        stats = getter() or {}
    except Exception:
        return {}
    out: Dict[str, float] = {}
    for key, value in stats.items():
        try:
            out[str(key)] = float(value)
        except Exception:
            continue
    return out


def summarize_wavelet_monitor(stats: Dict[str, float]) -> str:
    if not stats:
        return ""

    def _pick_suffix(suffix: str):
        for key in sorted(stats.keys()):
            if key.endswith(suffix):
                return key, stats[key]
        return None, None

    parts = []
    token_key, token_val = _pick_suffix("token_delta_over_pool")
    if token_key is not None:
        parts.append(f"{token_key}={token_val:.3e}")

    head_key, head_val = _pick_suffix("head_delta_over_input")
    if head_key is not None:
        parts.append(f"{head_key}={head_val:.3e}")

    scale_key, scale_val = _pick_suffix("head_effective_scale")
    if scale_key is not None:
        parts.append(f"{scale_key}={scale_val:.3e}")

    gate_key, gate_val = _pick_suffix("head_gate_abs_mean")
    if gate_key is not None:
        parts.append(f"{gate_key}={gate_val:.3e}")

    return " | ".join(parts)






# ----------------- Main -----------------
def main():
    # Optionally load defaults from a YAML config specified via CONFIG / CFG.
    load_yaml_config_if_present()

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
    log_every = env_int("LOG_EVERY", 100)
    wavelet_monitor = env_bool("WAVELET_MONITOR", True)
    wavelet_monitor_log_every = env_int("WAVELET_MONITOR_LOG_EVERY", log_every)

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

    # ----- Build datasets & loaders -----
    dataset_name = env_str("DATASET", "imagenet").lower()
    data_root = env_str("DATA_ROOT", env_str("IMAGENET_ROOT", ""))
    if not data_root:
        raise RuntimeError("Set DATA_ROOT (or IMAGENET_ROOT for backward compat).")

    train_loader, val_loader, train_sampler, num_classes, data_info = build_datasets_and_loaders(
        dataset_name=dataset_name,
        data_root=data_root,
        img_size=img_size,
        per_gpu_batch=per_gpu_batch,
        num_workers=num_workers,
        data_seed=data_seed,
        device=device,
    )

    # ----- Output (after dataset/model config is known) -----
    out_dir = env_str("OUT_DIR", "./outputs_psf")
    os.makedirs(out_dir, exist_ok=True)
    tag = env_str(
        "RUN_TAG",
        f"{dataset_name}_{model_kind}_{ablation_id}_img{img_size}_dim{dim}_d{depth}_ch{'-'.join(map(str, feat_ch))}_{dwt_fuse}",
    )
    run_dir = os.path.join(out_dir, tag)
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    config_path = os.path.join(run_dir, "config.json")
    ckpt_path = os.path.join(run_dir, "ema_best.pth")

    if is_main_process():
        ddp_print(f"[Run] dir={run_dir}")

    # ----- Build model -----
    model, _ = build_model_from_env(num_classes=num_classes, img_size=img_size)

    model.to(device)

    # Optional: report params on rank0
    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters())
        ddp_print(f"[Model] kind={model_kind} params={n_params/1e6:.3f}M | dwt_fuse={dwt_fuse} | ablation={ablation_id}")

    # Optional: warm-start training from an existing checkpoint (weights only).
    resume_ckpt = env_str("RESUME_CKPT", "").strip()
    if mode == "train" and resume_ckpt:
        if os.path.isfile(resume_ckpt):
            ddp_print(f"[Resume] Loading weights from RESUME_CKPT={resume_ckpt}")
            sd = torch.load(resume_ckpt, map_location="cpu")
            # Accept both plain state_dict and {name: tensor} style checkpoints.
            if isinstance(sd, dict) and not any(torch.is_tensor(v) for v in sd.values()):
                state = sd
            elif isinstance(sd, dict) and any(torch.is_tensor(v) for v in sd.values()):
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
            if os.path.isfile(ckpt_path):
                ckpt = ckpt_path
                ddp_print(f"[Eval] CKPT not set, fallback to {ckpt}")
            else:
                raise RuntimeError(
                    "MODE=eval* requires CKPT=/path/to/checkpoint, "
                    "or a valid ema_best.pth in the current run directory."
                )
        sd = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(sd, strict=False)
        model.eval()
        if mode == "eval":
            loss, acc = evaluate(model, val_loader, device, amp_autocast_dtype)
            ddp_print(f"[Eval] val_loss={loss:.4f} acc={acc:.4f}")
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
            wavelet_monitor=wavelet_monitor, wavelet_monitor_log_every=wavelet_monitor_log_every,
            wavelet_scale_init=env_float("WAVELET_SCALE_INIT", 0.0),
            wavelet_warmup_steps=env_int("WAVELET_WARMUP_STEPS", 0),
            token_wavelet_scale_init=env_float("TOKEN_WAVELET_SCALE_INIT", 0.1),
            token_wavelet_shrink=env_float("TOKEN_WAVELET_SHRINK", 0.02),
            token_wavelet_hf_only=env_bool("TOKEN_WAVELET_HF_ONLY", True),
            token_wavelet_per_channel=env_bool("TOKEN_WAVELET_PER_CHANNEL", True),
            token_wavelet_hidden_ch=env_int("TOKEN_WAVELET_HIDDEN_CH", 0),
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
        wavelet_sums: Dict[str, float] = {}
        wavelet_steps = 0
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
                
                if hasattr(base_model, 'set_wavelet_global_step'):
                    base_model.set_wavelet_global_step(global_step)
                if hasattr(ema_model, 'set_wavelet_global_step'):
                    ema_model.set_wavelet_global_step(global_step)

            running_loss += loss.item() * accum_steps
            if wavelet_monitor and is_main_process():
                stats = get_wavelet_monitor_stats(base_model)
                if stats:
                    wavelet_steps += 1
                    for key, value in stats.items():
                        wavelet_sums[key] = wavelet_sums.get(key, 0.0) + float(value)

            with torch.no_grad():
                pred = logits.argmax(1)
                soft_acc = soft_targets.gather(1, pred.unsqueeze(1)).squeeze(1).mean().item()
                soft_acc_hist.append(soft_acc)

            if is_main_process() and (it % log_every == 0):
                dt = max(1e-6, time.time() - it_t0)
                it_t0 = time.time()
                avg_soft = float(sum(soft_acc_hist) / max(1, len(soft_acc_hist)))
                lr0 = scheduler.get_last_lr()[0]
                # approximate global throughput
                imgs_seen = per_gpu_batch * get_world_size() * log_every
                ips = imgs_seen / dt
                ddp_print(f"  iter {it:5d}/{len(train_loader)} | loss {loss.item()*accum_steps:.4f} | soft_acc {avg_soft:.3f} | lr {lr0:.2e} | {ips:.0f} img/s")
                if wavelet_monitor and wavelet_steps > 0 and (it % max(1, wavelet_monitor_log_every) == 0):
                    wavelet_avg = {k: v / wavelet_steps for k, v in wavelet_sums.items()}
                    wavelet_msg = summarize_wavelet_monitor(wavelet_avg)
                    if wavelet_msg:
                        ddp_print(f"    [Wavelet] {wavelet_msg}")

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
        wavelet_avg = {k: v / wavelet_steps for k, v in wavelet_sums.items()} if wavelet_steps > 0 else {}

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
            rec.update(wavelet_avg)
            append_jsonl(metrics_path, rec)

            ddp_print(f"[Epoch {epoch}] Train loss={train_loss_epoch:.4f}, soft_acc={train_soft_acc:.4f}")
            ddp_print(f"[Epoch {epoch}] Val   loss={val_loss_g:.4f}, acc={val_acc_g:.4f} | epoch_sec={epoch_sec:.1f} | elapsed={elapsed/3600:.2f}h")
            if wavelet_avg:
                wavelet_msg = summarize_wavelet_monitor(wavelet_avg)
                if wavelet_msg:
                    ddp_print(f"[Epoch {epoch}] Wavelet {wavelet_msg}")

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
