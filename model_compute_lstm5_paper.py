#!/usr/bin/env python3
"""
Compute Params & FLOPs for lstm5 paper codepath (vision_lstm5_mod4_paper + stage1_paper envs).

Design goals:
- Match the *training script* (lstm5_stage1_pretrain_192_sample_ablation_paper.py) if available:
  we dynamically import it and try to call its model builder (best fidelity for both vil/vit_tiny).
- If that fails, we fall back to a lightweight builder for MODEL_KIND=vil using vision_lstm5_mod4_paper.VisionLSTM2.

Typical usage (same env vars as training):
  export MODEL_KIND=vil IMG_SIZE=64 DIM=192 DEPTH=12 FEAT_CH=32 PATCH_SIZE=8 STRIDE=8 \
         AUTO_PATCH_DWT=1 ABLATION=W3 DWT_FUSE=add DISABLE_BRANCH=1
  python model_compute_lstm5_paper.py

You can also override image/batch/device:
  python model_compute_lstm5_paper.py --img 64 --bs 1 --device cuda
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import os
from typing import Any, Dict, Optional, Tuple, List

import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count_table


# ------------------------- env helpers -------------------------

def env_str(key: str, default: str = "") -> str:
    v = os.getenv(key)
    return default if v is None or v == "" else str(v)

def env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    return default if v is None or v == "" else int(v)

def env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    return default if v is None or v == "" else float(v)

def env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None or v == "":
        return default
    v = v.strip().lower()
    if v in ("1", "true", "t", "yes", "y", "on"):
        return True
    if v in ("0", "false", "f", "no", "n", "off"):
        return False
    return True

def env_int_list(key: str, default: List[int]) -> List[int]:
    v = os.getenv(key)
    if v is None or v.strip() == "":
        return list(default)
    parts = [p.strip() for p in v.replace(";", ",").split(",") if p.strip() != ""]
    return [int(p) for p in parts]

def normalize_feat_ch(raw: List[int]) -> List[int]:
    """
    Training里你经常写 FEAT_CH=32 或 FEAT_CH=32,64,64。
    这里做一个稳健规范化：
      - 1个数：扩成 [x, x, x]
      - 2个数：扩成 [a, b, b]
      - 3个数：原样
      - >3：原样（交给模型自己处理）
    """
    if len(raw) == 1:
        return [raw[0], raw[0], raw[0]]
    if len(raw) == 2:
        return [raw[0], raw[1], raw[1]]
    return raw

# ------------------------- try build from training script -------------------------

def _import_module_from_path(py_path: str):
    py_path = os.path.abspath(py_path)
    if not os.path.exists(py_path):
        raise FileNotFoundError(py_path)
    mod_name = os.path.splitext(os.path.basename(py_path))[0] + "_dyn"
    spec = importlib.util.spec_from_file_location(mod_name, py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module

def try_build_model_via_train_script(train_script: str) -> Optional[Tuple[torch.nn.Module, Dict[str, Any], str]]:
    """
    Best-effort: import stage1 paper script and call its model builder.
    Returns (model, cfg, builder_name) or None.
    """
    try:
        m = _import_module_from_path(train_script)
    except Exception as e:
        print(f"[build] Skip train-script import ({train_script}): {e}")
        return None

    # Candidate builder names (common in your scripts)
    candidates = [
        "build_model_from_env",
        "build_model",
        "create_model",
        "get_model",
        "make_model",
    ]
    for name in candidates:
        fn = getattr(m, name, None)
        if fn is None or not callable(fn):
            continue
        try:
            out = fn()
        except TypeError:
            # Some builders want args; try a few common ones
            tries = [
                dict(),
                dict(num_classes=env_int("NUM_CLASSES", env_int("SUBSET_CLASSES", 1000))),
                dict(img_size=env_int("IMG_SIZE", 192)),
            ]
            out = None
            for kw in tries:
                try:
                    out = fn(**kw)
                    break
                except Exception:
                    out = None
            if out is None:
                continue
        except Exception:
            continue

        # Normalize return
        if isinstance(out, tuple) and len(out) >= 1 and isinstance(out[0], torch.nn.Module):
            model = out[0]
            cfg = out[1] if len(out) > 1 and isinstance(out[1], dict) else {}
            return model, cfg, name
        if isinstance(out, torch.nn.Module):
            return out, {}, name

    print(f"[build] Train-script imported but no known builder found in {train_script}")
    return None


# ------------------------- fallback builder: vil only -------------------------

def resolve_ablation_flags(ablation: str) -> Dict[str, Any]:
    """
    Minimal mapping that matches你命令里用到的关键 ablation：
    - A0/A1/A2/A3：stem/DWT 路径（传统 baseline）
    - W3 / W3_POOL_ONLY：启用 pre_patch_dwt（PSWF tokenizer），stem 不用 post_stem_dwt
    """
    ab = (ablation or "").strip().upper()
    if ab == "A0":
        return dict(use_conv_stem=True, use_dwt=True, pre_patch_dwt=False, post_stem_dwt=True)
    if ab == "A1":
        return dict(use_conv_stem=True, use_dwt=False, pre_patch_dwt=False, post_stem_dwt=False)
    if ab == "A2":
        return dict(use_conv_stem=False, use_dwt=False, pre_patch_dwt=True, post_stem_dwt=False)
    if ab == "A3":
        return dict(use_conv_stem=False, use_dwt=False, pre_patch_dwt=False, post_stem_dwt=False)
    if ab in ("W3", "W3_POOL_ONLY"):
        # PSWF tokenizer uses pre_patch_dwt, which requires no conv stem in VisionLSTM2 paper code
        return dict(use_conv_stem=False, use_dwt=False, pre_patch_dwt=True, post_stem_dwt=False)
    return {}

def build_vil_fallback() -> Tuple[torch.nn.Module, Dict[str, Any], str]:
    """
    Fallback for MODEL_KIND=vil: instantiate VisionLSTM2 from vision_lstm5_mod4_paper.
    We also filter cfg keys by VisionLSTM2.__init__ signature to stay robust.
    """
    try:
        from vision_lstm5_mod4_paper import VisionLSTM2  # type: ignore
    except Exception as e:
        raise ImportError(
            "Failed to import vision_lstm5_mod4_paper.VisionLSTM2. "
            "Make sure vision_lstm5_mod4_paper.py is in PYTHONPATH / current dir."
        ) from e

    img_size = env_int("IMG_SIZE", 192)
    num_classes = env_int("NUM_CLASSES", env_int("SUBSET_CLASSES", 1000))
    dim = env_int("DIM", 192)
    depth = env_int("DEPTH", 12)
    patch = env_int("PATCH_SIZE", 16)
    stride = env_int("STRIDE", patch)

    feat_raw = env_int_list("FEAT_CH", [32, 64, 64])
    feat_ch = normalize_feat_ch(feat_raw)

    dwt_fuse = env_str("DWT_FUSE", "add")
    auto_patch_dwt = env_bool("AUTO_PATCH_DWT", True)

    ablation = env_str("ABLATION", "A1")
    flags = resolve_ablation_flags(ablation)

    cfg: Dict[str, Any] = dict(
        dim=dim,
        input_shape=(3, img_size, img_size),
        patch_size=patch,
        stride=stride,
        depth=depth,
        output_shape=(num_classes,),
        mode="classifier",
        pooling=env_str("POOLING", "bilateral_flatten"),
        drop_path_rate=env_float("DROP_PATH", 0.0),
        drop_path_decay=env_bool("DROP_PATH_DECAY", False),
        legacy_norm=env_bool("LEGACY_NORM", False),
        conv_kind=env_str("CONV_KIND", "2d"),
        conv_kernel_size=env_int("CONV_KERNEL", 3),
        proj_bias=env_bool("PROJ_BIAS", True),
        norm_bias=env_bool("NORM_BIAS", True),
        feature_extractor_channels=feat_ch,
        dwt_fuse=dwt_fuse,
        auto_patch_dwt=auto_patch_dwt,
        # path flags
        use_conv_stem=env_bool("USE_CONV_STEM", True),
        use_dwt=env_bool("USE_DWT", False),
        pre_patch_dwt=env_bool("PRE_PATCH_DWT", False),
        post_stem_dwt=env_bool("POST_STEM_DWT", False),
        post_stem_merge=env_str("POST_STEM_MERGE", "concat"),
        # head / branch
        disable_branch=env_bool("DISABLE_BRANCH", True),
        head_inject_gated=env_bool("HEAD_INJECT_GATED", True),
        head_gate_hidden_ratio=env_float("HEAD_GATE_HIDDEN_RATIO", 0.0),
        head_gate_init_bias=env_float("HEAD_GATE_INIT_BIAS", -2.0),
        attn_pool_heads=env_int("ATTN_POOL_HEADS", 4),
    )
    cfg.update(flags)

    # Filter by signature to avoid breaking if your model init changes
    sig = inspect.signature(VisionLSTM2.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    cfg_f = {k: v for k, v in cfg.items() if k in allowed}

    model = VisionLSTM2(**cfg_f)
    return model, cfg_f, "fallback_vil"


def build_model() -> Tuple[torch.nn.Module, Dict[str, Any], str]:
    """
    Build model in this priority:
      1) import training script and call its builder (highest fidelity)
      2) fallback_vil for MODEL_KIND=vil
    """
    train_script = env_str("TRAIN_SCRIPT", "lstm5_stage1_pretrain_192_sample_ablation_paper.py")
    via = try_build_model_via_train_script(train_script)
    if via is not None:
        return via

    mk = env_str("MODEL_KIND", "vil").strip().lower()
    if mk == "vil":
        return build_vil_fallback()

    raise RuntimeError(
        f"MODEL_KIND={mk} but could not build via train script ({train_script}). "
        "Set TRAIN_SCRIPT to your stage1 paper script path, or put it in current dir."
    )


# ------------------------- reporting -------------------------

def report_one(model: torch.nn.Module, img: int, bs: int, device: str) -> None:
    model = model.to(device)
    model.eval()

    x = torch.randn(bs, 3, img, img, device=device)
    with torch.no_grad():
        flops = FlopCountAnalysis(model, x)

    print(f"\n================ FLOPs Report ================")
    print(f"Input: {bs}x3x{img}x{img}  device={device}")
    print(parameter_count_table(model))
    print(flop_count_table(flops))
    total = flops.total()
    # fvcore may return float; format robustly
    try:
        total_str = f"{total:,.0f}"
    except Exception:
        total_str = str(total)
    print(f"Total (fvcore 'flops' ~= MACs): {total_str}")
    print(f"Total (G): {float(total)/1e9:.3f}")
    unsup = flops.unsupported_ops()
    if len(unsup) > 0:
        print("Unsupported ops (not counted):", unsup)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=int, default=0, help="override IMG_SIZE for FLOPs; 0 means use env IMG_SIZE")
    ap.add_argument("--bs", type=int, default=1, help="batch size for FLOPs")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="device for FLOPs")
    ap.add_argument("--extra-img", type=int, nargs="*", default=[], help="additional image sizes to report")
    args = ap.parse_args()

    model, cfg, src = build_model()
    base_img = args.img if args.img > 0 else env_int("IMG_SIZE", 192)
    imgs = [base_img] + [i for i in args.extra_img if i != base_img]

    print("================ Build Info ================")
    print(f"builder: {src}")
    print("env MODEL_KIND:", env_str("MODEL_KIND", "vil"))
    print("env ABLATION:", env_str("ABLATION", ""))
    print("env DWT_FUSE:", env_str("DWT_FUSE", ""))
    print("env FEAT_CH:", env_str("FEAT_CH", ""))
    print("effective cfg keys:", len(cfg))
    # print cfg compact
    for k in sorted(cfg.keys()):
        if k in ("input_shape", "output_shape"):
            print(f"{k}: {cfg[k]}")
        elif isinstance(cfg[k], (int, float, str, bool)):
            print(f"{k}: {cfg[k]}")
        else:
            print(f"{k}: {cfg[k]}")

    for img in imgs:
        report_one(model, img=img, bs=args.bs, device=args.device)


if __name__ == "__main__":
    main()
