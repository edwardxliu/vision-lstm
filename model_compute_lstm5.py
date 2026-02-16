#!/usr/bin/env python3
"""Parameter analysis / sanity checker for VisionLSTM5 (vision_lstm5_mod2).

Usage examples:
  # (1) Minimal: count params for a checkpoint (state_dict or full checkpoint dict)
  python model_analyse_lstm5.py --ckpt /path/to/lstm5_A0_ema_best.pth

  # (2) Match your training config via env vars (recommended)
  SUBSET_CLASSES=500 IMG_SIZE=192 PATCH_SIZE=16 STRIDE=16 AUTO_PATCH_DWT=1 \
  FEAT_CH=32,64,64 DIM=192 DEPTH=12 ABLATION=A0 \
  python model_analyse_lstm5.py --ckpt /path/to/ckpt.pth

  # (3) Prefer EMA weights if checkpoint contains both model + ema
  python model_analyse_lstm5.py --ckpt ckpt_full.pth --prefer-ema

Notes:
- For Stage-A ablations (A0-A3) we usually set DISABLE_BRANCH=1.
- If you load a checkpoint trained with branch enabled but instantiate DISABLE_BRANCH=1,
  you'll see many 'unexpected keys' for the branch/head_adapter.
"""

import argparse
import os
from typing import Dict, Any, Optional, Tuple

import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count_table

# Your modified VisionLSTM5 model
from vision_lstm5_mod4 import VisionLSTM2


def env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None or v == "":
        return default
    return int(v)


def env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    if v is None or v == "":
        return default
    return float(v)


def env_str(key: str, default: str) -> str:
    v = os.getenv(key)
    if v is None or v == "":
        return default
    return str(v)


def env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None or v == "":
        return default
    v = v.strip().lower()
    if v in ("1", "true", "t", "yes", "y", "on"):
        return True
    if v in ("0", "false", "f", "no", "n", "off"):
        return False
    # Fallback: non-empty string means True
    return True


def env_int_list(key: str, default) -> list:
    v = os.getenv(key)
    if v is None or v.strip() == "":
        return list(default)
    parts = [p.strip() for p in v.replace(";", ",").split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def resolve_path_flags(ablation_id: str) -> Dict[str, Any]:
    """Return the stem/DWT path flags for a baseline ablation (A0-A3).

    A0: conv stem + stem DWT
    A1: conv stem only
    A2: DWT-only (pre-patch DWT, no stem)
    A3: RGB-only (no stem, no DWT)
    """
    ab = (ablation_id or "").strip().upper()
    if ab == "A0":
        return dict(use_conv_stem=True, use_dwt=True, pre_patch_dwt=False)
    if ab == "A1":
        return dict(use_conv_stem=True, use_dwt=False, pre_patch_dwt=False)
    if ab == "A2":
        return dict(use_conv_stem=False, use_dwt=False, pre_patch_dwt=True)
    if ab == "A3":
        return dict(use_conv_stem=False, use_dwt=False, pre_patch_dwt=False)
    # If unknown, do not override; caller may use env vars.
    return {}


def build_model_from_env() -> Tuple[VisionLSTM2, Dict[str, Any]]:
    """Instantiate VisionLSTM2 using env vars (and optional ABLATION mapping)."""

    # ----- basics -----
    img_size = env_int("IMG_SIZE", 192)
    num_classes = env_int("NUM_CLASSES", env_int("SUBSET_CLASSES", 1000))

    dim = env_int("DIM", 192)
    depth = env_int("DEPTH", 12)

    patch_size = env_int("PATCH_SIZE", 16)
    stride = env_int("STRIDE", patch_size)

    # Stem / DWT
    feat_ch = env_int_list("FEAT_CH", default=[32, 64, 64])
    dwt_fuse = env_str("DWT_FUSE", "gated")
    auto_patch_dwt = env_bool("AUTO_PATCH_DWT", True)

    # Head / pooling / branch
    pooling = env_str("POOLING", "bilateral_flatten")
    disable_branch_env = os.getenv("DISABLE_BRANCH")
    head_inject_gated = env_bool("HEAD_INJECT_GATED", True)
    head_gate_hidden_ratio = env_float("HEAD_GATE_HIDDEN_RATIO", 0.0)
    head_gate_init_bias = env_float("HEAD_GATE_INIT_BIAS", -2.0)
    attn_pool_heads = env_int("ATTN_POOL_HEADS", 4)

    # Regularization / norms
    drop_path_rate = env_float("DROP_PATH", 0.0)
    drop_path_decay = env_bool("DROP_PATH_DECAY", False)
    legacy_norm = env_bool("LEGACY_NORM", False)

    conv_kind = env_str("CONV_KIND", "2d")
    conv_kernel_size = env_int("CONV_KERNEL", 3)
    proj_bias = env_bool("PROJ_BIAS", True)
    norm_bias = env_bool("NORM_BIAS", True)

    # ----- ablation mapping (optional) -----
    ablation = env_str("ABLATION", "")
    baseline = env_str("BASELINE_ABLATION", "A0")

    # Start with env defaults
    use_conv_stem = env_bool("USE_CONV_STEM", True)
    use_dwt = env_bool("USE_DWT", False)
    pre_patch_dwt = env_bool("PRE_PATCH_DWT", False)

    ablation_id = env_str("ABLATION", "A0").strip().upper()
    baseline_ablation = env_str("BASELINE_ABLATION", "A0").strip().upper()
    
    dim = env_int("DIM", 192)
    depth = env_int("DEPTH", 12)
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

    cfg = dict(
        dim=dim,
        input_shape=(3, img_size, img_size),
        patch_size=patch_base,
        depth=depth,
        output_shape=(num_classes,),
        mode="classifier",
        pooling="bilateral_flatten",
        drop_path_rate=drop_path,
        drop_path_decay=drop_path_decay,
        stride=stride_base,
        legacy_norm=legacy_norm,
        conv_kind=conv_kind,
        conv_kernel_size=conv_kernel,
        proj_bias=proj_bias,
        norm_bias=norm_bias,
        feature_extractor_channels=[32, 64, 64],
        use_dwt=False,
        dwt_fuse=dwt_fuse,
        auto_patch_dwt=auto_patch_dwt,
        use_conv_stem=True,
        pre_patch_dwt=False,
        disable_branch=True,
        head_inject_gated=True,
        head_gate_hidden_ratio=head_gate_hidden_ratio,
        head_gate_init_bias=head_gate_init_bias,
        post_stem_dwt=True,
        post_stem_merge="concat",
        attn_pool_heads=attn_pool_heads,
    )

    model = VisionLSTM2(**cfg)
    return model, cfg


def extract_state_dict(ckpt_obj: Any, prefer_ema: bool = False) -> Tuple[Dict[str, torch.Tensor], str]:
    """Try to extract a model state_dict from various checkpoint formats."""

    if isinstance(ckpt_obj, dict):
        # Common patterns
        candidates = []
        if prefer_ema:
            candidates += ["ema", "model_ema", "state_dict_ema", "ema_state_dict"]
        candidates += ["model", "state_dict", "net", "module"]
        if not prefer_ema:
            candidates += ["ema", "model_ema", "state_dict_ema", "ema_state_dict"]

        for k in candidates:
            v = ckpt_obj.get(k)
            if isinstance(v, dict) and any(torch.is_tensor(t) for t in v.values()):
                return v, k

        # Some checkpoints store in "model" but nested
        v = ckpt_obj.get("model")
        if isinstance(v, dict) and any(torch.is_tensor(t) for t in v.values()):
            return v, "model"

        # If dict itself looks like a state_dict
        if any(torch.is_tensor(t) for t in ckpt_obj.values()):
            return ckpt_obj, "(root)"

    raise ValueError(
        "Could not find a state_dict in the checkpoint. "
        "If this is a full checkpoint, make sure it contains keys like 'model' or 'state_dict'."
    )


def count_state_dict_params(sd: Dict[str, torch.Tensor]) -> int:
    return int(sum(int(v.numel()) for v in sd.values() if torch.is_tensor(v)))


def count_module_params(m: Optional[torch.nn.Module]) -> int:
    if m is None:
        return 0
    return int(sum(int(p.numel()) for p in m.parameters()))



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to .pth checkpoint")
    ap.add_argument("--prefer-ema", action="store_true", help="Prefer EMA weights if checkpoint contains them")
    ap.add_argument("--strict", action="store_true", help="Use strict=True when loading state_dict")
    args = ap.parse_args()

    model, cfg = build_model_from_env()

    model.eval()
    def report(img=224, bs=1, device="cpu"):
        x = torch.randn(bs, 3, img, img, device=device)
        with torch.no_grad():
            flops = FlopCountAnalysis(model.to(device), x)
        print(f"Input: {bs}x3x{img}x{img}")
        print(parameter_count_table(model))
        print(flop_count_table(flops))
        print("Total (fvcore 'flops' ~= MACs):", flops.total())
        print("Unsupported ops:", flops.unsupported_ops())  # 看看哪些算子没被统计
    # 224（stage2口径）
    report(img=224, bs=1, device="cpu")

    # 192（stage1口径）
    report(img=192, bs=1, device="cpu")

if __name__ == "__main__":
    main()
