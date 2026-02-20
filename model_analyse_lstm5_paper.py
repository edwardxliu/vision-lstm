#!/usr/bin/env python3
"""
Checkpoint param analyser aligned with paper codepath.

- Builds the model the same way as model_compute_lstm5_paper.py (prefers importing your stage1_paper script).
- Loads a checkpoint (.pth), optionally preferring EMA weights.
- Prints missing/unexpected keys + parameter totals + top-level breakdown.

Usage (match your training env vars first, then):
  python model_analyse_lstm5_paper.py --ckpt /path/to/best.pth --prefer-ema
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import os
from typing import Any, Dict, Optional, Tuple, List

import torch


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
    if len(raw) == 1:
        return [raw[0], raw[0], raw[0]]
    if len(raw) == 2:
        return [raw[0], raw[1], raw[1]]
    return raw



def infer_num_classes_from_env(default: int = 1000) -> int:
    """Infer num_classes consistent with your stage1_paper conventions.
    Priority:
      1) env NUM_CLASSES (if set)
      2) env SUBSET_CLASSES (if set and >0)
      3) env DATASET heuristic: tiny_imagenet -> 200, imagenet -> 1000
      4) fallback default
    """
    v = os.getenv("NUM_CLASSES")
    if v is not None and v.strip() != "":
        try:
            return int(v)
        except Exception:
            pass
    v = os.getenv("SUBSET_CLASSES")
    if v is not None and v.strip() != "":
        try:
            sc = int(v)
            if sc > 0:
                return sc
        except Exception:
            pass
    ds = env_str("DATASET", "").strip().lower()
    if ds in ("tiny_imagenet", "tiny-imagenet", "tinyimagenet", "tiny_imagenet_200"):
        return 200
    if ds in ("imagenet", "imagenet1k", "imagenet-1k", "imagenet_1k", "in1k"):
        return 1000
    return default

# ------------------------- model builder (same as compute script) -------------------------

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
    try:
        m = _import_module_from_path(train_script)
    except Exception as e:
        print(f"[build] Skip train-script import ({train_script}): {e}")
        return None

    candidates = ["build_model_from_env", "build_model", "create_model", "get_model", "make_model"]
    for name in candidates:
        fn = getattr(m, name, None)
        if fn is None or not callable(fn):
            continue
        try:
            out = fn()
        except TypeError:
            tries = [
                dict(),
                dict(num_classes = infer_num_classes_from_env(default=1000)),
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

        if isinstance(out, tuple) and len(out) >= 1 and isinstance(out[0], torch.nn.Module):
            model = out[0]
            cfg = out[1] if len(out) > 1 and isinstance(out[1], dict) else {}
            return model, cfg, name
        if isinstance(out, torch.nn.Module):
            return out, {}, name

    print(f"[build] Train-script imported but no known builder found in {train_script}")
    return None

def resolve_ablation_flags(ablation: str) -> Dict[str, Any]:
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
        return dict(use_conv_stem=False, use_dwt=False, pre_patch_dwt=True, post_stem_dwt=False)
    return {}

def build_vil_fallback() -> Tuple[torch.nn.Module, Dict[str, Any], str]:
    try:
        from vision_lstm5_mod4_paper import VisionLSTM2  # type: ignore
    except Exception as e:
        raise ImportError(
            "Failed to import vision_lstm5_mod4_paper.VisionLSTM2. "
            "Make sure vision_lstm5_mod4_paper.py is in PYTHONPATH / current dir."
        ) from e

    img_size = env_int("IMG_SIZE", 192)
    num_classes = infer_num_classes_from_env(default=1000)
    dim = env_int("DIM", 192)
    depth = env_int("DEPTH", 12)
    patch = env_int("PATCH_SIZE", 16)
    stride = env_int("STRIDE", patch)
    feat_ch = normalize_feat_ch(env_int_list("FEAT_CH", [32, 64, 64]))

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
        dwt_fuse=env_str("DWT_FUSE", "add"),
        auto_patch_dwt=env_bool("AUTO_PATCH_DWT", True),
        use_conv_stem=env_bool("USE_CONV_STEM", True),
        use_dwt=env_bool("USE_DWT", False),
        pre_patch_dwt=env_bool("PRE_PATCH_DWT", False),
        post_stem_dwt=env_bool("POST_STEM_DWT", False),
        post_stem_merge=env_str("POST_STEM_MERGE", "concat"),
        disable_branch=env_bool("DISABLE_BRANCH", True),
        head_inject_gated=env_bool("HEAD_INJECT_GATED", True),
        head_gate_hidden_ratio=env_float("HEAD_GATE_HIDDEN_RATIO", 0.0),
        head_gate_init_bias=env_float("HEAD_GATE_INIT_BIAS", -2.0),
        attn_pool_heads=env_int("ATTN_POOL_HEADS", 4),
    )
    cfg.update(flags)

    sig = inspect.signature(VisionLSTM2.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    cfg_f = {k: v for k, v in cfg.items() if k in allowed}

    model = VisionLSTM2(**cfg_f)
    return model, cfg_f, "fallback_vil"

def build_model() -> Tuple[torch.nn.Module, Dict[str, Any], str]:
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


# ------------------------- checkpoint helpers -------------------------

def extract_state_dict(ckpt_obj: Any, prefer_ema: bool = False) -> Tuple[Dict[str, torch.Tensor], str]:
    if isinstance(ckpt_obj, dict):
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

        if any(torch.is_tensor(t) for t in ckpt_obj.values()):
            return ckpt_obj, "(root)"

    raise ValueError("Could not find a state_dict in the checkpoint dict.")

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

    model, cfg, src = build_model()

    ckpt_obj = torch.load(args.ckpt, map_location="cpu")
    sd, sd_src = extract_state_dict(ckpt_obj, prefer_ema=args.prefer_ema)

    missing, unexpected = model.load_state_dict(sd, strict=bool(args.strict))

    total_params = int(sum(int(p.numel()) for p in model.parameters()))
    trainable_params = int(sum(int(p.numel()) for p in model.parameters() if p.requires_grad))
    sd_params = count_state_dict_params(sd)

    print("\n================ Build Info ================")
    print(f"builder: {src}")
    print(f"MODEL_KIND: {env_str('MODEL_KIND','vil')}")
    print(f"ABLATION:   {env_str('ABLATION','')}")
    print(f"DWT_FUSE:   {env_str('DWT_FUSE','')}")
    print(f"FEAT_CH:    {env_str('FEAT_CH','')}")
    print(f"TRAIN_SCRIPT: {env_str('TRAIN_SCRIPT','lstm5_stage1_pretrain_192_sample_ablation_paper.py')}")

    print("\n================ Model Config (effective kwargs) ================")
    for k in sorted(cfg.keys()):
        print(f"{k}: {cfg[k]}")

    print("\n================ Checkpoint ================")
    print(f"ckpt_path: {args.ckpt}")
    print(f"state_dict_source: {sd_src}")
    print(f"state_dict_tensor_params: {sd_params:,d}  ({sd_params/1e6:.3f} M)")

    print("\n================ Load Report ================")
    print(f"strict: {bool(args.strict)}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")
    if len(missing) > 0:
        print("  [missing]", "\n  ".join(missing[:30]), "..." if len(missing) > 30 else "")
    if len(unexpected) > 0:
        print("  [unexpected]", "\n  ".join(unexpected[:30]), "..." if len(unexpected) > 30 else "")

    print("\n================ Parameter Count (model instance) ================")
    print(f"Total parameters:     {total_params:,d}")
    print(f"Trainable parameters: {trainable_params:,d}")
    print(f"Total (Millions):     {total_params/1e6:.3f} M")

    # Top-level breakdown (best-effort; names may differ by MODEL_KIND)
    breakdown = [
        ("pre_patch", getattr(model, "pre_patch", None)),
        ("feature_extractor", getattr(model, "feature_extractor", None)),
        ("feature_extractor_branch", getattr(model, "feature_extractor_branch", None)),
        ("patch_embed", getattr(model, "patch_embed", None)),
        ("pos_embed", getattr(model, "pos_embed", None)),
        ("blocks", getattr(model, "blocks", None)),
        ("mixers", getattr(model, "mixers", None)),
        ("attn_pool", getattr(model, "attn_pool", None)),
        ("norm", getattr(model, "norm", None)),
        ("head", getattr(model, "head", None)),
        ("head_adapter", getattr(model, "head_adapter", None)),
    ]
    print("\n================ Parameter Breakdown (top-level) ================")
    for name, mod in breakdown:
        n = count_module_params(mod)
        if n > 0:
            print(f"{name:>24s}: {n:,d}  ({n/1e6:.3f} M)")


if __name__ == "__main__":
    main()
