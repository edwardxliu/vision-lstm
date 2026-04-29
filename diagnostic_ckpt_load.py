import argparse
import os
from typing import Dict, Iterable, List, Tuple

import torch

from env_ddp import load_yaml_config_if_present
from model_builder import build_model_from_env


SUSPICIOUS_PATTERNS = (
    "wavelet",
    "dwt",
    "post_stem",
    "pre_patch",
    "side",
    "side_token",
    "patch_embed_side",
    "wave_",
    "hf_",
    "mix",
    "mixer",
    "mixers",
)


def _extract_state_dict(obj) -> Dict[str, torch.Tensor]:
    """Accept plain state_dict or common checkpoint wrappers."""
    if isinstance(obj, dict):
        for key in ("ema", "ema_model", "model_ema", "model", "module", "state_dict"):
            inner = obj.get(key)
            if isinstance(inner, dict) and any(torch.is_tensor(v) for v in inner.values()):
                return inner
    if isinstance(obj, dict) and any(torch.is_tensor(v) for v in obj.values()):
        return obj
    raise TypeError("Checkpoint does not look like a state_dict or known checkpoint wrapper.")


def _is_suspicious(name: str) -> bool:
    low = name.lower()
    return any(pattern in low for pattern in SUSPICIOUS_PATTERNS)


def _print_key_list(title: str, keys: Iterable[str], limit: int = 0) -> None:
    keys = list(keys)
    print(f"{title} ({len(keys)}):")
    if not keys:
        return
    shown = keys if limit <= 0 else keys[:limit]
    for key in shown:
        print(f"  {key}")
    if limit > 0 and len(keys) > limit:
        print(f"  ... {len(keys) - limit} more")


def _compare_state_dicts(
    model_state: Dict[str, torch.Tensor],
    ckpt_state: Dict[str, torch.Tensor],
) -> Tuple[List[str], List[str], List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]]]:
    model_keys = set(model_state.keys())
    ckpt_keys = set(ckpt_state.keys())
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    shape_mismatch = []
    for key in sorted(model_keys & ckpt_keys):
        model_value = model_state[key]
        ckpt_value = ckpt_state[key]
        if torch.is_tensor(model_value) and torch.is_tensor(ckpt_value):
            model_shape = tuple(model_value.shape)
            ckpt_shape = tuple(ckpt_value.shape)
            if model_shape != ckpt_shape:
                shape_mismatch.append((key, model_shape, ckpt_shape))
    return missing, unexpected, shape_mismatch


def _print_shape_mismatches(
    title: str,
    mismatches: Iterable[Tuple[str, Tuple[int, ...], Tuple[int, ...]]],
    limit: int = 0,
) -> None:
    mismatches = list(mismatches)
    print(f"{title} ({len(mismatches)}):")
    if not mismatches:
        return
    shown = mismatches if limit <= 0 else mismatches[:limit]
    for key, model_shape, ckpt_shape in shown:
        print(f"  {key}: model={model_shape} ckpt={ckpt_shape}")
    if limit > 0 and len(mismatches) > limit:
        print(f"  ... {len(mismatches) - limit} more")


def _set_env_if_present(name: str, value) -> None:
    if value is not None:
        os.environ[name] = str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the model from env/config, load a checkpoint, and report "
            "missing/unexpected/shape-mismatched keys."
        )
    )
    parser.add_argument("--ckpt", default=os.environ.get("CKPT", ""), help="Checkpoint path. Defaults to CKPT env.")
    parser.add_argument("--config", default=os.environ.get("CONFIG", ""), help="YAML config path to load before building.")
    parser.add_argument("--model-kind", default=None, help="Override MODEL_KIND, e.g. vil.")
    parser.add_argument("--ablation", default=None, help="Override ABLATION, e.g. W3_TOKENONLY.")
    parser.add_argument("--dwt-fuse", default=None, help="Override DWT_FUSE, e.g. add/none.")
    parser.add_argument("--img-size", type=int, default=None, help="Override IMG_SIZE.")
    parser.add_argument("--num-classes", type=int, default=None, help="Number of classes used for model head.")
    parser.add_argument("--print-limit", type=int, default=0, help="Limit printed keys per section; 0 means no limit.")
    parser.add_argument("--strict", action="store_true", help="Also try strict=True loading after diagnostics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config:
        os.environ["CONFIG"] = args.config
    load_yaml_config_if_present()

    _set_env_if_present("MODEL_KIND", args.model_kind)
    _set_env_if_present("ABLATION", args.ablation)
    _set_env_if_present("DWT_FUSE", args.dwt_fuse)
    _set_env_if_present("IMG_SIZE", args.img_size)
    _set_env_if_present("NUM_CLASSES", args.num_classes)

    ckpt = args.ckpt or os.environ.get("CKPT", "")
    if not ckpt:
        raise SystemExit("Set --ckpt or CKPT.")
    if not os.path.isfile(ckpt):
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    img_size = args.img_size
    if img_size is None and os.environ.get("IMG_SIZE"):
        img_size = int(os.environ["IMG_SIZE"])
    num_classes = args.num_classes
    if num_classes is None and os.environ.get("NUM_CLASSES"):
        num_classes = int(os.environ["NUM_CLASSES"])

    print("=== Build config ===")
    for key in (
        "CONFIG",
        "MODEL_KIND",
        "ABLATION",
        "DWT_FUSE",
        "IMG_SIZE",
        "NUM_CLASSES",
        "FEAT_CH",
        "TOKEN_WAVELET_HIDDEN_CH",
        "TOKEN_WAVELET_SIDE_CH",
        "TOKEN_WAVELET_SIDE_MODE",
        "TOKEN_WAVELET_SPLIT_BANDS",
        "WAVELET_INPUT_IMAGE",
    ):
        print(f"{key}={os.environ.get(key, '')}")

    print("\n=== Loading checkpoint ===")
    print(f"ckpt={ckpt}")
    raw = torch.load(ckpt, map_location="cpu")
    state = _extract_state_dict(raw)
    print(f"checkpoint tensors={sum(1 for v in state.values() if torch.is_tensor(v))}")

    model, cfg = build_model_from_env(num_classes=num_classes, img_size=img_size)
    model_state = model.state_dict()
    print(f"model tensors={len(model_state)}")
    print(f"builder_cfg={cfg}")

    missing, unexpected, shape_mismatch = _compare_state_dicts(model_state, state)
    suspicious_missing = [key for key in missing if _is_suspicious(key)]
    suspicious_unexpected = [key for key in unexpected if _is_suspicious(key)]
    suspicious_shape = [item for item in shape_mismatch if _is_suspicious(item[0])]

    print("\n=== Manual key comparison ===")
    _print_key_list("Missing keys", missing, args.print_limit)
    _print_key_list("Unexpected keys", unexpected, args.print_limit)
    _print_shape_mismatches("Shape mismatches", shape_mismatch, args.print_limit)

    print("\n=== Suspicious architecture keys ===")
    _print_key_list("Suspicious missing", suspicious_missing, args.print_limit)
    _print_key_list("Suspicious unexpected", suspicious_unexpected, args.print_limit)
    _print_shape_mismatches("Suspicious shape mismatches", suspicious_shape, args.print_limit)

    print("\n=== PyTorch load_state_dict(strict=False) ===")
    try:
        incompatible = model.load_state_dict(state, strict=False)
        _print_key_list("PyTorch missing", incompatible.missing_keys, args.print_limit)
        _print_key_list("PyTorch unexpected", incompatible.unexpected_keys, args.print_limit)
    except RuntimeError as exc:
        print("RuntimeError during strict=False load:")
        print(exc)

    if args.strict:
        print("\n=== PyTorch load_state_dict(strict=True) ===")
        try:
            model.load_state_dict(state, strict=True)
            print("strict=True load succeeded.")
        except RuntimeError as exc:
            print("RuntimeError during strict=True load:")
            print(exc)

    has_wavelet_issue = bool(suspicious_missing or suspicious_unexpected or suspicious_shape)
    print("\n=== Summary ===")
    print(f"total_missing={len(missing)}")
    print(f"total_unexpected={len(unexpected)}")
    print(f"total_shape_mismatch={len(shape_mismatch)}")
    print(f"suspicious_arch_issue={int(has_wavelet_issue)}")
    if has_wavelet_issue:
        print("WARNING: wavelet/DWT/post_stem/side/mix-related checkpoint mismatch detected.")
    else:
        print("No suspicious wavelet/DWT/post_stem/side/mix mismatch detected.")


if __name__ == "__main__":
    main()
