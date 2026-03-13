import os
import random
import collections
import itertools
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist


def load_yaml_config_if_present() -> None:
    """
    If CONFIG (or CFG) env var is set to a YAML file path, load it and
    populate os.environ with any keys that are not already set.

    This allows you to define default hyperparameters in a YAML file while
    still overriding any value via explicit environment variables.
    """
    config_path = os.environ.get("CONFIG") or os.environ.get("CFG")
    if not config_path:
        return
    if not os.path.isfile(config_path):
        raise RuntimeError(f"CONFIG points to non-existent file: {config_path}")
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"CONFIG is set ({config_path}) but PyYAML is not available. "
            f"Install it via `pip install pyyaml` to use YAML configs."
        ) from e
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise RuntimeError(f"YAML config at {config_path} must contain a mapping at top level.")
    for k, v in cfg.items():
        key = str(k)
        cur = os.environ.get(key)
        if cur is not None and str(cur).strip() != "":
            continue  # explicit env overrides config
        if isinstance(v, bool):
            os.environ[key] = "1" if v else "0"
        else:
            os.environ[key] = str(v)


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
    if v is None or str(v).strip() == "":
        return default
    return [int(x.strip()) for x in str(v).split(sep) if x.strip()]


# ----------------- Reproducibility -----------------
def set_global_seed(seed: int) -> None:
    """
    Set torch / numpy / random seeds for reproducibility.
    Call once after DDP init.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # If you need full determinism, you can enable:
    # torch.use_deterministic_algorithms(True)


def worker_init_fn(worker_id: int, base_seed: int) -> None:
    """
    Per-worker RNG seed so that augmentations (RandomCrop, etc.) are deterministic.

    Call with base_seed = data_seed + get_rank() * 10000 so each rank's workers
    get distinct but reproducible seeds.
    """
    seed = base_seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


# ----------------- DDP helpers -----------------
def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def ddp_print(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs, flush=True)

