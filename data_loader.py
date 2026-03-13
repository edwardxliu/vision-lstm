import os
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from env_ddp import (
    is_dist,
    get_rank,
    get_world_size,
    is_main_process,
    ddp_print,
    env_int,
)
from env_ddp import worker_init_fn


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".JPEG")


class TinyImageNetCDataset(Dataset):
    """
    Tiny-ImageNet-C: root/<corruption>/<severity>/<wnid>/*.JPEG
    Labels are mapped using wnids.txt order from tiny-imagenet-200.
    """

    def __init__(self, severity_root: str, wnids_path: str, transform=None):
        self.severity_root = Path(severity_root)
        self.transform = transform

        with open(wnids_path, "r") as f:
            wnids = [x.strip() for x in f.read().splitlines() if x.strip()]
        self.wnid_to_idx = {w: i for i, w in enumerate(wnids)}

        self.samples: List[Tuple[str, int]] = []
        # iterate by wnids.txt order to guarantee deterministic mapping
        for wnid in wnids:
            d = self.severity_root / wnid
            if not d.is_dir():
                continue
            for p in d.rglob("*"):
                if p.suffix.lower() in [".jpeg", ".jpg", ".png"]:
                    self.samples.append((str(p), self.wnid_to_idx[wnid]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class PathLabelDataset(Dataset):
    """A minimal ImageFolder-like dataset that holds (path, label) samples."""

    def __init__(self, samples: List[Tuple[str, int]], classes: List[str], transform=None):
        self.samples = samples
        self.targets = [y for _, y in samples]
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, y


def load_tiny_imagenet(root: str, split: str, transform=None) -> PathLabelDataset:
    """
    root: tiny-imagenet-200 directory (contains train/, val/, wnids.txt)
    """
    wnids_path = os.path.join(root, "wnids.txt")
    if not os.path.isfile(wnids_path):
        raise RuntimeError(f"Tiny-ImageNet root missing wnids.txt: {wnids_path}")
    with open(wnids_path, "r") as f:
        wnids = [ln.strip() for ln in f if ln.strip()]
    class_to_idx = {w: i for i, w in enumerate(wnids)}

    samples: List[Tuple[str, int]] = []
    if split == "train":
        for w in wnids:
            img_dir = os.path.join(root, "train", w, "images")
            if not os.path.isdir(img_dir):
                continue
            for fn in os.listdir(img_dir):
                if fn.endswith(IMG_EXTS):
                    samples.append((os.path.join(img_dir, fn), class_to_idx[w]))
    elif split == "val":
        ann_path = os.path.join(root, "val", "val_annotations.txt")
        img_dir = os.path.join(root, "val", "images")
        if not os.path.isfile(ann_path):
            raise RuntimeError(f"Tiny-ImageNet val_annotations.txt missing: {ann_path}")
        mapping = {}
        with open(ann_path, "r") as f:
            for ln in f:
                parts = ln.strip().split("\t")
                if len(parts) >= 2:
                    mapping[parts[0]] = parts[1]
        for fn, w in mapping.items():
            p = os.path.join(img_dir, fn)
            if os.path.isfile(p) and w in class_to_idx:
                samples.append((p, class_to_idx[w]))
    else:
        raise ValueError("split must be 'train' or 'val'")
    return PathLabelDataset(samples=samples, classes=wnids, transform=transform)


def stable_hash_u64(s: str, seed: int) -> int:
    """Deterministic per-path hashing for per-class sampling (stable across ranks/machines)."""
    h = hashlib.blake2b(digest_size=8)
    h.update(str(seed).encode("utf-8"))
    h.update(b"::")
    h.update(s.encode("utf-8"))
    return int.from_bytes(h.digest(), byteorder="little", signed=False)


def filter_samples(
    samples: List[Tuple[str, int]],
    keep_class_indices: List[int],
    per_class_cap: int = 0,
    cap_seed: int = 0,
) -> Tuple[List[Tuple[str, int]], int]:
    """
    Filter + remap labels to 0..K-1. Optional per-class cap with stable hashing.
    """
    keep_set = set(keep_class_indices)
    remap = {old_i: new_i for new_i, old_i in enumerate(keep_class_indices)}

    tmp: List[Tuple[str, int]] = []
    for path, y in samples:
        if y in keep_set:
            tmp.append((path, remap[y]))

    if per_class_cap and per_class_cap > 0:
        buckets: List[List[Tuple[int, Tuple[str, int]]]] = [[] for _ in range(len(keep_class_indices))]
        for path, y in tmp:
            hv = stable_hash_u64(path, cap_seed)
            buckets[y].append((hv, (path, y)))
        tmp2: List[Tuple[str, int]] = []
        for b in buckets:
            b.sort(key=lambda t: t[0])
            tmp2.extend([s for _, s in b[:per_class_cap]])
        tmp = tmp2

    return tmp, len(keep_class_indices)


def select_subset_classes(total_classes: int, subset_k: int, seed: int, device: torch.device) -> List[int]:
    """
    Pick K class indices on rank0 then broadcast to all ranks (DDP-safe).
    """
    if subset_k <= 0 or subset_k >= total_classes:
        return list(range(total_classes))

    if not is_dist() or get_rank() == 0:
        rng = np.random.RandomState(seed)
        keep = list(range(total_classes))
        rng.shuffle(keep)
        keep = sorted(keep[:subset_k])
        keep_tensor = torch.tensor(keep, device=device, dtype=torch.int64)
    else:
        keep_tensor = torch.empty((subset_k,), device=device, dtype=torch.int64)

    if is_dist():
        torch.distributed.broadcast(keep_tensor, src=0)

    keep = keep_tensor.tolist()
    keep.sort()
    return keep


def build_datasets_and_loaders(
    dataset_name: str,
    data_root: str,
    img_size: int,
    per_gpu_batch: int,
    num_workers: int,
    data_seed: int,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader, Optional[DistributedSampler], int, Dict]:
    """
    High-level helper that builds train/val datasets and DataLoaders, including:
      - standard ImageNet / Tiny-ImageNet transforms
      - optional SUBSET_CLASSES / per-class caps via env vars
      - DDP-aware samplers and worker seeds
    Returns:
      train_loader, val_loader, num_classes, info_dict
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if dataset_name == "imagenet":
        train_tf = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
            ]
        )
    else:
        # Tiny-ImageNet style (64x64): keep spatial structure, avoid aggressive random-resize
        train_tf = transforms.Compose(
            [
                transforms.RandomCrop(img_size, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std),
            ]
        )

    val_tf = transforms.Compose(
        [
            transforms.Resize(224 if dataset_name == "imagenet" else img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )

    # Base datasets
    if dataset_name == "imagenet":
        train_base = datasets.ImageFolder(root=os.path.join(data_root, "train"), transform=train_tf)
        val_base = datasets.ImageFolder(root=os.path.join(data_root, "val"), transform=val_tf)
        classes = train_base.classes
        train_samples = list(train_base.samples)
        val_samples = list(val_base.samples)
    elif dataset_name in ("tiny_imagenet", "tiny-imagenet"):
        train_base = load_tiny_imagenet(data_root, "train", transform=train_tf)
        val_base = load_tiny_imagenet(data_root, "val", transform=val_tf)
        classes = train_base.classes
        train_samples = list(train_base.samples)
        val_samples = list(val_base.samples)
    else:
        raise ValueError("DATASET must be imagenet or tiny_imagenet")

    num_classes = len(classes)

    # Optional subset / per-class caps (env-driven)
    subset_k = env_int("SUBSET_CLASSES", 0)
    subset_seed = env_int("SUBSET_SEED", data_seed)

    keep_classes = (
        select_subset_classes(num_classes, subset_k, subset_seed, device=device)
        if subset_k > 0
        else list(range(num_classes))
    )

    train_cap = env_int("TRAIN_SAMPLES_PER_CLASS", 0)
    val_cap = env_int("VAL_SAMPLES_PER_CLASS", 0)
    cap_seed = env_int("CAP_SEED", subset_seed)

    train_samples_f, _ = filter_samples(train_samples, keep_classes, per_class_cap=train_cap, cap_seed=cap_seed)
    val_samples_f, _ = filter_samples(val_samples, keep_classes, per_class_cap=val_cap, cap_seed=cap_seed)

    classes_f = [classes[i] for i in keep_classes]
    train_dataset = PathLabelDataset(train_samples_f, classes_f, transform=train_tf)
    val_dataset = PathLabelDataset(val_samples_f, classes_f, transform=val_tf)
    num_classes = len(classes_f)

    if is_main_process():
        ddp_print(
            f"[Data] dataset={dataset_name} train={len(train_dataset)} "
            f"val={len(val_dataset)} classes={num_classes}"
        )

    # Dataloaders
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_dist() else None
    val_sampler = None  # keep validation non-sharded by default

    worker_seed_base = data_seed + (get_rank() * 10000) if is_dist() else data_seed
    train_worker_init = lambda wid: worker_init_fn(wid, worker_seed_base)
    val_worker_init = lambda wid: worker_init_fn(wid, worker_seed_base + 5000)

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_gpu_batch,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=train_worker_init if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=per_gpu_batch,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        worker_init_fn=val_worker_init if num_workers > 0 else None,
    )

    info: Dict[str, object] = dict(
        dataset_name=dataset_name,
        data_root=data_root,
        img_size=img_size,
        per_gpu_batch=per_gpu_batch,
        num_workers=num_workers,
        world_size=get_world_size(),
    )
    return train_loader, val_loader, train_sampler, num_classes, info

