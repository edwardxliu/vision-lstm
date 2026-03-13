from typing import Tuple, Dict, Optional, List

import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.amp import autocast

from data_loader import TinyImageNetCDataset


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp_dtype) -> Tuple[float, float]:
    model.eval()
    total, correct = 0, 0
    loss_sum = 0.0
    with torch.inference_mode(), autocast(device_type=device.type, dtype=amp_dtype):
        for imgs, target in loader:
            imgs = imgs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = model(imgs)
            loss = F.cross_entropy(logits, target)
            loss_sum += loss.item() * target.size(0)
            pred = logits.argmax(1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return loss_sum / max(1, total), correct / max(1, total)


@torch.no_grad()
def evaluate_imagenet_c(
    model: nn.Module,
    imagenetc_root: str,
    img_size: int,
    device: torch.device,
    amp_dtype,
    batch_size: int,
    num_workers: int,
    dataset_name: str = "imagenet",
    data_root: str = "",
    corruptions: Optional[List[str]] = None,
    severities: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Evaluate ImageNet-C folder layout:
      IMAGENETC_ROOT/<corruption>/<severity>/<class>/*.JPEG
    For tiny_imagenet, labels are mapped using wnids.txt in DATA_ROOT.
    """
    # 1) auto-detect corruptions if not provided
    if corruptions is None:
        corruptions = sorted(
            [
                d
                for d in os.listdir(imagenetc_root)
                if os.path.isdir(os.path.join(imagenetc_root, d)) and not d.startswith(".")
            ]
        )

    # 2) default severities
    if severities is None:
        severities = [1, 2, 3, 4, 5]

    resize_size = 224 if img_size >= 128 else img_size
    val_tf = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    wnids_path = ""
    if dataset_name == "tiny_imagenet":
        wnids_path = os.path.join(data_root, "wnids.txt")
        if not os.path.isfile(wnids_path):
            raise RuntimeError(f"tiny_imagenet requires wnids.txt at: {wnids_path}")

    results: Dict[str, float] = {}
    all_acc = []

    for c in corruptions:
        accs = []
        for s in severities:
            d = os.path.join(imagenetc_root, c, str(s))
            if not os.path.isdir(d):
                continue

            if dataset_name == "tiny_imagenet":
                ds = TinyImageNetCDataset(severity_root=d, wnids_path=wnids_path, transform=val_tf)
            else:
                ds = datasets.ImageFolder(root=d, transform=val_tf)

            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
            )
            _, acc = evaluate(model, loader, device, amp_dtype)
            accs.append(acc)

        if accs:
            m = float(sum(accs) / len(accs))
            results[c] = m
            all_acc.append(m)

    results["mean"] = float(sum(all_acc) / max(1, len(all_acc)))
    return results

