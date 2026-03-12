#!/usr/bin/env python3
"""
Appendix plotting helpers for PSF paper.

Examples
--------

1) Training curves with threshold + inset:
python plot_appendix_figs.py curves \
  --x time \
  --title "ImageNet-1K@192 / ViT / Validation Top-1 vs Time" \
  --ylabel "Validation Top-1 (%)" \
  --out app_in1k_vit_curve.pdf \
  --threshold 58.26 \
  --annotate-threshold \
  --inset \
  --inset-xmin 2.4 \
  --inset-xmax 4.6 \
  --inset-ymin 56.0 \
  --inset-ymax 59.5 \
  --series "ViT-Base=in1k192_vit_A3_ch32_metrics.jsonl" \
  --series "PSF-Pool=in1k192_vit_W3_poolonly_ch32_metrics.jsonl" \
  --series "PSF-Both=in1k192_vit_W3_add_ch32_metrics.jsonl" \
  --series "PSF-HeadWarmup-Mul=in1k192_vit_W3_improved_warmup_ch32_fuse_multiply_metrics.jsonl" \
  --series "PSF-HeadMod=in1k192_vit_W3_residual_ch32_metrics.jsonl"

2) Training curves by epoch:
python plot_appendix_figs.py curves \
  --x epoch \
  --title "Tiny-ImageNet / ViT / Validation Top-1 vs Epoch" \
  --out app_tiny_vit_reg_curve.pdf \
  --series "ViT-Base=tiny_vit_A3_ch32_patch8_reg_metrics.jsonl" \
  --series "PSF-Pool=tiny_vit_W3_poolonly_ch32_patch8_reg_metrics.jsonl"

3) Grouped robustness bars:
python plot_appendix_figs.py bars \
  --title "Tiny-ImageNet-C / ViT / Grouped Robustness" \
  --out app_tinyc_vit_grouped.pdf \
  --series "ViT-Base=eval_tinyc_vit_A3_ch32_patch8_reg.json" \
  --series "PSF-Pool=eval_tinyc_vit_W3_poolonly_ch32_patch8_reg.json" \
  --series "PSF-TokenWavelet=eval_tinyc_vit_W3_tokenonly_ch32_patch8_reg.json" \
  --series "PSF-HeadWarmup-Add=eval_tinyc_vit_W3_improved_warmup_ch32_patch8_reg.json"
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_first(obj: Any, candidates: List[str]) -> Optional[Any]:
    if isinstance(obj, dict):
        for key in candidates:
            if key in obj:
                return obj[key]
        for v in obj.values():
            found = find_first(v, candidates)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_first(item, candidates)
            if found is not None:
                return found
    return None


def normalize_percent(v: float) -> float:
    return v * 100.0 if v <= 1.5 else v


VAL_CANDIDATES = [
    "val_acc1", "acc1", "top1", "val_top1", "val/acc1", "val_acc",
    "eval_acc1", "best_acc1", "validation_acc1", "imagenet_c_top1"
]
TIME_CANDIDATES = ["elapsed_sec", "time_sec", "elapsed_seconds", "wall_time_sec", "wall_sec"]
EPOCH_CANDIDATES = ["epoch", "ep", "step_epoch"]

GROUP_KEYS = {
    "Mean": ["mean", "overall_mean", "all_mean", "avg", "top1_mean"],
    "Noise": ["noise", "noise_mean"],
    "Blur": ["blur", "blur_mean"],
    "Digital": ["digital", "digital_mean"],
    "Weather": ["weather", "weather_mean"],
}

# Fallback grouping when only per-corruption metrics are present
CORRUPTION_GROUPS = {
    "Noise": [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "speckle_noise",
    ],
    "Blur": [
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "gaussian_blur",
    ],
    "Digital": [
        "contrast",
        "elastic_transform",
        "jpeg_compression",
        "pixelate",
        "saturate",
    ],
    "Weather": [
        "brightness",
        "fog",
        "frost",
        "snow",
        "spatter",
    ],
}


def extract_curve(path: Path, x_mode: str) -> Tuple[List[float], List[float]]:
    rows = load_jsonl(path)
    xs, ys = [], []
    for row in rows:
        y = find_first(row, VAL_CANDIDATES)
        x = find_first(row, TIME_CANDIDATES if x_mode == "time" else EPOCH_CANDIDATES)
        if y is None or x is None:
            continue
        y = normalize_percent(float(y))
        x = float(x)
        if x_mode == "time":
            x = x / 3600.0
        xs.append(x)
        ys.append(y)

    if not xs:
        raise ValueError(
            f"Could not find curve data in {path}. "
            f"Tried y keys={VAL_CANDIDATES} and "
            f"x keys={TIME_CANDIDATES if x_mode == 'time' else EPOCH_CANDIDATES}."
        )
    return xs, ys


def first_crossing(xs: List[float], ys: List[float], threshold: float) -> Optional[Tuple[float, float]]:
    for x, y in zip(xs, ys):
        if y >= threshold:
            return x, y
    return None


def extract_group_metrics(path: Path) -> Dict[str, float]:
    obj = load_json(path)
    out = {}

    # 1) Mean: prefer explicit mean if present
    mean_val = find_first(obj, GROUP_KEYS["Mean"])
    if mean_val is not None:
        out["Mean"] = normalize_percent(float(mean_val))
    else:
        # fallback: average over all known corruption entries
        vals = []
        for names in CORRUPTION_GROUPS.values():
            for name in names:
                v = find_first(obj, [name])
                if v is not None:
                    vals.append(normalize_percent(float(v)))
        if not vals:
            raise ValueError(f"Could not find any corruption metrics in {path}.")
        out["Mean"] = sum(vals) / len(vals)

    # 2) Grouped means: prefer explicit grouped keys, otherwise aggregate manually
    for display_name in ["Noise", "Blur", "Digital", "Weather"]:
        v = find_first(obj, GROUP_KEYS[display_name])
        if v is not None:
            out[display_name] = normalize_percent(float(v))
            continue

        vals = []
        for cname in CORRUPTION_GROUPS[display_name]:
            cv = find_first(obj, [cname])
            if cv is not None:
                vals.append(normalize_percent(float(cv)))

        if not vals:
            raise ValueError(
                f"Could not find grouped metric '{display_name}' in {path}, "
                f"and none of the fallback corruption keys were present: {CORRUPTION_GROUPS[display_name]}"
            )

        out[display_name] = sum(vals) / len(vals)

    return out


def plot_curves(
    series,
    x_mode,
    title,
    ylabel,
    out,
    threshold=None,
    annotate_threshold=False,
    inset=False,
    inset_xmin=None,
    inset_xmax=None,
    inset_ymin=None,
    inset_ymax=None,
    ymin=None,
    ymax=None,
):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    curve_data = {}
    line_styles = {}

    for label, path in series:
        xs, ys = extract_curve(path, x_mode=x_mode)
        curve_data[label] = (xs, ys)
        line, = ax.plot(xs, ys, linewidth=2, label=label)
        line_styles[label] = {
            "color": line.get_color(),
            "linestyle": line.get_linestyle(),
            "linewidth": 2,
        }

    # 先画主图 threshold 线
    if threshold is not None:
        ax.axhline(
            threshold,
            color="gray",
            linestyle="--",
            linewidth=1.5,
            label=f"T={threshold:.2f}",
        )

    ax.set_xlabel("Time (hours)" if x_mode == "time" else "Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    

    if ymin is not None or ymax is not None:
        ax.set_ylim(
            ymin if ymin is not None else ax.get_ylim()[0],
            ymax if ymax is not None else ax.get_ylim()[1],
        )

    # 先创建 inset，后面 annotation 才能安全使用 axins
    axins = None
    if inset:
        if None in (inset_xmin, inset_xmax, inset_ymin, inset_ymax):
            raise ValueError("Inset requested, but inset_xmin/xmax/ymin/ymax are not fully specified.")

        #axins = inset_axes(ax, width="45%", height="45%", loc="lower right", borderpad=2)
        axins = inset_axes(ax, width="40%", height="40%", loc="center right", borderpad=2)

        for label, (xs, ys) in curve_data.items():
            style = line_styles[label]
            axins.plot(
                xs, ys,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
            )

        if threshold is not None:
            axins.axhline(threshold, color="gray", linestyle="--", linewidth=1.0)

        axins.set_xlim(inset_xmin, inset_xmax)
        axins.set_ylim(inset_ymin, inset_ymax)
        axins.grid(True, alpha=0.3)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # 再做 crossing 标注
    if annotate_threshold and threshold is not None:
        crossings = []
        for label, (xs, ys) in curve_data.items():
            crossing = first_crossing(xs, ys, threshold)
            if crossing is None:
                continue
            cx, cy = crossing
            style = line_styles[label]
            ax.scatter([cx], [cy], color=style["color"], s=28, zorder=5)
            crossings.append((label, cx, cy, style))

        crossings = sorted(crossings, key=lambda t: t[1])

        if axins is not None:
            # 只在 inset 里写文字，主图只保留点
            y_offsets = [16, 8, 0, -8, -16, -24, -32]
            for i, (label, cx, cy, style) in enumerate(crossings):
                if inset_xmin <= cx <= inset_xmax and inset_ymin <= cy <= inset_ymax:
                    axins.scatter([cx], [cy], color=style["color"], s=24, zorder=5)
                    axins.annotate(
                        f"{label}: {cx:.2f}{'h' if x_mode == 'time' else ''}",
                        xy=(cx, cy),
                        xytext=(6, y_offsets[i % len(y_offsets)]),
                        textcoords="offset points",
                        fontsize=8,
                        color=style["color"],
                    )
        else:
            # 没有 inset 时，主图错层标注
            y_offsets = [18, 10, 2, -6, -14, -22, -30]
            for i, (label, cx, cy, style) in enumerate(crossings):
                ax.annotate(
                    f"{label}: {cx:.2f}{'h' if x_mode == 'time' else ''}",
                    xy=(cx, cy),
                    xytext=(8, y_offsets[i % len(y_offsets)]),
                    textcoords="offset points",
                    fontsize=8,
                    color=style["color"],
                )

    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_grouped_bars(series: List[Tuple[str, Path]], title: str, out: Path):
    group_names = list(GROUP_KEYS.keys())
    labels = []
    values = []
    for label, path in series:
        labels.append(label)
        metrics = extract_group_metrics(path)
        values.append([metrics[g] for g in group_names])

    n_groups = len(group_names)
    n_series = len(series)
    width = 0.8 / n_series
    xs = list(range(n_groups))

    plt.figure(figsize=(8.0, 4.8))
    for i, (label, vals) in enumerate(zip(labels, values)):
        offset = (i - (n_series - 1) / 2.0) * width
        xlocs = [x + offset for x in xs]
        plt.bar(xlocs, vals, width=width, label=label)

    plt.xticks(xs, group_names)
    plt.ylabel("Top-1 (%)")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    #plt.legend()
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    plt.tight_layout()
    #plt.savefig(out)
    plt.savefig(out, bbox_inches="tight")
    plt.close()

def plot_timebar(labels: List[str], times: List[float], title: str, ylabel: str, out: Path):
    plt.figure(figsize=(6.8, 4.2))
    plt.bar(labels, times)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def parse_label_value(items: List[str]) -> Tuple[List[str], List[float]]:
    labels, values = [], []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--value must be LABEL=NUMBER, got: {item}")
        label, value = item.split("=", 1)
        labels.append(label)
        values.append(float(value))
    return labels, values

def parse_series(items: List[str]) -> List[Tuple[str, Path]]:
    parsed = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--series must be LABEL=PATH, got: {item}")
        label, path = item.split("=", 1)
        parsed.append((label, Path(path)))
    return parsed


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_curves = sub.add_parser("curves")
    p_curves.add_argument("--x", choices=["time", "epoch"], default="time")
    p_curves.add_argument("--title", required=True)
    p_curves.add_argument("--ylabel", default="Validation Top-1 (%)")
    p_curves.add_argument("--out", required=True)
    p_curves.add_argument("--series", action="append", required=True)

    p_curves.add_argument("--threshold", type=float, default=None)
    p_curves.add_argument("--annotate-threshold", action="store_true")
    p_curves.add_argument("--inset", action="store_true")
    p_curves.add_argument("--inset-xmin", type=float, default=None)
    p_curves.add_argument("--inset-xmax", type=float, default=None)
    p_curves.add_argument("--inset-ymin", type=float, default=None)
    p_curves.add_argument("--inset-ymax", type=float, default=None)
    p_curves.add_argument("--ymin", type=float, default=None)
    p_curves.add_argument("--ymax", type=float, default=None)

    p_bars = sub.add_parser("bars")
    p_bars.add_argument("--title", required=True)
    p_bars.add_argument("--out", required=True)
    p_bars.add_argument("--series", action="append", required=True)

    p_timebar = sub.add_parser("timebar")
    p_timebar.add_argument("--title", required=True)
    p_timebar.add_argument("--ylabel", required=True)
    p_timebar.add_argument("--out", required=True)
    p_timebar.add_argument("--value", action="append", required=True,
                           help="LABEL=NUMBER ; may be repeated")
    

    args = parser.parse_args()

    if args.cmd == "curves":
        plot_curves(
            parse_series(args.series),
            x_mode=args.x,
            title=args.title,
            ylabel=args.ylabel,
            out=Path(args.out),
            threshold=args.threshold,
            annotate_threshold=args.annotate_threshold,
            inset=args.inset,
            inset_xmin=args.inset_xmin,
            inset_xmax=args.inset_xmax,
            inset_ymin=args.inset_ymin,
            inset_ymax=args.inset_ymax,
            ymin=args.ymin,
            ymax=args.ymax,
        )
    elif args.cmd == "bars":
        plot_grouped_bars(parse_series(args.series), args.title, Path(args.out))
    elif args.cmd == "timebar":
        labels, values = parse_label_value(args.value)
        plot_timebar(labels, values, args.title, args.ylabel, Path(args.out))


if __name__ == "__main__":
    main()