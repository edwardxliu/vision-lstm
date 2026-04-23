import json
import os


def save_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def append_jsonl(path: str, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def plot_metrics(metrics_jsonl: str, out_dir: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        # plotting is optional; failing silently (with a log) is acceptable
        print(f"[Plot] matplotlib not available, skip plots: {e}")
        return

    rows = []
    with open(metrics_jsonl, "r") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    if not rows:
        return

    def _plot(xkey, ykey, title, fname):
        pairs = [
            (r.get(xkey), r.get(ykey))
            for r in rows
            if r.get(xkey) is not None and r.get(ykey) is not None
        ]
        if not pairs:
            return
        xs, ys = zip(*pairs)
        plt.figure()
        plt.plot(xs, ys)
        plt.xlabel(xkey)
        plt.ylabel(ykey)
        plt.title(title)
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=160)
        plt.close()

    _plot("epoch", "val_acc", "Val Acc vs Epoch", "curve_val_acc_epoch.png")
    _plot("global_step", "val_acc", "Val Acc vs Step", "curve_val_acc_step.png")
    _plot("elapsed_sec", "val_acc", "Val Acc vs Time (s)", "curve_val_acc_time.png")
    _plot("epoch", "train_loss", "Train Loss vs Epoch", "curve_train_loss_epoch.png")
    _plot("epoch", "val_loss", "Val Loss vs Epoch", "curve_val_loss_epoch.png")

