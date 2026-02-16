# plot_runs_pswf.py
# Overlay curves from multiple runs (metrics.jsonl) produced by lstm5_stage1_pretrain_192_sample_ablation_paper.py
#
# Usage:
#   python plot_runs_pswf.py --out compare.png run1/metrics.jsonl run2/metrics.jsonl ...
#
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_metrics(p):
    rows=[]
    with open(p,"r") as f:
        for ln in f:
            ln=ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows

def label_from_path(p):
    # prefer run directory name
    return os.path.basename(os.path.dirname(os.path.abspath(p)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--x", default="epoch", choices=["epoch","global_step","elapsed_sec"])
    ap.add_argument("--y", default="val_acc", choices=["val_acc","train_loss","val_loss","train_soft_acc"])
    ap.add_argument("runs", nargs="+", help="paths to metrics.jsonl")
    args=ap.parse_args()

    plt.figure()
    for p in args.runs:
        rows=load_metrics(p)
        xs=[r.get(args.x) for r in rows if args.x in r and args.y in r]
        ys=[r.get(args.y) for r in rows if args.x in r and args.y in r]
        if xs and ys:
            plt.plot(xs, ys, label=label_from_path(p))
    plt.xlabel(args.x)
    plt.ylabel(args.y)
    plt.title(f"{args.y} vs {args.x}")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()

if __name__=="__main__":
    main()
