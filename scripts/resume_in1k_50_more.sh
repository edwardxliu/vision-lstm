#!/usr/bin/env bash
set -euo pipefail

# Resume ImageNet-1K training from a previous checkpoint for "50 more" epochs (or user-specified).
# This script is a thin wrapper around train_ablation_ddp.py:
#   - uses configs/in1k_vil_50ep.yaml as the default VIL configuration
#   - expects RESUME_CKPT to point to an existing checkpoint (ema_best.pth or similar)

NPROC="${NPROC:-8}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "${RESUME_CKPT:-}" ]]; then
  echo "[resume_in1k_50_more] Please set RESUME_CKPT=/path/to/ema_best.pth" >&2
  exit 1
fi

MODEL_KIND="${MODEL_KIND:-vil}"   # default to VIL; set MODEL_KIND=vit_tiny if needed
EXTRA_EPOCHS="${EXTRA_EPOCHS:-50}"

echo "[resume_in1k_50_more] NPROC=${NPROC}, MODEL_KIND=${MODEL_KIND}, EXTRA_EPOCHS=${EXTRA_EPOCHS}"
echo "[resume_in1k_50_more] RESUME_CKPT=${RESUME_CKPT}"

if [[ "${MODEL_KIND}" == "vil" ]]; then
  export CONFIG="configs/in1k_vil_50ep.yaml"
else
  export CONFIG="configs/in1k_vit_50ep.yaml"
fi

export MODEL_KIND
export RESUME_CKPT

# You can override EPOCHS via EXTRA_EPOCHS (interpreted as total epochs or "run for EXTRA_EPOCHS more"
# depending on how you set it; here we simply set EPOCHS to EXTRA_EPOCHS).
export EPOCHS="${EXTRA_EPOCHS}"

# RUN_TAG is free-form; default uses a simple suffix.
export RUN_TAG="${RUN_TAG:-resume_${MODEL_KIND}_in1k_${EXTRA_EPOCHS}ep}"

${PYTHON_BIN} -m torch.distributed.run --nproc_per_node="${NPROC}" train_ablation_ddp.py

