#!/usr/bin/env bash
set -euo pipefail

# ImageNet-C evaluation for ViL and ViT-Tiny.
# Assumes training outputs (with ema_best.pth) already exist under OUT_DIR.
# Default checkpoint directory names follow the current ImageNet-1K run tags:
#   in1k192_vil_A1_ch32_reg
#   in1k192_vil_W3_poolonly_ch32_reg
#   in1k192_vil_W3_tokenonly_ch32_reg
#   in1k192_vil_W3_residualonly_ch32_reg
#   in1k192_vil_W3_improved_warmup_ch32_reg_fuse_add
#   in1k192_vit_A3_ch32
#   in1k192_vit_W3_poolonly_ch32
#   in1k192_vit_W3_tokenonly_ch32
#   in1k192_vit_W3_residual_ch32
#   in1k192_vit_W3_improved_warmup_ch32_fuse_add
#
# Usage examples:
#   bash eval_in1kc.sh                     # run all listed evals
#   ONLY=vil bash eval_in1kc.sh            # run only ViL
#   ONLY=vit bash eval_in1kc.sh            # run only ViT
#   NPROC=8 OUT_DIR=./outputs_psf bash eval_in1kc.sh
#   CONFIG=configs/in1kc_vil_eval.yaml ONLY=vil bash eval_in1kc.sh

NPROC="${NPROC:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-./outputs_psf}"
ONLY="${ONLY:-all}"   # all | vil | vit

# Conservative default for 192x192 eval. Override if your GPU memory allows.
PER_GPU_BATCH="${PER_GPU_BATCH:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"

common_eval_env() {
  export MODE="eval_imagenetc"
  export DATASET="imagenet"
  export IMG_SIZE=192
  export PER_GPU_BATCH="${PER_GPU_BATCH}"
  export NUM_WORKERS="${NUM_WORKERS}"
  export AMP_DTYPE="${AMP_DTYPE}"
  export OUT_DIR="${OUT_DIR}"
}

run_eval() {
  local model_kind="$1"    # vil | vit_tiny
  local ablation="$2"
  local dwt_fuse="$3"
  local tag_prefix="$4"
  local ckpt_rel="$5"
  local extra_env="${6:-}"

  common_eval_env

  # Select default CONFIG per model kind if not already set by caller.
  if [[ "${model_kind}" == "vil" ]]; then
    export CONFIG="${CONFIG:-configs/in1kc_vil_eval.yaml}"
  else
    export CONFIG="${CONFIG:-configs/in1kc_vit_eval.yaml}"
  fi

  export MODEL_KIND="${model_kind}"
  export ABLATION="${ablation}"
  export DWT_FUSE="${dwt_fuse}"
  export RUN_TAG="${tag_prefix}"
  export CKPT="${OUT_DIR}/${ckpt_rel}"

  if [[ -n "${extra_env}" ]]; then
    eval "${extra_env}"
  fi

  echo "[eval_in1kc] MODEL_KIND=${MODEL_KIND} ABLATION=${ABLATION} CKPT=${CKPT} CONFIG=${CONFIG}"
  "${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node="${NPROC}" train_ablation_ddp.py
}

if [[ "${ONLY}" == "all" || "${ONLY}" == "vil" ]]; then
  ######## ViL ########
  run_eval "vil" "A1" "none" \
    "eval_in1kc_vil_A1_ch32_reg" \
    "in1k192_vil_A1_ch32_reg/ema_best.pth"

  run_eval "vil" "W3_POOL_ONLY" "none" \
    "eval_in1kc_vil_W3_poolonly_ch32_reg" \
    "in1k192_vil_W3_poolonly_ch32_reg/ema_best.pth"

  run_eval "vil" "W3_TOKENONLY" "add" \
    "eval_in1kc_vil_W3_tokenonly_ch32_reg" \
    "in1k192_vil_W3_tokenonly_ch32_reg/ema_best.pth" \
    'unset WAVELET_WARMUP_STEPS WAVELET_FUSE_MODE'

  run_eval "vil" "W3_RESIDUALONLY" "none" \
    "eval_in1kc_vil_W3_residualonly_ch32_reg" \
    "in1k192_vil_W3_residualonly_ch32_reg/ema_best.pth"

  run_eval "vil" "W3_IMPROVED_WARMUP" "add" \
    "eval_in1kc_vil_W3_improved_warmup_ch32_reg_fuse_add" \
    "in1k192_vil_W3_improved_warmup_ch32_reg_fuse_add/ema_best.pth" \
    'export WAVELET_FUSE_MODE=add'
fi

if [[ "${ONLY}" == "all" || "${ONLY}" == "vit" ]]; then
  ######## ViT-Tiny ########
  run_eval "vit_tiny" "A3" "add" \
    "eval_in1kc_vit_A3_ch32" \
    "in1k192_vit_A3_ch32/ema_best.pth"

  run_eval "vit_tiny" "W3_POOL_ONLY" "none" \
    "eval_in1kc_vit_W3_poolonly_ch32" \
    "in1k192_vit_W3_poolonly_ch32/ema_best.pth"

  run_eval "vit_tiny" "W3_TOKENONLY" "add" \
    "eval_in1kc_vit_W3_tokenonly_ch32" \
    "in1k192_vit_W3_tokenonly_ch32/ema_best.pth" \
    'unset WAVELET_FUSE_MODE'

  run_eval "vit_tiny" "W3_RESIDUAL" "add" \
    "eval_in1kc_vit_W3_residual_ch32" \
    "in1k192_vit_W3_residual_ch32/ema_best.pth"

  run_eval "vit_tiny" "W3_IMPROVED_WARMUP" "add" \
    "eval_in1kc_vit_W3_improved_warmup_ch32_fuse_add" \
    "in1k192_vit_W3_improved_warmup_ch32_fuse_add/ema_best.pth" \
    'export WAVELET_FUSE_MODE=add'
fi
