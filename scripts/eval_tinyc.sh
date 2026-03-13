#!/usr/bin/env bash
set -euo pipefail

# Tiny-ImageNet-C evaluation for VIL and ViT-Tiny.
# Assumes that training outputs (with ema_best.pth) already exist under OUT_DIR.

NPROC="${NPROC:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

OUT_DIR="${OUT_DIR:-./outputs_psf}"

echo "[eval_tinyc] NPROC=${NPROC}, OUT_DIR=${OUT_DIR}"

common_eval_env() {
  export MODE="eval_imagenetc"
  export DATASET="tiny_imagenet"
  export IMG_SIZE=64
  export PER_GPU_BATCH=256
  export NUM_WORKERS=8
  export AMP_DTYPE=bf16
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
    export CONFIG="${CONFIG:-configs/tinyc_vil_eval.yaml}"
  else
    export CONFIG="${CONFIG:-configs/tinyc_vit_eval.yaml}"
  fi
  export MODEL_KIND="${model_kind}"
  export ABLATION="${ablation}"
  export DWT_FUSE="${dwt_fuse}"
  export RUN_TAG="${tag_prefix}"
  export CKPT="${OUT_DIR}/${ckpt_rel}"

  if [[ -n "${extra_env}" ]]; then
    eval "${extra_env}"
  fi

  echo "[eval_tinyc] MODEL_KIND=${MODEL_KIND} ABLATION=${ABLATION} CKPT=${CKPT}"
  ${PYTHON_BIN} -m torch.distributed.run --nproc_per_node="${NPROC}" train_ablation_ddp.py
}

######## VIL ########

run_eval "vil" "A1" "none" \
  "eval_tinyc_vil_A1_ch32_patch8_reg" \
  "tiny_vil_A1_ch32_patch8_reg/ema_best.pth"

run_eval "vil" "W3" "add" \
  "eval_tinyc_vil_W3_add_ch32_patch8_reg" \
  "tiny_vil_W3_add_ch32_patch8_reg/ema_best.pth"

run_eval "vil" "W3_POOL_ONLY" "none" \
  "eval_tinyc_vil_W3_poolonly_ch32_patch8_reg" \
  "tiny_vil_W3_poolonly_ch32_patch8_reg/ema_best.pth"

run_eval "vil" "W3_IMPROVED_WARMUP" "add" \
  "eval_tinyc_vil_W3_improved_warmup_ch32_patch8_reg" \
  "tiny_vil_W3_improved_warmup_ch32_patch8_reg/ema_best.pth" \
  'export WAVELET_FUSE_MODE=add'

run_eval "vil" "W3_IMPROVED_WARMUP" "add" \
  "eval_tinyc_vil_W3_improved_warmup_ch32_patch8_reg_fuse_multiply" \
  "tiny_vil_W3_improved_warmup_ch32_patch8_reg_fuse_multiply/ema_best.pth" \
  'export WAVELET_FUSE_MODE=multiply'

run_eval "vil" "W3_TOKENONLY" "add" \
  "eval_tinyc_vil_W3_tokenonly_ch32_patch8_reg" \
  "tiny_vil_W3_tokenonly_ch32_patch8_reg/ema_best.pth" \
  'unset WAVELET_WARMUP_STEPS WAVELET_FUSE_MODE'

run_eval "vil" "W3_RESIDUALONLY" "none" \
  "eval_tinyc_vil_W3_residualonly_ch32_patch8_reg" \
  "tiny_vil_W3_residualonly_ch32_patch8_reg/ema_best.pth"

######## ViT-Tiny ########

run_eval "vit_tiny" "A3" "add" \
  "eval_tinyc_vit_A3_ch32_patch8_reg" \
  "tiny_vit_A3_ch32_patch8_reg/ema_best.pth"

run_eval "vit_tiny" "W3" "add" \
  "eval_tinyc_vit_W3_add_ch32_patch8_reg" \
  "tiny_vit_W3_add_ch32_patch8_reg/ema_best.pth"

run_eval "vit_tiny" "W3_IMPROVED_WARMUP" "add" \
  "eval_tinyc_vit_W3_improved_warmup_ch32_patch8_reg" \
  "tiny_vit_W3_improved_warmup_ch32_patch8_reg/ema_best.pth" \
  'export WAVELET_FUSE_MODE=add'

run_eval "vit_tiny" "W3_IMPROVED_WARMUP" "add" \
  "eval_tinyc_vit_W3_improved_warmup_ch32_patch8_reg_fuse_multiply" \
  "tiny_vit_W3_improved_warmup_ch32_patch8_reg_fuse_multiply/ema_best.pth" \
  'export WAVELET_FUSE_MODE=multiply'

run_eval "vit_tiny" "W3_TOKENONLY" "add" \
  "eval_tinyc_vit_W3_tokenonly_ch32_patch8_reg" \
  "tiny_vit_W3_tokenonly_ch32_patch8_reg/ema_best.pth" \
  'unset WAVELET_FUSE_MODE'

run_eval "vit_tiny" "W3_POOL_ONLY" "none" \
  "eval_tinyc_vit_W3_poolonly_ch32_patch8_reg" \
  "tiny_vit_W3_poolonly_ch32_patch8_reg/ema_best.pth"

run_eval "vit_tiny" "W3_RESIDUAL" "add" \
  "eval_tinyc_vit_W3_residual_ch32_patch8_reg" \
  "tiny_vit_W3_residual_ch32_patch8_reg/ema_best.pth"

