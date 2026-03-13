#!/usr/bin/env bash
set -euo pipefail

# Tiny-ImageNet, regularized training (VIL + ViT ablations).
# Shared defaults are defined in configs/tiny_reg_vil.yaml and configs/tiny_reg_vit.yaml.

NPROC="${NPROC:-8}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[train_tiny_reg] NPROC=${NPROC}, PYTHON_BIN=${PYTHON_BIN}"

run_experiment() {
  local model_kind="$1"   # vil | vit_tiny
  local ablation="$2"
  local dwt_fuse="$3"
  local run_tag="$4"
  local extra_env="${5:-}"

  export MODEL_KIND="${model_kind}"
  if [[ "${model_kind}" == "vil" ]]; then
    export CONFIG="configs/tiny_reg_vil.yaml"
  else
    export CONFIG="configs/tiny_reg_vit.yaml"
  fi

  export ABLATION="${ablation}"
  export DWT_FUSE="${dwt_fuse}"
  export RUN_TAG="${run_tag}"

  # Allow caller to pass additional environment overrides as a string.
  if [[ -n "${extra_env}" ]]; then
    eval "${extra_env}"
  fi

  echo "[train_tiny_reg] MODEL_KIND=${MODEL_KIND} ABLATION=${ABLATION} DWT_FUSE=${DWT_FUSE} RUN_TAG=${RUN_TAG}"
  ${PYTHON_BIN} -m torch.distributed.run --nproc_per_node="${NPROC}" train_ablation_ddp.py
}

######## VIL ########

# A1 baseline (no PSWF)
run_experiment "vil" "A1" "none" "tiny_vil_A1_ch32_patch8_reg"

# W3_POOL_ONLY
run_experiment "vil" "W3_POOL_ONLY" "none" "tiny_vil_W3_poolonly_ch32_patch8_reg"

# W3 (PSWF add, scale=0)
run_experiment "vil" "W3" "add" "tiny_vil_W3_add_ch32_patch8_reg"

# W3_IMPROVED_WARMUP (add vs multiply)
run_experiment "vil" "W3_IMPROVED_WARMUP" "add" "tiny_vil_W3_improved_warmup_ch32_patch8_reg" \
  'export WAVELET_SCALE_INIT=0.1; export WAVELET_WARMUP_STEPS=10000; export WAVELET_FUSE_MODE=add'

run_experiment "vil" "W3_IMPROVED_WARMUP" "add" "tiny_vil_W3_improved_warmup_ch32_patch8_reg_fuse_multiply" \
  'export WAVELET_SCALE_INIT=0.1; export WAVELET_WARMUP_STEPS=10000; export WAVELET_FUSE_MODE=multiply'

# W3_TOKENONLY
run_experiment "vil" "W3_TOKENONLY" "add" "tiny_vil_W3_tokenonly_ch32_patch8_reg" \
  'unset WAVELET_WARMUP_STEPS WAVELET_FUSE_MODE WAVELET_SCALE_INIT'

# W3_RESIDUALONLY
run_experiment "vil" "W3_RESIDUALONLY" "none" "tiny_vil_W3_residualonly_ch32_patch8_reg" \
  'unset WAVELET_WARMUP_STEPS WAVELET_SCALE_INIT'

######## ViT-Tiny ########

# ViT baseline A3
run_experiment "vit_tiny" "A3" "add" "tiny_vit_A3_ch32_patch8_reg"

# ViT + Pool Only
run_experiment "vit_tiny" "W3_POOL_ONLY" "none" "tiny_vit_W3_poolonly_ch32_patch8_reg"

# ViT + W3 (PSWF add)
run_experiment "vit_tiny" "W3" "add" "tiny_vit_W3_add_ch32_patch8_reg"

# ViT + W3_IMPROVED_WARMUP (add vs multiply)
run_experiment "vit_tiny" "W3_IMPROVED_WARMUP" "add" "tiny_vit_W3_improved_warmup_ch32_patch8_reg" \
  'export WAVELET_SCALE_INIT=0.1; export WAVELET_WARMUP_STEPS=10000; export WAVELET_FUSE_MODE=add'

run_experiment "vit_tiny" "W3_IMPROVED_WARMUP" "add" "tiny_vit_W3_improved_warmup_ch32_patch8_reg_fuse_multiply" \
  'export WAVELET_SCALE_INIT=0.1; export WAVELET_WARMUP_STEPS=10000; export WAVELET_FUSE_MODE=multiply'

# ViT + W3_TOKENONLY
run_experiment "vit_tiny" "W3_TOKENONLY" "add" "tiny_vit_W3_tokenonly_ch32_patch8_reg" \
  'unset WAVELET_WARMUP_STEPS WAVELET_FUSE_MODE WAVELET_SCALE_INIT'

# ViT + W3_RESIDUAL
run_experiment "vit_tiny" "W3_RESIDUAL" "add" "tiny_vit_W3_residual_ch32_patch8_reg"

