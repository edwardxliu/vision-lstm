#!/usr/bin/env bash
set -euo pipefail

# Tiny-ImageNet, no-regularization training (VIL + ViT ablations).
# Shared defaults are defined in configs/tiny_noreg_vil.yaml and configs/tiny_reg_vit.yaml
# (for ViT we reuse the regularization config but override LABEL_SMOOTH/MIXUP_* to 0 here).

NPROC="${NPROC:-8}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[train_tiny_noreg] NPROC=${NPROC}, PYTHON_BIN=${PYTHON_BIN}"

run_experiment_vil() {
  local ablation="$1"
  local dwt_fuse="$2"
  local run_tag="$3"
  local extra_env="${4:-}"

  export MODEL_KIND="vil"
  export CONFIG="configs/tiny_noreg_vil.yaml"
  export ABLATION="${ablation}"
  export DWT_FUSE="${dwt_fuse}"
  export RUN_TAG="${run_tag}"

  if [[ -n "${extra_env}" ]]; then
    eval "${extra_env}"
  fi

  echo "[train_tiny_noreg][VIL] ABLATION=${ABLATION} DWT_FUSE=${DWT_FUSE} RUN_TAG=${RUN_TAG}"
  ${PYTHON_BIN} -m torch.distributed.run --nproc_per_node="${NPROC}" train_ablation_ddp.py
}

run_experiment_vit() {
  local ablation="$1"
  local dwt_fuse="$2"
  local run_tag="$3"
  local extra_env="${4:-}"

  export MODEL_KIND="vit_tiny"
  export CONFIG="configs/tiny_reg_vit.yaml"
  # override regularization to "no reg"
  export LABEL_SMOOTH=0.0
  export MIXUP_PROB=0.0
  export CUTMIX_ALPHA=0.0
  export MIXUP_ALPHA=0.0
  export SWITCH_PROB=0.0

  export ABLATION="${ablation}"
  export DWT_FUSE="${dwt_fuse}"
  export RUN_TAG="${run_tag}"

  if [[ -n "${extra_env}" ]]; then
    eval "${extra_env}"
  fi

  echo "[train_tiny_noreg][ViT] ABLATION=${ABLATION} DWT_FUSE=${DWT_FUSE} RUN_TAG=${RUN_TAG}"
  ${PYTHON_BIN} -m torch.distributed.run --nproc_per_node="${NPROC}" train_ablation_ddp.py
}

######## VIL ########

run_experiment_vil "A1" "none" "tiny_vil_A1_ch32_patch8_noreg"
run_experiment_vil "W3_POOL_ONLY" "none" "tiny_vil_W3_poolonly_ch32_patch8_noreg"
run_experiment_vil "W3" "add" "tiny_vil_W3_add_ch32_patch8_noreg"

run_experiment_vil "W3_IMPROVED_WARMUP" "add" "tiny_vil_W3_improved_warmup_ch32_patch8_noreg" \
  'export WAVELET_WARMUP_STEPS=10000; export WAVELET_SCALE_INIT=0.1; export WAVELET_FUSE_MODE=add'

run_experiment_vil "W3_IMPROVED_WARMUP" "add" "tiny_vil_W3_improved_warmup_ch32_patch8_noreg_fuse_multiply" \
  'export WAVELET_WARMUP_STEPS=10000; export WAVELET_SCALE_INIT=0.1; export WAVELET_FUSE_MODE=multiply'

run_experiment_vil "W3_TOKENONLY" "add" "tiny_vil_W3_tokenonly_ch32_patch8_noreg" \
  'unset WAVELET_WARMUP_STEPS WAVELET_FUSE_MODE WAVELET_SCALE_INIT'

run_experiment_vil "W3_RESIDUALONLY" "none" "tiny_vil_W3_residualonly_ch32_patch8_noreg" \
  'unset WAVELET_WARMUP_STEPS WAVELET_SCALE_INIT'

######## ViT-Tiny ########

run_experiment_vit "A3" "add" "tiny_vit_A3_ch32_patch8_noreg"
run_experiment_vit "W3_POOL_ONLY" "none" "tiny_vit_W3_poolonly_ch32_patch8_noreg"
run_experiment_vit "W3" "add" "tiny_vit_W3_add_ch32_patch8_noreg"

run_experiment_vit "W3_IMPROVED_WARMUP" "add" "tiny_vit_W3_improved_warmup_ch32_patch8_noreg" \
  'export WAVELET_WARMUP_STEPS=10000; export WAVELET_SCALE_INIT=0.1; export WAVELET_FUSE_MODE=add'

run_experiment_vit "W3_IMPROVED_WARMUP" "add" "tiny_vit_W3_improved_warmup_ch32_patch8_noreg_fuse_multiply" \
  'export WAVELET_WARMUP_STEPS=10000; export WAVELET_SCALE_INIT=0.1; export WAVELET_FUSE_MODE=multiply'

run_experiment_vit "W3_TOKENONLY" "add" "tiny_vit_W3_tokenonly_ch32_patch8_noreg" \
  'unset WAVELET_WARMUP_STEPS WAVELET_FUSE_MODE WAVELET_SCALE_INIT'

run_experiment_vit "W3_RESIDUAL" "add" "tiny_vit_W3_residual_ch32_patch8_noreg"

