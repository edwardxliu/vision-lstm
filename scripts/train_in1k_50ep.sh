#!/usr/bin/env bash
set -euo pipefail

# ImageNet-1K, 50-epoch training for VIL and ViT-Tiny.
# Shared defaults are defined in configs/in1k_vil_50ep.yaml and configs/in1k_vit_50ep.yaml.

NPROC="${NPROC:-8}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[train_in1k_50ep] NPROC=${NPROC}, PYTHON_BIN=${PYTHON_BIN}"

run_experiment_vil() {
  local ablation="$1"
  local dwt_fuse="$2"
  local run_tag="$3"
  local extra_env="${4:-}"

  export MODEL_KIND="vil"
  export CONFIG="configs/in1k_vil_50ep.yaml"
  export ABLATION="${ablation}"
  export DWT_FUSE="${dwt_fuse}"
  export RUN_TAG="${run_tag}"

  if [[ -n "${extra_env}" ]]; then
    eval "${extra_env}"
  fi

  echo "[train_in1k_50ep][VIL] ABLATION=${ABLATION} DWT_FUSE=${DWT_FUSE} RUN_TAG=${RUN_TAG}"
  ${PYTHON_BIN} -m torch.distributed.run --nproc_per_node="${NPROC}" train_ablation_ddp.py
}

run_experiment_vit() {
  local ablation="$1"
  local dwt_fuse="$2"
  local run_tag="$3"
  local extra_env="${4:-}"

  export MODEL_KIND="vit_tiny"
  export CONFIG="configs/in1k_vit_50ep.yaml"
  export ABLATION="${ablation}"
  export DWT_FUSE="${dwt_fuse}"
  export RUN_TAG="${run_tag}"

  if [[ -n "${extra_env}" ]]; then
    eval "${extra_env}"
  fi

  echo "[train_in1k_50ep][ViT] ABLATION=${ABLATION} DWT_FUSE=${DWT_FUSE} RUN_TAG=${RUN_TAG}"
  ${PYTHON_BIN} -m torch.distributed.run --nproc_per_node="${NPROC}" train_ablation_ddp.py
}

######## VIL ########

run_experiment_vil "A1" "none" "in1k192_vil_A1_ch32_reg"
run_experiment_vil "W3_POOL_ONLY" "none" "in1k192_vil_W3_poolonly_ch32_reg" 'unset WAVELET_WARMUP_STEPS'
run_experiment_vil "W3" "add" "in1k192_vil_W3_add_ch32_reg" 'unset WAVELET_WARMUP_STEPS WAVELET_SCALE_INIT'

run_experiment_vil "W3_IMPROVED_WARMUP" "add" "in1k192_vil_W3_improved_warmup_ch32_reg_fuse_add" \
  'export WAVELET_WARMUP_STEPS=40000; export WAVELET_SCALE_INIT=0.1; export WAVELET_FUSE_MODE=add'

run_experiment_vil "W3_IMPROVED_WARMUP" "add" "in1k192_vil_W3_improved_warmup_ch32_reg_fuse_multiply" \
  'export WAVELET_WARMUP_STEPS=40000; export WAVELET_SCALE_INIT=0.1; export WAVELET_FUSE_MODE=multiply'

run_experiment_vil "W3_TOKENONLY" "add" "in1k192_vil_W3_tokenonly_ch32_reg" \
  'unset WAVELET_WARMUP_STEPS WAVELET_FUSE_MODE WAVELET_SCALE_INIT'

run_experiment_vil "W3_RESIDUALONLY" "none" "in1k192_vil_W3_residualonly_ch32_reg" \
  'unset WAVELET_WARMUP_STEPS WAVELET_SCALE_INIT'

######## ViT-Tiny ########

run_experiment_vit "A3" "add" "in1k192_vit_A3_ch32"
run_experiment_vit "W3_POOL_ONLY" "none" "in1k192_vit_W3_poolonly_ch32"
run_experiment_vit "W3" "add" "in1k192_vit_W3_add_ch32"

run_experiment_vit "W3_IMPROVED_WARMUP" "add" "in1k192_vit_W3_improved_warmup_ch32_fuse_multiply" \
  'export WAVELET_WARMUP_STEPS=40000; export WAVELET_SCALE_INIT=0.1; export WAVELET_FUSE_MODE=multiply'

run_experiment_vit "W3_TOKENONLY" "add" "in1k192_vit_W3_tokenonly_ch32" \
  'unset WAVELET_WARMUP_STEPS WAVELET_FUSE_MODE WAVELET_SCALE_INIT'

run_experiment_vit "W3_RESIDUAL" "add" "in1k192_vit_W3_residual_ch32"

