#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  echo "Run this script with bash: bash scripts/train_in1k_200ep.sh" >&2
  exit 1
fi
set -euo pipefail

# ImageNet-1K, 200-epoch training for VIL and ViT-Tiny.
# Shared defaults come from the 300-epoch configs, with EPOCHS forced to 200 here.

NPROC="${NPROC:-8}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-200}"

echo "[train_in1k_200ep] NPROC=${NPROC}, PYTHON_BIN=${PYTHON_BIN}"

WAVELET_SANITY_ENV='export CUTMIX_ALPHA=0.0; export MIXUP_ALPHA=0.8; export MIXUP_PROB=1.0; export SWITCH_PROB=0.0; export WAVELET_MONITOR=1; export WAVELET_MONITOR_LOG_EVERY=100'
WAVELET_CUTMIX_ENV='export CUTMIX_ALPHA=1.0; export MIXUP_ALPHA=0.0; export MIXUP_PROB=1.0; export SWITCH_PROB=1.0; export WAVELET_MONITOR=1; export WAVELET_MONITOR_LOG_EVERY=100'

reset_experiment_overrides() {
  unset WAVELET_WARMUP_STEPS WAVELET_SCALE_INIT WAVELET_FUSE_MODE
  unset TOKEN_WAVELET_SCALE_INIT TOKEN_WAVELET_SHRINK TOKEN_WAVELET_HF_ONLY TOKEN_WAVELET_PER_CHANNEL TOKEN_WAVELET_HIDDEN_CH
  unset TOKEN_WAVELET_INNER_SCALE_INIT TOKEN_WAVELET_OUTER_SCALE_INIT
  unset TOKEN_WAVELET_SIDE_CH TOKEN_WAVELET_SIDE_MODE TOKEN_WAVELET_SIDE_BETA_INIT TOKEN_WAVELET_OUTER_GATE TOKEN_WAVELET_SPLIT_BANDS
  unset WAVELET_MONITOR WAVELET_MONITOR_LOG_EVERY
  unset MIXUP_ALPHA CUTMIX_ALPHA MIXUP_PROB SWITCH_PROB
}

run_experiment_vil() {
  local ablation="$1"
  local dwt_fuse="$2"
  local run_tag="$3"
  local extra_env="${4:-}"

  reset_experiment_overrides
  export MODEL_KIND="vil"
  export CONFIG="configs/in1k_vil_200ep.yaml"
  export EPOCHS="${EPOCHS}"
  export ABLATION="${ablation}"
  export DWT_FUSE="${dwt_fuse}"
  export RUN_TAG="${run_tag}"

  if [[ -n "${extra_env}" ]]; then
    eval "${extra_env}"
  fi

  echo "[train_in1k_200ep][VIL] ABLATION=${ABLATION} DWT_FUSE=${DWT_FUSE} RUN_TAG=${RUN_TAG}"
  ${PYTHON_BIN} -m torch.distributed.run --nproc_per_node="${NPROC}" train_ablation_ddp.py
}

run_experiment_vit() {
  local ablation="$1"
  local dwt_fuse="$2"
  local run_tag="$3"
  local extra_env="${4:-}"

  reset_experiment_overrides
  export MODEL_KIND="vit_tiny"
  export CONFIG="configs/in1k_vit_300ep.yaml"
  export EPOCHS="${EPOCHS}"
  export ABLATION="${ablation}"
  export DWT_FUSE="${dwt_fuse}"
  export RUN_TAG="${run_tag}"

  if [[ -n "${extra_env}" ]]; then
    eval "${extra_env}"
  fi

  echo "[train_in1k_200ep][ViT] ABLATION=${ABLATION} DWT_FUSE=${DWT_FUSE} RUN_TAG=${RUN_TAG}"
  ${PYTHON_BIN} -m torch.distributed.run --nproc_per_node="${NPROC}" train_ablation_ddp.py
}

######## VIL ########

# run_experiment_vil "A1" "none" "in1k192_vil_A1_ch32_reg"
# run_experiment_vil "W3_POOL_ONLY" "none" "in1k192_vil_W3_poolonly_ch32_reg" 'unset WAVELET_WARMUP_STEPS'
# #run_experiment_vil "W3" "add" "in1k192_vil_W3_add_ch32_reg_mixuponly_dbg" \
# #  "${WAVELET_SANITY_ENV}; export WAVELET_SCALE_INIT=0.1; unset WAVELET_WARMUP_STEPS WAVELET_FUSE_MODE"

# run_experiment_vil "W3_IMPROVED_WARMUP" "add" "in1k192_vil_W3_improved_warmup_ch32_reg_fuse_add_mixuponly_dbg" \
#   "${WAVELET_SANITY_ENV}; export WAVELET_WARMUP_STEPS=40000; export WAVELET_SCALE_INIT=0.1; export WAVELET_FUSE_MODE=add"

# run_experiment_vil "W3_IMPROVED_WARMUP" "add" "in1k192_vil_W3_improved_warmup_ch32_reg_fuse_multiply_mixuponly_dbg" \
#   "${WAVELET_SANITY_ENV}; export WAVELET_WARMUP_STEPS=40000; export WAVELET_SCALE_INIT=0.1; export WAVELET_FUSE_MODE=multiply"

run_experiment_vil "W3_TOKENONLY" "add" "in1k192_vil_W3_tokenonly_ch32_reg_hfalpha01_shrink002_cutmixonly_xgate_relmod_sidepatch16_split_i1_o05_b01_dbg" \
  "${WAVELET_CUTMIX_ENV}; export WAVELET_WARMUP_STEPS=40000; unset WAVELET_FUSE_MODE WAVELET_SCALE_INIT; \
   export TOKEN_WAVELET_INNER_SCALE_INIT=1.0; \
   export TOKEN_WAVELET_OUTER_SCALE_INIT=0.5; \
   export TOKEN_WAVELET_SHRINK=0.02; \
   export TOKEN_WAVELET_HF_ONLY=1; \
   export TOKEN_WAVELET_PER_CHANNEL=1; \
   export TOKEN_WAVELET_HIDDEN_CH=64; \
   export TOKEN_WAVELET_SIDE_CH=16; \
   export TOKEN_WAVELET_SIDE_MODE=patch; \
   export TOKEN_WAVELET_SIDE_BETA_INIT=0.1; \
   export TOKEN_WAVELET_OUTER_GATE=1; \
   export TOKEN_WAVELET_SPLIT_BANDS=1"

# run_experiment_vil "W3_RESIDUALONLY" "none" "in1k192_vil_W3_residualonly_ch32_reg_mixuponly_dbg" \
#   "${WAVELET_SANITY_ENV}; export WAVELET_SCALE_INIT=0.1; unset WAVELET_WARMUP_STEPS WAVELET_FUSE_MODE"

# ######## ViT-Tiny ########

# run_experiment_vit "A3" "add" "in1k192_vit_A3_ch32"
# run_experiment_vit "W3_POOL_ONLY" "none" "in1k192_vit_W3_poolonly_ch32"
# run_experiment_vit "W3" "add" "in1k192_vit_W3_add_ch32_mixuponly_dbg" \
#   "${WAVELET_SANITY_ENV}; export WAVELET_SCALE_INIT=0.1; unset WAVELET_WARMUP_STEPS WAVELET_FUSE_MODE"

# run_experiment_vit "W3_IMPROVED_WARMUP" "add" "in1k192_vit_W3_improved_warmup_ch32_fuse_multiply_mixuponly_dbg" \
#   "${WAVELET_SANITY_ENV}; export WAVELET_WARMUP_STEPS=40000; export WAVELET_SCALE_INIT=0.1; export WAVELET_FUSE_MODE=multiply"

# run_experiment_vit "W3_TOKENONLY" "add" "in1k192_vit_W3_tokenonly_ch32_mixuponly_dbg" \
#   "${WAVELET_SANITY_ENV}; unset WAVELET_WARMUP_STEPS WAVELET_FUSE_MODE WAVELET_SCALE_INIT"

# run_experiment_vit "W3_RESIDUAL" "add" "in1k192_vit_W3_residual_ch32_mixuponly_dbg" \
#   "${WAVELET_SANITY_ENV}; export WAVELET_SCALE_INIT=0.1; unset WAVELET_WARMUP_STEPS WAVELET_FUSE_MODE"
