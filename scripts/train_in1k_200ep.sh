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

reset_experiment_overrides() {
  unset WAVELET_WARMUP_STEPS WAVELET_SCALE_INIT WAVELET_FUSE_MODE
  unset TOKEN_WAVELET_SCALE_INIT TOKEN_WAVELET_SHRINK TOKEN_WAVELET_HF_ONLY TOKEN_WAVELET_PER_CHANNEL TOKEN_WAVELET_HIDDEN_CH
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

######## VIL (v3b = bk20260417 + HF-only reduce ONLY; orthonormal Haar kept) ########
# Tags use _v3b suffix.
# Iteration history (Tiny VIL last-50-ep mean vs paper bk20260417):
#   v3_bugfix : avg-form Haar + HF-only reduce + shrink=0.02 + wave_alpha=0.1
#               → RESID -0.88%, all others within ±0.3% noise
#   v3_min    : avg-form Haar + HF-only reduce (no shrink/wave_alpha)
#               → RESID -0.88%, others within ±0.3% (shrink+wave_alpha was neutral)
#   v3b       : orthonormal Haar (REVERT) + HF-only reduce (KEEP)
#               → expected RESID parity with paper, others unchanged
# Reason for revert: avg-form Haar halves head-residual signal (no learned absorber),
# costing ~0.88% on RESIDUALONLY. HF-only reduce stays because mix conv handles it.

# Currently active: W3_TOKENONLY + W3_RESIDUALONLY (first batch).
run_experiment_vil "W3_TOKENONLY" "add" "in1k192_vil_W3_tokenonly_ch32_reg_v3b" \
  'unset WAVELET_WARMUP_STEPS WAVELET_FUSE_MODE WAVELET_SCALE_INIT'

run_experiment_vil "W3_RESIDUALONLY" "none" "in1k192_vil_W3_residualonly_ch32_reg_v3b" \
  'unset WAVELET_WARMUP_STEPS WAVELET_FUSE_MODE WAVELET_SCALE_INIT'

# Remaining VIL ablations (uncomment to run):
# run_experiment_vil "A1" "none" "in1k192_vil_A1_ch32_reg_v3b"
# run_experiment_vil "W3_POOL_ONLY" "none" "in1k192_vil_W3_poolonly_ch32_reg_v3b" \
#   'unset WAVELET_WARMUP_STEPS'
# run_experiment_vil "W3_IMPROVED_WARMUP" "add" "in1k192_vil_W3_improved_warmup_ch32_reg_fuse_add_v3b" \
#   'export WAVELET_WARMUP_STEPS=40000; export WAVELET_SCALE_INIT=0.1; export WAVELET_FUSE_MODE=add'

######## ViT-Tiny (uncomment after VIL batch finishes) ########

# run_experiment_vit "A3" "add" "in1k192_vit_A3_ch32_v3b"
# run_experiment_vit "W3_POOL_ONLY" "none" "in1k192_vit_W3_poolonly_ch32_v3b"
# run_experiment_vit "W3_TOKENONLY" "add" "in1k192_vit_W3_tokenonly_ch32_v3b" \
#   'unset WAVELET_WARMUP_STEPS WAVELET_FUSE_MODE WAVELET_SCALE_INIT'
# run_experiment_vit "W3_RESIDUAL" "add" "in1k192_vit_W3_residual_ch32_v3b"
# run_experiment_vit "W3_IMPROVED_WARMUP" "add" "in1k192_vit_W3_improved_warmup_ch32_fuse_add_v3b" \
#   'export WAVELET_WARMUP_STEPS=40000; export WAVELET_SCALE_INIT=0.1; export WAVELET_FUSE_MODE=add'
