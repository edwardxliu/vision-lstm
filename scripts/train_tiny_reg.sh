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
run_experiment "vil" "W3_TOKENONLY" "add" "tiny_vil_W3_tokenonly_ch32_patch8_reg_v3b" \
  'unset WAVELET_WARMUP_STEPS WAVELET_FUSE_MODE WAVELET_SCALE_INIT'

run_experiment "vil" "W3_RESIDUALONLY" "none" "tiny_vil_W3_residualonly_ch32_patch8_reg_v3b" \
  'unset WAVELET_WARMUP_STEPS WAVELET_SCALE_INIT'

# Remaining VIL ablations (uncomment to run):
# run_experiment "vil" "A1" "none" "tiny_vil_A1_ch32_patch8_reg_v3b"
# run_experiment "vil" "W3_POOL_ONLY" "none" "tiny_vil_W3_poolonly_ch32_patch8_reg_v3b"
# run_experiment "vil" "W3" "add" "tiny_vil_W3_add_ch32_patch8_reg_v3b"
# run_experiment "vil" "W3_IMPROVED_WARMUP" "add" "tiny_vil_W3_improved_warmup_ch32_patch8_reg_v3b" \
#   'export WAVELET_SCALE_INIT=0.1; export WAVELET_WARMUP_STEPS=10000; export WAVELET_FUSE_MODE=add'

######## ViT-Tiny (uncomment after VIL batch) ########

# run_experiment "vit_tiny" "A3" "add" "tiny_vit_A3_ch32_patch8_reg_v3b"
# run_experiment "vit_tiny" "W3_POOL_ONLY" "none" "tiny_vit_W3_poolonly_ch32_patch8_reg_v3b"
# run_experiment "vit_tiny" "W3_TOKENONLY" "add" "tiny_vit_W3_tokenonly_ch32_patch8_reg_v3b" \
#   'unset WAVELET_WARMUP_STEPS WAVELET_FUSE_MODE WAVELET_SCALE_INIT'
# run_experiment "vit_tiny" "W3_RESIDUAL" "add" "tiny_vit_W3_residual_ch32_patch8_reg_v3b"
# run_experiment "vit_tiny" "W3_IMPROVED_WARMUP" "add" "tiny_vit_W3_improved_warmup_ch32_patch8_reg_v3b" \
#   'export WAVELET_SCALE_INIT=0.1; export WAVELET_WARMUP_STEPS=10000; export WAVELET_FUSE_MODE=add'

