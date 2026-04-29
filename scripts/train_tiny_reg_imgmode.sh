#!/usr/bin/env bash
set -euo pipefail

# Tiny-ImageNet image-mode (DWT on raw RGB) ablation, mirroring the IN-1k v2
# fancy setup but with the few scale-dependent knobs adjusted for Tiny.
#
# Compare against the existing stem-mode baseline tiny_vil_W3_tokenonly_ch32_patch8_reg
# (val_acc=46.06%, IN-C mean=12.88%, see outputs_pswf_paper_20260225/开正则/VIL).
#
# Scale-dependent adjustments vs train_in1k_200ep.sh v2:
#   - WAVELET_WARMUP_STEPS: 40000 -> 3000
#       IN-1k has ~5000 steps/epoch x 200ep = 1M steps (40000 = 4% warmup).
#       Tiny has ~100 steps/epoch x 300ep ~= 29k steps; 3000 ~= 10% warmup.
#   - WAVELET_MONITOR_LOG_EVERY: 100 -> 50
#       Tiny only has ~98 steps/epoch, so log_every=100 = 1 log/epoch (too sparse).
#
# Cutmix env vars from IN-1k WAVELET_CUTMIX_ENV are NOT re-exported here because
# configs/tiny_reg_vil.yaml already sets CUTMIX_ALPHA=1.0 / MIXUP_ALPHA=0.0 /
# MIXUP_PROB=1.0 / SWITCH_PROB=1.0.
#
# Channel-/shape-related flags (HIDDEN_CH, SIDE_CH, SCALE_INIT, FUSE_SHAPE,
# SIDE_NONNEG, SIDE_MODE, SPLIT_BANDS, OUTER_GATE, SHRINK, HF_ONLY,
# PER_CHANNEL) are kept identical to IN-1k v2 since they operate on channel/token
# dims rather than spatial size.

NPROC="${NPROC:-8}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[train_tiny_reg_imgmode] NPROC=${NPROC}, PYTHON_BIN=${PYTHON_BIN}"

reset_experiment_overrides() {
  unset WAVELET_WARMUP_STEPS WAVELET_SCALE_INIT WAVELET_FUSE_MODE
  unset TOKEN_WAVELET_SCALE_INIT TOKEN_WAVELET_SHRINK TOKEN_WAVELET_HF_ONLY TOKEN_WAVELET_PER_CHANNEL TOKEN_WAVELET_HIDDEN_CH
  unset TOKEN_WAVELET_INNER_SCALE_INIT TOKEN_WAVELET_OUTER_SCALE_INIT
  unset TOKEN_WAVELET_SIDE_CH TOKEN_WAVELET_SIDE_MODE TOKEN_WAVELET_SIDE_BETA_INIT TOKEN_WAVELET_OUTER_GATE TOKEN_WAVELET_SPLIT_BANDS
  unset WAVELET_INPUT_IMAGE
  unset WAVELET_FUSE_SHAPE WAVELET_SIDE_NONNEG TOKEN_WAVELET_SIDE_SCALE_INIT
  unset WAVELET_MONITOR WAVELET_MONITOR_LOG_EVERY
  unset MIXUP_ALPHA CUTMIX_ALPHA MIXUP_PROB SWITCH_PROB
}

run_experiment() {
  local model_kind="$1"
  local ablation="$2"
  local dwt_fuse="$3"
  local run_tag="$4"
  local extra_env="${5:-}"

  reset_experiment_overrides
  export MODEL_KIND="${model_kind}"
  if [[ "${model_kind}" == "vil" ]]; then
    export CONFIG="configs/tiny_reg_vil.yaml"
  else
    export CONFIG="configs/tiny_reg_vit.yaml"
  fi

  export ABLATION="${ablation}"
  export DWT_FUSE="${dwt_fuse}"
  export RUN_TAG="${run_tag}"

  if [[ -n "${extra_env}" ]]; then
    eval "${extra_env}"
  fi

  echo "[train_tiny_reg_imgmode] MODEL_KIND=${MODEL_KIND} ABLATION=${ABLATION} DWT_FUSE=${DWT_FUSE} RUN_TAG=${RUN_TAG}"
  ${PYTHON_BIN} -m torch.distributed.run --nproc_per_node="${NPROC}" train_ablation_ddp.py
}

######## VIL: W3_TOKENONLY image-mode (fancy setup, scaled for Tiny) ########
run_experiment "vil" "W3_TOKENONLY" "add" "tiny_vil_W3_tokenonly_ch32_patch8_reg_imgmode_v2" \
  "export WAVELET_MONITOR=1; export WAVELET_MONITOR_LOG_EVERY=50; \
   export WAVELET_WARMUP_STEPS=3000; \
   unset WAVELET_FUSE_MODE WAVELET_SCALE_INIT; \
   export TOKEN_WAVELET_INNER_SCALE_INIT=1.0; \
   export TOKEN_WAVELET_OUTER_SCALE_INIT=0.0; \
   export TOKEN_WAVELET_SIDE_SCALE_INIT=0.1; \
   export TOKEN_WAVELET_SHRINK=0.02; \
   export TOKEN_WAVELET_HF_ONLY=1; \
   export TOKEN_WAVELET_PER_CHANNEL=1; \
   export TOKEN_WAVELET_HIDDEN_CH=64; \
   export TOKEN_WAVELET_SIDE_CH=16; \
   export TOKEN_WAVELET_SIDE_MODE=patch; \
   export TOKEN_WAVELET_SIDE_BETA_INIT=0.1; \
   export TOKEN_WAVELET_OUTER_GATE=1; \
   export TOKEN_WAVELET_SPLIT_BANDS=1; \
   export WAVELET_INPUT_IMAGE=1; \
   export WAVELET_FUSE_SHAPE=add_sigmoid_safe; \
   export WAVELET_SIDE_NONNEG=1"
