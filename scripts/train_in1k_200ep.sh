#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  echo "Run this script with bash: bash scripts/train_in1k_200ep.sh" >&2
  exit 1
fi
set -euo pipefail

# ImageNet-1K, 200-epoch training for VIL and ViT-Tiny.
# VIL defaults mirror run.bash's ImageNet-1K regularized block, with EPOCHS forced to 200.

NPROC="${NPROC:-8}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-200}"

echo "[train_in1k_200ep] NPROC=${NPROC}, PYTHON_BIN=${PYTHON_BIN}, EPOCHS=${EPOCHS}"

reset_experiment_overrides() {
  unset WAVELET_WARMUP_STEPS WAVELET_SCALE_INIT WAVELET_FUSE_MODE
  unset TOKEN_WAVELET_SCALE_INIT TOKEN_WAVELET_SHRINK TOKEN_WAVELET_HF_ONLY TOKEN_WAVELET_PER_CHANNEL
  unset TOKEN_WAVELET_HIDDEN_CH TOKEN_WAVELET_SIDE_CH TOKEN_WAVELET_SIDE_MODE TOKEN_WAVELET_SIDE_BETA_INIT
  unset TOKEN_WAVELET_INNER_SCALE_INIT TOKEN_WAVELET_OUTER_SCALE_INIT TOKEN_WAVELET_SIDE_SCALE_INIT
  unset WAVELET_MONITOR WAVELET_MONITOR_LOG_EVERY WAVELET_FUSE_SHAPE
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
  export CONFIG="configs/in1k_vit_200ep.yaml"
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

######## VIL: ImageNet-1K regularized ablations, aligned with run.bash ########
# run.bash uses the in1k192 tag prefix even though IMG_SIZE=224; keep that prefix
# for continuity and add _200ep to separate these reruns from the 300ep outputs.

#run_experiment_vil "A1" "none" "in1k192_vil_A1_ch32_reg_200ep"

# run_experiment_vil "W3" "add" "in1k192_vil_W3_add_ch32_reg_200ep" \
#   'unset WAVELET_WARMUP_STEPS WAVELET_SCALE_INIT'

# run_experiment_vil "W3_IMPROVED_WARMUP" "add" "in1k192_vil_W3_improved_warmup_ch32_reg_fuse_add_200ep" \
#   'export WAVELET_WARMUP_STEPS=40000; export WAVELET_SCALE_INIT=0.1; export WAVELET_FUSE_MODE=add'

# run_experiment_vil "W3_IMPROVED_WARMUP" "add" "in1k192_vil_W3_improved_warmup_ch32_reg_fuse_multiply_200ep" \
#   'export WAVELET_WARMUP_STEPS=40000; export WAVELET_SCALE_INIT=0.1; export WAVELET_FUSE_MODE=multiply'

run_experiment_vil "W3_TOKENONLY" "add" "in1k192_vil_W3_tokenonly_ch32_reg_200ep" \
  'unset WAVELET_WARMUP_STEPS WAVELET_FUSE_MODE WAVELET_SCALE_INIT'

run_experiment_vil "W3_RESIDUALONLY" "none" "in1k192_vil_W3_residualonly_ch32_reg_200ep" \
  'unset WAVELET_WARMUP_STEPS WAVELET_SCALE_INIT'

run_experiment_vil "W3_POOL_ONLY" "none" "in1k192_vil_W3_poolonly_ch32_reg_200ep" \
  'unset WAVELET_WARMUP_STEPS'


######## ViT-Tiny: aligned with run.bash, kept commented until needed ########
# run_experiment_vit "A3" "add" "in1k192_vit_A3_ch32_200ep"
# run_experiment_vit "W3_POOL_ONLY" "none" "in1k192_vit_W3_poolonly_ch32_200ep"
# run_experiment_vit "W3" "add" "in1k192_vit_W3_add_ch32_200ep"
# run_experiment_vit "W3_IMPROVED_WARMUP" "add" "in1k192_vit_W3_improved_warmup_ch32_fuse_multiply_200ep" \
#   'export WAVELET_WARMUP_STEPS=40000; export WAVELET_SCALE_INIT=0.1; export WAVELET_FUSE_MODE=multiply'
# run_experiment_vit "W3_TOKENONLY" "add" "in1k192_vit_W3_tokenonly_ch32_200ep"
# run_experiment_vit "W3_RESIDUAL" "add" "in1k192_vit_W3_residual_ch32_200ep"
