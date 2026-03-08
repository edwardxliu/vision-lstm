pip download torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118 --proxy http://localhost:10808 --only-binary=:all: --platform manylinux2014_x86_64 --implementation cp --python-version 3.9 --abi cp39

ssh -N -L 8888:127.0.0.1:8888 omnisky@10.12.49.211

export http_proxy=http://10.12.49.11:10808
export https_proxy=http://10.12.49.11:10808
export all_proxy=http://10.12.49.11:10808

export RESUME_LR=3e-5
export RESUME_CKPT=best_model_stage2.pth
PYTHONUNBUFFERED=1

tmux new -s train -d "RESUME_LR=3e-5 RESUME_CKPT=best_model_stage2.pth PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 train_extra.py > /home/omnisky/edward_train.log 2>&1"

tmux new -s train -d "torchrun --nproc_per_node=8 lstm6_stage1_pretrain_192.py > /home/omnisky/edward_train_lstm6.log 2>&1"


tmux new -s train -d "torchrun --nproc_per_node=8 lstm6_stage1_pretrain_192_sample.py > /home/omnisky/edward_train_lstm6.log 2>&1"


tmux new -s train
# 进入 tmux 后
export PYRAMID=full
export MERGE_KINDS=haar,patch,patch
export LAST_MERGE_KIND=patch
export USE_DWT=1

torchrun --nproc_per_node=8 lstm6_stage1_pretrain_192_sample.py > /home/omnisky/edward_train_lstm6.log 2>&1

试验1
tmux new -s train -d \
"PRETRAIN_CKPT=lstm6_half_only.pth PATCH_SIZE=16 STRIDE=16 EPOCHS=150 PYRAMID=half MERGE_KIND=patch USE_DWT=0 BRANCH_ALPHA_MAX=0 STAGE_DIMS=384,384 STAGE_DEPTHS=2,6 COL_EVERY=0 MIXER_EVERY=9999\
 torchrun --nproc_per_node=8 lstm6_stage1_pretrain_192_sample.py \
 > /home/omnisky/edward_train_lstm6.log 2>&1"

试验2
tmux new -s train -d \
"PRETRAIN_CKPT=lstm6_none.pth PATCH_SIZE=16 STRIDE=16 EPOCHS=150 PYRAMID=none MERGE_KIND=patch USE_DWT=0 BRANCH_ALPHA_MAX=0 COL_EVERY=0 MIXER_EVERY=9999\
 torchrun --nproc_per_node=8 lstm6_stage1_pretrain_192_sample.py \
 > /home/omnisky/edward_train_lstm6.log 2>&1"

试验3
tmux new -s train -d \
"PRETRAIN_CKPT=lstm6_half_patchsize8.pth PATCH_SIZE=8 STRIDE=8 EPOCHS=150 PYRAMID=half MERGE_KIND=patch USE_DWT=0 BRANCH_ALPHA_MAX=0 STAGE_DIMS=256,384 STAGE_DEPTHS=0,8 COL_EVERY=0 MIXER_EVERY=9999\
 torchrun --nproc_per_node=8 lstm6_stage1_pretrain_192_sample.py \
 > /home/omnisky/edward_train_lstm6.log 2>&1"


PATCH_SIZE=8 试验1: COL_EVERY=4
tmux new -s train -d \
"PRETRAIN_CKPT=lstm6_half_patchsize8_col.pth SUBSET_CLASSES=500 PATCH_SIZE=8 STRIDE=8 EPOCHS=80 PYRAMID=half MERGE_KIND=patch USE_DWT=0 BRANCH_ALPHA_MAX=0 STAGE_DIMS=256,384 STAGE_DEPTHS=0,8 COL_EVERY=4 MIXER_EVERY=9999 \
torchrun --nproc_per_node=8 lstm6_stage1_pretrain_192_sample.py \
 > /home/omnisky/lstm6_half_patchsize8_col.log 2>&1"

PATCH_SIZE=8 试验2: COL_EVERY=4 + MIXER_EVERY=4
tmux new -s train -d \
"PRETRAIN_CKPT=lstm6_half_patchsize8_col_mixer.pth SUBSET_CLASSES=500 PATCH_SIZE=8 STRIDE=8 EPOCHS=80 PYRAMID=half MERGE_KIND=patch USE_DWT=0 BRANCH_ALPHA_MAX=0 STAGE_DIMS=256,384 STAGE_DEPTHS=0,8 COL_EVERY=4 MIXER_EVERY=4 \
torchrun --nproc_per_node=8 lstm6_stage1_pretrain_192_sample.py \
 > /home/omnisky/lstm6_half_patchsize8_col_mixer.log 2>&1"


PATCH_SIZE=8 试验3: COL_EVERY=4 + MIXER_EVERY=4 + BRANCH
tmux new -s train -d \
"PRETRAIN_CKPT=lstm6_half_patchsize8_col_mixer_branch.pth SUBSET_CLASSES=500 PATCH_SIZE=8 STRIDE=8 EPOCHS=80 PYRAMID=half MERGE_KIND=patch USE_DWT=0 BRANCH_ALPHA_MAX=1e-3 BRANCH_START=25 BRANCH_RAMP=60 STAGE_DIMS=256,384 STAGE_DEPTHS=0,8 COL_EVERY=4 MIXER_EVERY=4 \
torchrun --nproc_per_node=8 lstm6_stage1_pretrain_192_sample.py \
 > /home/omnisky/lstm6_half_patchsize8_col_mixer_branch.log 2>&1"

试验A0: 
tmux new -s train -d \
"PRETRAIN_CKPT=lstm5_a0.pth ABLATION=A0 SUBSET_CLASSES=500 PATCH_SIZE=16 STRIDE=16 EPOCHS=150 \
torchrun --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation.py \
 > /home/omnisky/lstm5_a0.log 2>&1"


export KD_ON=1
export TEACHER=convnext_base
export KD_ALPHA=0.3
export KD_T=4.0
export KD_WARMUP_EPOCHS=2
export TEACHER=convnext_base
export TEACHER_WEIGHTS=./convnext_base-6075fbad.pth
tmux new -s train -d "torchrun --nproc_per_node=8 stage2_kd_224.py > /home/omnisky/edward_train.log 2>&1"


python eval_stage2_alpha.py \
  --data /home/omnisky/Public/edward/workspace/data/imagenet_dataset \
  --ckpt vil_stage2_224.pth \
  --alpha 0 0.003


tmux new -s v5ab -d "bash -lc '
set -e

source /home/omnisky/anaconda3/etc/profile.d/conda.sh
conda activate d2l

export SUBSET_CLASSES=500
export SUBSET_SEED=1234
export TRAIN_SAMPLES_PER_CLASS=500
export IMG_SIZE=192
export EPOCHS=150
export PER_GPU_BATCH=32
export ACCUM_STEPS=1
export AMP_DTYPE=bf16

export DIM=192
export DEPTH=12
export FEAT_CH=32,64,64
export PATCH_SIZE=16
export STRIDE=16
export AUTO_PATCH_DWT=1

for A in A0 A1 A2 A3; do
  export ABLATION=$A
  echo \"===== Running \$A at \$(date) =====\"
  torchrun --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation.py \
    > /home/omnisky/lstm5_\${A}_sample500.log 2>&1
  echo \"===== Done \$A at \$(date) =====\"
done
'"



tmux new -s v5ab -d "bash -lc '
set -e
source /home/omnisky/anaconda3/etc/profile.d/conda.sh
conda activate d2l
echo ENV=$CONDA_DEFAULT_ENV
which python
python -c \"import torch; print(torch.__version__, torch.__file__) \"
which torchrun || true
torchrun --version || true

export SUBSET_CLASSES=150
export SUBSET_SEED=1234
export TRAIN_SAMPLES_PER_CLASS=200
export EPOCHS=150

export IMG_SIZE=192
export PER_GPU_BATCH=32
export ACCUM_STEPS=1
export AMP_DTYPE=bf16

export DIM=192
export DEPTH=12
export FEAT_CH=32,64,64
export PATCH_SIZE=16
export STRIDE=16
export AUTO_PATCH_DWT=1

export ABLATION=W3
export DWT_FUSE=gated
python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation.py \
    > /home/omnisky/lstm5_W3_gated_sample150.log 2>&1

export ABLATION=W3
export DWT_FUSE=LL
python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation.py \
    > /home/omnisky/lstm5_W3_LL_sample150.log 2>&1


export ABLATION=C1
export DWT_FUSE=gated
export BASELINE_ABLATION=W3
python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation.py \
    > /home/omnisky/lstm5_C1_gated_sample150.log 2>&1

export ABLATION=C1
export DWT_FUSE=add
export BASELINE_ABLATION=W3
python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation.py \
    > /home/omnisky/lstm5_C1_add_sample150.log 2>&1

export ABLATION=C1
export DWT_FUSE=LL
export BASELINE_ABLATION=W3
python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation.py \
    > /home/omnisky/lstm5_C1_LL_sample150.log 2>&1
'"





tmux new -s v5ab -d "bash -lc '
set -e
source /home/omnisky/anaconda3/etc/profile.d/conda.sh
conda activate d2l
echo ENV=$CONDA_DEFAULT_ENV
which python
python -c \"import torch; print(torch.__version__, torch.__file__) \"
which torchrun || true
torchrun --version || true

export NUM_WORKERS=7

export SUBSET_CLASSES=0
export SUBSET_SEED=1234
export TRAIN_SAMPLES_PER_CLASS=0
export VAL_SAMPLES_PER_CLASS=0
export EPOCHS=400

export IMG_SIZE=192
export PER_GPU_BATCH=32
export ACCUM_STEPS=1
export AMP_DTYPE=bf16

export DIM=192
export DEPTH=12
export FEAT_CH=32
export PATCH_SIZE=16
export STRIDE=16
export AUTO_PATCH_DWT=1

export ABLATION=W3
export DWT_FUSE=gated
export DISABLE_BRANCH=1

export RUN_TAG=W3_gated_ch32_full
export RESUME_CKPT=outputs_lstm5_stage1/W3_gated_ch32_full_ema_best.pth

python -m torch.distributed.run --nproc_per_node=7 lstm5_stage1_pretrain_192_sample_ablation.py \
    > /home/omnisky/lstm5_W3_gated_ch32.log 2>&1
'"    





tmux new -s v5ab -d "bash -lc '
set -e
source /home/omnisky/anaconda3/etc/profile.d/conda.sh
conda activate d2l
echo ENV=$CONDA_DEFAULT_ENV
which python
python -c \"import torch; print(torch.__version__, torch.__file__) \"
which torchrun || true
torchrun --version || true

export SUBSET_CLASSES=0
export TRAIN_SAMPLES_PER_CLASS=0
export VAL_SAMPLES_PER_CLASS=0
export EPOCHS=30

export IMG_SIZE=224
export PER_GPU_BATCH=32
export ACCUM_STEPS=1
export AMP_DTYPE=bf16

export BASE_LR=3e-5
export WARMUP_EPOCHS=2
export WEIGHT_DECAY=0.05
export EMA_DECAY=0.9997

export MIXUP_ALPHA=0.1
export CUTMIX_ALPHA=0
export MIXUP_PROB=0
export LABEL_SMOOTH=0

export DROP_PATH=0
export DROP_PATH_DECAY=0

export DIM=192
export DEPTH=12
export FEAT_CH=32
export PATCH_SIZE=16
export STRIDE=16
export AUTO_PATCH_DWT=1

export ABLATION=W3
export DWT_FUSE=add
export DISABLE_BRANCH=1

export OUT_DIR=./outputs_lstm5_stage2
export RUN_TAG=W3_gated_ch32_ft224

export RESUME_CKPT=outputs_lstm5_stage1/W3_gated_ch32_full_ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation.py \
    > /home/omnisky/lstm5_W3_gated_lr3e_ch32_stage2.log 2>&1

export BASE_LR=5e-5
python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation.py \
    > /home/omnisky/lstm5_W3_add_lr5e_stage2.log 2>&1

export BASE_LR=8e-5
python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation.py \
    > /home/omnisky/lstm5_W3_add_lr8e_stage2.log 2>&1


export BASE_LR=5e-5
export WARMUP_EPOCHS=2
export DWT_FUSE=gated
python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation.py \
    > /home/omnisky/lstm5_W3_gated_lr5e_stage2_warmup2.log 2>&1
'"    




tmux new -s v5ab -d "bash -lc '
set -e
source /home/omnisky/anaconda3/etc/profile.d/conda.sh
conda activate d2l
echo ENV=$CONDA_DEFAULT_ENV
which python
python -c \"import torch; print(torch.__version__, torch.__file__) \"
which torchrun || true
torchrun --version || true

export SUBSET_CLASSES=150
export SUBSET_SEED=1234
export TRAIN_SAMPLES_PER_CLASS=200
export EPOCHS=150

export IMG_SIZE=192
export PER_GPU_BATCH=32
export ACCUM_STEPS=1
export AMP_DTYPE=bf16

export DIM=192
export DEPTH=12
export FEAT_CH=32
export PATCH_SIZE=16
export STRIDE=16
export AUTO_PATCH_DWT=1

export ABLATION=W3
export DWT_FUSE=add
export RUN_TAG=W3_ch32
export RESUME_CKPT=outputs_lstm5_stage1/W3_ch32_ema_best.pth
python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation.py \
    > /home/omnisky/lstm5_W3_add_featCH32_sample150.log 2>&1
'"











#==============================================
#Smoke Test
#==============================================

tmux new -s v5ab -d "bash -lc '
set -e
source /home/omnisky/anaconda3/etc/profile.d/conda.sh
conda activate d2l
echo ENV=$CONDA_DEFAULT_ENV
which python
python -c \"import torch; print(torch.__version__, torch.__file__) \"
which torchrun || true
torchrun --version || true


export DATASET=tiny_imagenet
export DATA_ROOT=../data/tiny-imagenet-200
export MODEL_KIND=vil
export IMG_SIZE=64
export EPOCHS=2
export PER_GPU_BATCH=128
export ACCUM_STEPS=1
export AMP_DTYPE=bf16

export DIM=192
export DEPTH=12
export FEAT_CH=32
export PATCH_SIZE=16
export STRIDE=16
export AUTO_PATCH_DWT=1

export ABLATION=A1
export DWT_FUSE=none
export DISABLE_BRANCH=1

export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=smoke_tiny_A1

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/smoke_tiny_A1.log 2>&1
'"

#==============================================
# 正则化测试:  Baseline: A1(没有 PSWF)测试
#==============================================
tmux new -s v5ab -d "bash -lc '
set -e
source /home/omnisky/anaconda3/etc/profile.d/conda.sh
conda activate d2l
echo ENV=$CONDA_DEFAULT_ENV
which python
python -c \"import torch; print(torch.__version__, torch.__file__) \"
which torchrun || true
torchrun --version || true

export DATA_SEED=1234
export DATASET=tiny_imagenet
export DATA_ROOT=../data/tiny-imagenet-200
export MODEL_KIND=vil
export IMG_SIZE=64
export EPOCHS=300
export PER_GPU_BATCH=128
export ACCUM_STEPS=1
export AMP_DTYPE=bf16

export DIM=192
export DEPTH=12
export FEAT_CH=32
export PATCH_SIZE=8
export STRIDE=8
export AUTO_PATCH_DWT=1

export LABEL_SMOOTH=0.1
export MIXUP_PROB=1.0
export CUTMIX_ALPHA=1.0
export MIXUP_ALPHA=0.0
export SWITCH_PROB=1.0
export DISABLE_BRANCH=1

export OUT_DIR=./outputs_pswf_paper

#==============================================
# Baseline: A1
#==============================================
export ABLATION=A1
export DWT_FUSE=none

export RUN_TAG=tiny_vil_A1_ch32_patch8_reg

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_A1_ch32_patch8_reg.log 2>&1

#==============================================
# (3) 关键消融: W3_POOL_ONLY(证明 wavelet 分支是否真的贡献)
#==============================================
export ABLATION=W3_POOL_ONLY
export DWT_FUSE=none
export RUN_TAG=tiny_vil_W3_poolonly_ch32_patch8_reg

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_W3_poolonly_ch32_patch8_reg.log 2>&1

#==============================================
# (2) PSWF 主线: W3(scale=0 + 加性融合, 与 W3_IMPROVED 一致, 只保留 W3)
#==============================================
export ABLATION=W3
export DWT_FUSE=add
export RUN_TAG=tiny_vil_W3_add_ch32_patch8_reg

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_W3_add_ch32_patch8_reg.log 2>&1


#==============================================
# (4) W3_IMPROVED_WARMUP + 小波融合模式对比(add vs multiply, 合并原4与4b, 去重)
# 须设 WAVELET_SCALE_INIT 非 0，否则 effective_scale=0 时 add/multiply 完全等价、结果一致
# wavelet_warmup_steps: 控制小波残差的 warmup 步数（Tiny-ImageNet VIL 300 epochs 下，20000 步约占前 2/3 训练）
#==============================================
export ABLATION=W3_IMPROVED_WARMUP
export DWT_FUSE=add
export WAVELET_SCALE_INIT=0.1
export WAVELET_WARMUP_STEPS=10000

# 加性融合（主 RUN_TAG, 与原先 (4) 一致）
export WAVELET_FUSE_MODE=add
export RUN_TAG=tiny_vil_W3_improved_warmup_ch32_patch8_reg
python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_W3_improved_warmup_ch32_patch8_reg.log 2>&1

# 乘性融合
export WAVELET_FUSE_MODE=multiply
export RUN_TAG=tiny_vil_W3_improved_warmup_ch32_patch8_reg_fuse_multiply
python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_W3_improved_warmup_ch32_patch8_reg_fuse_multiply.log 2>&1

#==============================================
# (4c) W3_TOKENONLY(只开 post-stem concat+1×1, 关掉 head residual)
#==============================================
export ABLATION=W3_TOKENONLY
export DWT_FUSE=add

unset WAVELET_WARMUP_STEPS
unset WAVELET_FUSE_MODE
unset WAVELET_SCALE_INIT

export RUN_TAG=tiny_vil_W3_tokenonly_ch32_patch8_reg
python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_W3_tokenonly_ch32_patch8_reg.log 2>&1

#==============================================
# (4d) W3_RESIDUALONLY(主路 pool-only, 单独开 head wavelet residual)
#==============================================
export ABLATION=W3_RESIDUALONLY
export DWT_FUSE=none

unset WAVELET_WARMUP_STEPS
unset WAVELET_SCALE_INIT

export RUN_TAG=tiny_vil_W3_residualonly_ch32_patch8_reg
python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_W3_residualonly_ch32_patch8_reg.log 2>&1


#==============================================
# Plug-and-Play: ViT-Tiny(Tiny 上先跑两条就够)
# ViT baseline
#==============================================
export DATASET=tiny_imagenet
export DATA_ROOT=../data/tiny-imagenet-200
export MODEL_KIND=vit_tiny
export IMG_SIZE=64
export EPOCHS=300
export PER_GPU_BATCH=128
export ACCUM_STEPS=1
export AMP_DTYPE=bf16

export DIM=192
export DEPTH=12
export FEAT_CH=32
export PATCH_SIZE=8
export STRIDE=8
export AUTO_PATCH_DWT=1

export LABEL_SMOOTH=0.1
export MIXUP_PROB=1.0
export CUTMIX_ALPHA=1.0
export MIXUP_ALPHA=0.0
export SWITCH_PROB=1.0

export ABLATION=A3
export DWT_FUSE=add
export DISABLE_BRANCH=1

export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=tiny_vit_A3_ch32_patch8_reg

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vit_A3_ch32_patch8_reg.log 2>&1

#==============================================
# ViT + Pool Only
#==============================================
export ABLATION=W3_POOL_ONLY
export DWT_FUSE=none

export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=tiny_vit_W3_poolonly_ch32_patch8_reg

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vit_W3_poolonly_ch32_patch8_reg.log 2>&1

#==============================================
# ViT + PSWF(只要把 ABLATION 改成 W3 即可启用 PSWF tokenizer)
#==============================================

export ABLATION=W3
export DWT_FUSE=add
export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=tiny_vit_W3_add_ch32_patch8_reg

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vit_W3_add_ch32_patch8_reg.log 2>&1

#==============================================
# ViT + W3_IMPROVED_WARMUP (Tiny, 正则版, 加性 vs 乘性)
# 与 VIL Tiny 正则部分保持一致：使用 W3_IMPROVED_WARMUP + WAVELET_WARMUP_STEPS=10000 + WAVELET_SCALE_INIT=0.1
# 通过 WAVELET_FUSE_MODE=add/multiply 对比融合方式
#==============================================
export ABLATION=W3_IMPROVED_WARMUP
export DWT_FUSE=add
export WAVELET_WARMUP_STEPS=10000
export WAVELET_SCALE_INIT=0.1

# 加性融合
export WAVELET_FUSE_MODE=add
export RUN_TAG=tiny_vit_W3_improved_warmup_ch32_patch8_reg
python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vit_W3_improved_warmup_ch32_patch8_reg.log 2>&1

# 乘性融合
export WAVELET_FUSE_MODE=multiply
export RUN_TAG=tiny_vit_W3_improved_warmup_ch32_patch8_reg_fuse_multiply
python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vit_W3_improved_warmup_ch32_patch8_reg_fuse_multiply.log 2>&1

# 清理 wavelet 相关环境变量，避免影响后续 TOKENONLY / RESIDUAL
unset WAVELET_WARMUP_STEPS
unset WAVELET_FUSE_MODE
unset WAVELET_SCALE_INIT

#==============================================
# ViT + W3_TOKENONLY(仅 tokenization, 无 head modulation，便于写清可插拔 tokenization 收益)
#==============================================
export ABLATION=W3_TOKENONLY
export DWT_FUSE=add
export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=tiny_vit_W3_tokenonly_ch32_patch8_reg

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vit_W3_tokenonly_ch32_patch8_reg.log 2>&1

#==============================================
# ViT + W3_RESIDUAL(主路 pool-only, 小波单独残差调 CLS)
#==============================================
export ABLATION=W3_RESIDUAL
export DWT_FUSE=add

export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=tiny_vit_W3_residual_ch32_patch8_reg

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vit_W3_residual_ch32_patch8_reg.log 2>&1
'"

#==============================================
# Tiny-ImageNet 无正则 ABLATION（与有正则测试内容一致，仅关闭正则）
#==============================================
tmux new -s v5ab_tiny_noreg -d "bash -lc '
set -e
source /home/omnisky/anaconda3/etc/profile.d/conda.sh
conda activate d2l
echo ENV=$CONDA_DEFAULT_ENV
which python
python -c \"import torch; print(torch.__version__, torch.__file__) \"
which torchrun || true
torchrun --version || true

export DATA_SEED=1234
export DATASET=tiny_imagenet
export DATA_ROOT=../data/tiny-imagenet-200
export MODEL_KIND=vil
export IMG_SIZE=64
export EPOCHS=300
export PER_GPU_BATCH=128
export ACCUM_STEPS=1
export AMP_DTYPE=bf16

export DIM=192
export DEPTH=12
export FEAT_CH=32
export PATCH_SIZE=8
export STRIDE=8
export AUTO_PATCH_DWT=1

# 无正则
export LABEL_SMOOTH=0.0
export MIXUP_PROB=0.0
export CUTMIX_ALPHA=0.0
export MIXUP_ALPHA=0.0
export SWITCH_PROB=0.0
export DISABLE_BRANCH=1

export OUT_DIR=./outputs_pswf_paper

#==============================================
# Baseline: A1
#==============================================
export ABLATION=A1
export DWT_FUSE=none

export RUN_TAG=tiny_vil_A1_ch32_patch8_noreg

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_A1_ch32_patch8_noreg.log 2>&1

#==============================================
# W3_POOL_ONLY
#==============================================
export ABLATION=W3_POOL_ONLY
export DWT_FUSE=none
export RUN_TAG=tiny_vil_W3_poolonly_ch32_patch8_noreg

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_W3_poolonly_ch32_patch8_noreg.log 2>&1

#==============================================
# W3(scale=0 + 加性融合)
#==============================================
export ABLATION=W3
export DWT_FUSE=add
export RUN_TAG=tiny_vil_W3_add_ch32_patch8_noreg

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_W3_add_ch32_patch8_noreg.log 2>&1

#==============================================
# W3_IMPROVED_WARMUP + 小波融合对比(add vs multiply)
#==============================================
export WAVELET_WARMUP_STEPS=10000
export ABLATION=W3_IMPROVED_WARMUP
export DWT_FUSE=add
export WAVELET_SCALE_INIT=0.1

export WAVELET_FUSE_MODE=add
export RUN_TAG=tiny_vil_W3_improved_warmup_ch32_patch8_noreg
python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_W3_improved_warmup_ch32_patch8_noreg.log 2>&1

export WAVELET_FUSE_MODE=multiply
export RUN_TAG=tiny_vil_W3_improved_warmup_ch32_patch8_noreg_fuse_multiply
python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_W3_improved_warmup_ch32_patch8_noreg_fuse_multiply.log 2>&1

#==============================================
# W3_TOKENONLY
#==============================================
export ABLATION=W3_TOKENONLY
export DWT_FUSE=add

unset WAVELET_WARMUP_STEPS
unset WAVELET_FUSE_MODE
unset WAVELET_SCALE_INIT

export RUN_TAG=tiny_vil_W3_tokenonly_ch32_patch8_noreg
python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_W3_tokenonly_ch32_patch8_noreg.log 2>&1

#==============================================
# W3_RESIDUALONLY
#==============================================
export ABLATION=W3_RESIDUALONLY
export DWT_FUSE=none

unset WAVELET_WARMUP_STEPS
unset WAVELET_SCALE_INIT

export RUN_TAG=tiny_vil_W3_residualonly_ch32_patch8_noreg
python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_W3_residualonly_ch32_patch8_noreg.log 2>&1

#==============================================
# ViT-Tiny: 无正则，与有正则相同 ablations
#==============================================
export DATASET=tiny_imagenet
export DATA_ROOT=../data/tiny-imagenet-200
export MODEL_KIND=vit_tiny
export IMG_SIZE=64
export EPOCHS=300
export PER_GPU_BATCH=128
export ACCUM_STEPS=1
export AMP_DTYPE=bf16

export DIM=192
export DEPTH=12
export FEAT_CH=32
export PATCH_SIZE=8
export STRIDE=8
export AUTO_PATCH_DWT=1

export LABEL_SMOOTH=0.0
export MIXUP_PROB=0.0
export CUTMIX_ALPHA=0.0
export MIXUP_ALPHA=0.0
export SWITCH_PROB=0.0

export ABLATION=A3
export DWT_FUSE=add
export DISABLE_BRANCH=1

export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=tiny_vit_A3_ch32_patch8_noreg

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vit_A3_ch32_patch8_noreg.log 2>&1

#==============================================
# ViT + W3_POOL_ONLY
#==============================================
export ABLATION=W3_POOL_ONLY
export DWT_FUSE=none

export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=tiny_vit_W3_poolonly_ch32_patch8_noreg

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vit_W3_poolonly_ch32_patch8_noreg.log 2>&1

#==============================================
# ViT + W3 (PSWF)
#==============================================
export ABLATION=W3
export DWT_FUSE=add
export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=tiny_vit_W3_add_ch32_patch8_noreg

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vit_W3_add_ch32_patch8_noreg.log 2>&1

#==============================================
# ViT + W3_IMPROVED_WARMUP (add / multiply，与有正则对齐)
#==============================================
export ABLATION=W3_IMPROVED_WARMUP
export DWT_FUSE=add
export WAVELET_WARMUP_STEPS=10000
export WAVELET_SCALE_INIT=0.1

export WAVELET_FUSE_MODE=add
export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=tiny_vit_W3_improved_warmup_ch32_patch8_noreg
python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vit_W3_improved_warmup_ch32_patch8_noreg.log 2>&1

export WAVELET_FUSE_MODE=multiply
export RUN_TAG=tiny_vit_W3_improved_warmup_ch32_patch8_noreg_fuse_multiply
python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vit_W3_improved_warmup_ch32_patch8_noreg_fuse_multiply.log 2>&1

unset WAVELET_WARMUP_STEPS
unset WAVELET_FUSE_MODE
unset WAVELET_SCALE_INIT

#==============================================
# ViT + W3_TOKENONLY(仅 tokenization, 无 head modulation)
#==============================================
export ABLATION=W3_TOKENONLY
export DWT_FUSE=add
export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=tiny_vit_W3_tokenonly_ch32_patch8_noreg

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vit_W3_tokenonly_ch32_patch8_noreg.log 2>&1

#==============================================
# ViT + W3_RESIDUAL
#==============================================
export ABLATION=W3_RESIDUAL
export DWT_FUSE=add

export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=tiny_vit_W3_residual_ch32_patch8_noreg

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vit_W3_residual_ch32_patch8_noreg.log 2>&1
'"

#==============================================
# 第二阶段: ImageNet-1K VIL 正则 PSWF 测试(A1 / W3_IMPROVED_WARMUP / W3_POOL_ONLY)
#==============================================
tmux new -s v5ab -d "bash -lc '
set -e
source /home/omnisky/anaconda3/etc/profile.d/conda.sh
conda activate d2l
echo ENV=$CONDA_DEFAULT_ENV
which python
python -c \"import torch; print(torch.__version__, torch.__file__) \"
which torchrun || true
torchrun --version || true

export DATA_SEED=1234
export DATASET=imagenet
export DATA_ROOT=../data/imagenet_dataset
export MODEL_KIND=vil
export NUM_WORKERS=8

export IMG_SIZE=192
export EPOCHS=50
export PER_GPU_BATCH=32
export ACCUM_STEPS=1
export AMP_DTYPE=bf16
export FEAT_CH=32

export DIM=192
export DEPTH=12
export PATCH_SIZE=16
export STRIDE=16
export AUTO_PATCH_DWT=1

export BASE_LR=2e-4
export WARMUP_EPOCHS=5
export WEIGHT_DECAY=0.05
export EMA_DECAY=0.9997

# 正则化参数(与 ViT 正则实验一致)
export LABEL_SMOOTH=0.1
export MIXUP_PROB=1.0
export CUTMIX_ALPHA=1.0
export MIXUP_ALPHA=0.0
export SWITCH_PROB=1.0

export DISABLE_BRANCH=1
export OUT_DIR=./outputs_pswf_paper

#=============================================
# 3.1 VIL Baseline(A1)
#=============================================
export ABLATION=A1
export DWT_FUSE=none
export RUN_TAG=in1k192_vil_A1_ch32_reg

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/in1k192_vil_A1_ch32_reg.log 2>&1

#=============================================
# 3.3 VIL + Pool Only(W3_POOL_ONLY)
#=============================================
export ABLATION=W3_POOL_ONLY
export DWT_FUSE=none
unset WAVELET_WARMUP_STEPS
export RUN_TAG=in1k192_vil_W3_poolonly_ch32_reg

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/in1k192_vil_W3_poolonly_ch32_reg.log 2>&1

#=============================================
# 3.2 VIL + W3 (PSWF add)
#=============================================
export ABLATION=W3
export DWT_FUSE=add
unset WAVELET_WARMUP_STEPS
unset WAVELET_SCALE_INIT
export RUN_TAG=in1k192_vil_W3_add_ch32_reg

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/in1k192_vil_W3_add_ch32_reg.log 2>&1


#=============================================
# 3.2b 小波融合模式对比(add vs multiply, 需单独跑两次对比)
# 须设 WAVELET_SCALE_INIT 非 0，否则 add/multiply 等价、结果一致
#=============================================
# 加性融合
export ABLATION=W3_IMPROVED_WARMUP
export DWT_FUSE=add
export WAVELET_WARMUP_STEPS=40000
export WAVELET_SCALE_INIT=0.1
export WAVELET_FUSE_MODE=add
export RUN_TAG=in1k192_vil_W3_improved_warmup_ch32_reg_fuse_add
python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/in1k192_vil_W3_improved_warmup_ch32_reg_fuse_add.log 2>&1

# 乘性融合
export WAVELET_FUSE_MODE=multiply
export RUN_TAG=in1k192_vil_W3_improved_warmup_ch32_reg_fuse_multiply
python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/in1k192_vil_W3_improved_warmup_ch32_reg_fuse_multiply.log 2>&1

#=============================================
# 3.2c VIL + W3_TOKENONLY(只开 post-stem concat+1×1, 关掉 head residual)
#=============================================
export ABLATION=W3_TOKENONLY
export DWT_FUSE=add
unset WAVELET_WARMUP_STEPS
unset WAVELET_FUSE_MODE
unset WAVELET_SCALE_INIT
export RUN_TAG=in1k192_vil_W3_tokenonly_ch32_reg
python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/in1k192_vil_W3_tokenonly_ch32_reg.log 2>&1

#=============================================
# 3.2d VIL + W3_RESIDUALONLY(主路 pool-only, 单独开 head wavelet residual)
#=============================================
export ABLATION=W3_RESIDUALONLY
export DWT_FUSE=none
unset WAVELET_WARMUP_STEPS
unset WAVELET_SCALE_INIT
export RUN_TAG=in1k192_vil_W3_residualonly_ch32_reg
python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/in1k192_vil_W3_residualonly_ch32_reg.log 2>&1

#=============================================
# ImageNet-1K ViT 配置
#=============================================
export MODEL_KIND=vit_tiny


#=============================================
# 4.1 ViT Baseline(A3)
#=============================================
export ABLATION=A3
export DWT_FUSE=add
export FEAT_CH=32
export RUN_TAG=in1k192_vit_A3_ch32

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/in1k192_vit_A3_ch32.log 2>&1


#=============================================
# 4.3 ViT + Pool Only(W3_POOL_ONLY)
#=============================================
export ABLATION=W3_POOL_ONLY
export DWT_FUSE=none
export FEAT_CH=32
export RUN_TAG=in1k192_vit_W3_poolonly_ch32

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/in1k192_vit_W3_poolonly_ch32.log 2>&1


#=============================================
# 4.2 ViT + W3 (PSWF add)
#=============================================
export ABLATION=W3
export DWT_FUSE=add
export RUN_TAG=in1k192_vit_W3_add_ch32

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/in1k192_vit_W3_add_ch32.log 2>&1


#=============================================
# 4.2c ViT + W3_IMPROVED_WARMUP (PSWF multiply, 带 wavelet warmup)
# 与 VIL 部分对齐：使用 W3_IMPROVED_WARMUP + WAVELET_WARMUP_STEPS=40000 + WAVELET_SCALE_INIT=0.1 + multiply 融合
#=============================================
export ABLATION=W3_IMPROVED_WARMUP
export DWT_FUSE=add
export WAVELET_WARMUP_STEPS=40000
export WAVELET_SCALE_INIT=0.1
export WAVELET_FUSE_MODE=multiply
export RUN_TAG=in1k192_vit_W3_improved_warmup_ch32_fuse_multiply

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/in1k192_vit_W3_improved_warmup_ch32_fuse_multiply.log 2>&1

# 清理 wavelet 相关环境变量，避免影响后续 TOKENONLY / RESIDUAL
unset WAVELET_WARMUP_STEPS
unset WAVELET_FUSE_MODE
unset WAVELET_SCALE_INIT


#=============================================
# 4.2b ViT + W3_TOKENONLY(仅 tokenization, 无 head modulation)
#=============================================
export ABLATION=W3_TOKENONLY
export DWT_FUSE=add
export FEAT_CH=32
export RUN_TAG=in1k192_vit_W3_tokenonly_ch32

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/in1k192_vit_W3_tokenonly_ch32.log 2>&1


#=============================================
# 4.4 ViT + W3_RESIDUAL(主路 pool-only, 小波单独残差调 CLS)
#=============================================
export ABLATION=W3_RESIDUAL
export DWT_FUSE=add
export FEAT_CH=32
export RUN_TAG=in1k192_vit_W3_residual_ch32

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/in1k192_vit_W3_residual_ch32.log 2>&1

'"


#==============================================
# 第三阶段: Tiny-ImageNet-C 测试（对应加正则的 Tiny-ImageNet VIL/ViT patch8_reg 训练）
# CKPT 使用 outputs_pswf_paper 下对应 RUN_TAG 的 ema_best.pth
#==============================================
tmux new -s v5ab_tinyc_eval -d "bash -lc '
set -e
source /home/omnisky/anaconda3/etc/profile.d/conda.sh
conda activate d2l
echo ENV=$CONDA_DEFAULT_ENV
which python
python -c \"import torch; print(torch.__version__, torch.__file__) \"
which torchrun || true
torchrun --version || true

export MODE=eval_imagenetc
export DATASET=tiny_imagenet
export DATA_ROOT=/home/omnisky/Public/edward/workspace/data/tiny-imagenet-200
export IMAGENETC_ROOT=/home/omnisky/Public/edward/workspace/data/Tiny-ImageNet-C

export IMG_SIZE=64
export PER_GPU_BATCH=256
export NUM_WORKERS=8
export AMP_DTYPE=bf16
export OUT_DIR=./outputs_pswf_paper

export MODEL_KIND=vil
export DIM=192
export DEPTH=12
export FEAT_CH=32
export PATCH_SIZE=8
export STRIDE=8
export AUTO_PATCH_DWT=1
export DISABLE_BRANCH=1

#=============================================
# VIL: Baseline A1
#=============================================
export ABLATION=A1
export DWT_FUSE=none
export RUN_TAG=eval_tinyc_vil_A1_ch32_patch8_reg
export CKPT=outputs_pswf_paper/tiny_vil_A1_ch32_patch8_reg/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/eval_tinyc_vil_A1_ch32_patch8_reg.log 2>&1

#=============================================
# VIL: W3 (PSWF add)
#=============================================
export ABLATION=W3
export DWT_FUSE=add
export RUN_TAG=eval_tinyc_vil_W3_add_ch32_patch8_reg
export CKPT=outputs_pswf_paper/tiny_vil_W3_add_ch32_patch8_reg/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/eval_tinyc_vil_W3_add_ch32_patch8_reg.log 2>&1

#=============================================
# VIL: W3_POOL_ONLY
#=============================================
export ABLATION=W3_POOL_ONLY
export DWT_FUSE=none
export RUN_TAG=eval_tinyc_vil_W3_poolonly_ch32_patch8_reg
export CKPT=outputs_pswf_paper/tiny_vil_W3_poolonly_ch32_patch8_reg/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/eval_tinyc_vil_W3_poolonly_ch32_patch8_reg.log 2>&1

#=============================================
# VIL: W3_IMPROVED_WARMUP (add)
#=============================================
export ABLATION=W3_IMPROVED_WARMUP
export DWT_FUSE=add
export WAVELET_FUSE_MODE=add
export RUN_TAG=eval_tinyc_vil_W3_improved_warmup_ch32_patch8_reg
export CKPT=outputs_pswf_paper/tiny_vil_W3_improved_warmup_ch32_patch8_reg/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/eval_tinyc_vil_W3_improved_warmup_ch32_patch8_reg.log 2>&1

#=============================================
# VIL: W3_IMPROVED_WARMUP (multiply)
#=============================================
export WAVELET_FUSE_MODE=multiply
export RUN_TAG=eval_tinyc_vil_W3_improved_warmup_ch32_patch8_reg_fuse_multiply
export CKPT=outputs_pswf_paper/tiny_vil_W3_improved_warmup_ch32_patch8_reg_fuse_multiply/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/eval_tinyc_vil_W3_improved_warmup_ch32_patch8_reg_fuse_multiply.log 2>&1

#=============================================
# VIL: W3_TOKENONLY
#=============================================
unset WAVELET_WARMUP_STEPS
unset WAVELET_FUSE_MODE
export ABLATION=W3_TOKENONLY
export DWT_FUSE=add
export RUN_TAG=eval_tinyc_vil_W3_tokenonly_ch32_patch8_reg
export CKPT=outputs_pswf_paper/tiny_vil_W3_tokenonly_ch32_patch8_reg/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/eval_tinyc_vil_W3_tokenonly_ch32_patch8_reg.log 2>&1

#=============================================
# VIL: W3_RESIDUALONLY
#=============================================
export ABLATION=W3_RESIDUALONLY
export DWT_FUSE=none
export RUN_TAG=eval_tinyc_vil_W3_residualonly_ch32_patch8_reg
export CKPT=outputs_pswf_paper/tiny_vil_W3_residualonly_ch32_patch8_reg/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/eval_tinyc_vil_W3_residualonly_ch32_patch8_reg.log 2>&1

#==============================================
# Tiny-ImageNet-C ViT 测试
#==============================================
export MODE=eval_imagenetc
export DATASET=tiny_imagenet
export DATA_ROOT=/home/omnisky/Public/edward/workspace/data/tiny-imagenet-200
export IMAGENETC_ROOT=/home/omnisky/Public/edward/workspace/data/Tiny-ImageNet-C

export IMG_SIZE=64
export PER_GPU_BATCH=256
export NUM_WORKERS=8
export AMP_DTYPE=bf16
export OUT_DIR=./outputs_pswf_paper

export MODEL_KIND=vit_tiny
export DIM=192
export DEPTH=12
export FEAT_CH=32
export PATCH_SIZE=8
export STRIDE=8
export AUTO_PATCH_DWT=1
export DISABLE_BRANCH=1

#=============================================
# ViT: Baseline A3
#=============================================
export ABLATION=A3
export DWT_FUSE=add
export RUN_TAG=eval_tinyc_vit_A3_ch32_patch8_reg
export CKPT=outputs_pswf_paper/tiny_vit_A3_ch32_patch8_reg/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/eval_tinyc_vit_A3_ch32_patch8_reg.log 2>&1

#=============================================
# ViT: W3 (PSWF add)
#=============================================
export ABLATION=W3
export DWT_FUSE=add
export RUN_TAG=eval_tinyc_vit_W3_add_ch32_patch8_reg
export CKPT=outputs_pswf_paper/tiny_vit_W3_add_ch32_patch8_reg/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/eval_tinyc_vit_W3_add_ch32_patch8_reg.log 2>&1

#=============================================
# ViT: W3_IMPROVED_WARMUP (add)
#=============================================
export ABLATION=W3_IMPROVED_WARMUP
export DWT_FUSE=add
export WAVELET_FUSE_MODE=add
export RUN_TAG=eval_tinyc_vit_W3_improved_warmup_ch32_patch8_reg
export CKPT=outputs_pswf_paper/tiny_vit_W3_improved_warmup_ch32_patch8_reg/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/eval_tinyc_vit_W3_improved_warmup_ch32_patch8_reg.log 2>&1

#=============================================
# ViT: W3_IMPROVED_WARMUP (multiply)
#=============================================
export WAVELET_FUSE_MODE=multiply
export RUN_TAG=eval_tinyc_vit_W3_improved_warmup_ch32_patch8_reg_fuse_multiply
export CKPT=outputs_pswf_paper/tiny_vit_W3_improved_warmup_ch32_patch8_reg_fuse_multiply/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/eval_tinyc_vit_W3_improved_warmup_ch32_patch8_reg_fuse_multiply.log 2>&1

unset WAVELET_FUSE_MODE

#=============================================
# ViT: W3_TOKENONLY(仅 tokenization, 无 head modulation)
#=============================================
export ABLATION=W3_TOKENONLY
export DWT_FUSE=add
export RUN_TAG=eval_tinyc_vit_W3_tokenonly_ch32_patch8_reg
export CKPT=outputs_pswf_paper/tiny_vit_W3_tokenonly_ch32_patch8_reg/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/eval_tinyc_vit_W3_tokenonly_ch32_patch8_reg.log 2>&1

#=============================================
# ViT: W3_POOL_ONLY
#=============================================
export ABLATION=W3_POOL_ONLY
export DWT_FUSE=none
export RUN_TAG=eval_tinyc_vit_W3_poolonly_ch32_patch8_reg
export CKPT=outputs_pswf_paper/tiny_vit_W3_poolonly_ch32_patch8_reg/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/eval_tinyc_vit_W3_poolonly_ch32_patch8_reg.log 2>&1

#=============================================
# ViT: W3_RESIDUAL
#=============================================
export ABLATION=W3_RESIDUAL
export DWT_FUSE=add
export RUN_TAG=eval_tinyc_vit_W3_residual_ch32_patch8_reg
export CKPT=outputs_pswf_paper/tiny_vit_W3_residual_ch32_patch8_reg/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/eval_tinyc_vit_W3_residual_ch32_patch8_reg.log 2>&1

'"




#==============================================
# ch32,64,64测试
#==============================================
tmux new -s v5ab -d "bash -lc '
set -e
source /home/omnisky/anaconda3/etc/profile.d/conda.sh
conda activate d2l
echo ENV=$CONDA_DEFAULT_ENV
which python
python -c \"import torch; print(torch.__version__, torch.__file__) \"
which torchrun || true
torchrun --version || true

export DATASET=tiny_imagenet
export DATA_ROOT=../data/tiny-imagenet-200
export MODEL_KIND=vil
export IMG_SIZE=64
export EPOCHS=300
export PER_GPU_BATCH=128
export ACCUM_STEPS=1
export AMP_DTYPE=bf16

export DIM=192
export DEPTH=12
export FEAT_CH=32,64,64
export PATCH_SIZE=8
export STRIDE=8
export AUTO_PATCH_DWT=1

export ABLATION=A1
export DWT_FUSE=none
export DISABLE_BRANCH=1

export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=tiny_vil_A1_ch326464_patch8

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_A1_ch326464_patch8.log 2>&1


#==============================================
# (3) 关键消融: W3_POOL_ONLY(证明 wavelet 分支是否真的贡献)
#==============================================
export ABLATION=W3_POOL_ONLY
export DWT_FUSE=none
export FEAT_CH=32,64,64
export PATCH_SIZE=8
export STRIDE=8
export AUTO_PATCH_DWT=1
export RUN_TAG=tiny_vil_W3_poolonly_ch326464_patch8

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_W3_poolonly_ch326464_patch8.log 2>&1


#==============================================
# Plug-and-Play: ViT-Tiny(Tiny 上先跑两条就够)
# ViT baseline
#==============================================
export DATASET=tiny_imagenet
export DATA_ROOT=../data/tiny-imagenet-200
export MODEL_KIND=vit_tiny
export IMG_SIZE=64
export EPOCHS=300
export PER_GPU_BATCH=128
export ACCUM_STEPS=1
export AMP_DTYPE=bf16

export DIM=192
export DEPTH=12
export FEAT_CH=32,64,64
export PATCH_SIZE=8
export STRIDE=8
export AUTO_PATCH_DWT=1

export ABLATION=A3
export DWT_FUSE=add
export DISABLE_BRANCH=1

export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=tiny_vit_A3_ch326464_patch8

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vit_A3_ch326464_patch8.log 2>&1

#==============================================
# ViT + PSWF(只要把 ABLATION 改成 W3 即可启用 PSWF tokenizer)
#==============================================
export AUTO_PATCH_DWT=1
export DISABLE_BRANCH=1
export PATCH_SIZE=8
export STRIDE=8

export FEAT_CH=32,64,64
export ABLATION=W3
export DWT_FUSE=add

export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=tiny_vit_pswf_W3_add_ch326464_patch8

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vit_pswf_W3_add_ch326464_patch8.log 2>&1


#==============================================
# ViT + Pool Only
#==============================================
export AUTO_PATCH_DWT=1
export DISABLE_BRANCH=1
export PATCH_SIZE=8
export STRIDE=8

export FEAT_CH=32,64,64
export ABLATION=W3_POOL_ONLY
export DWT_FUSE=none

export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=tiny_vit_W3_poolonly_ch326464_patch8

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vit_W3_poolonly_ch326464_patch8.log 2>&1
'"