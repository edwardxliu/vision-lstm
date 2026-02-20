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


PATCH_SIZE=8 试验1：COL_EVERY=4
tmux new -s train -d \
"PRETRAIN_CKPT=lstm6_half_patchsize8_col.pth SUBSET_CLASSES=500 PATCH_SIZE=8 STRIDE=8 EPOCHS=80 PYRAMID=half MERGE_KIND=patch USE_DWT=0 BRANCH_ALPHA_MAX=0 STAGE_DIMS=256,384 STAGE_DEPTHS=0,8 COL_EVERY=4 MIXER_EVERY=9999 \
torchrun --nproc_per_node=8 lstm6_stage1_pretrain_192_sample.py \
 > /home/omnisky/lstm6_half_patchsize8_col.log 2>&1"

PATCH_SIZE=8 试验2：COL_EVERY=4 + MIXER_EVERY=4
tmux new -s train -d \
"PRETRAIN_CKPT=lstm6_half_patchsize8_col_mixer.pth SUBSET_CLASSES=500 PATCH_SIZE=8 STRIDE=8 EPOCHS=80 PYRAMID=half MERGE_KIND=patch USE_DWT=0 BRANCH_ALPHA_MAX=0 STAGE_DIMS=256,384 STAGE_DEPTHS=0,8 COL_EVERY=4 MIXER_EVERY=4 \
torchrun --nproc_per_node=8 lstm6_stage1_pretrain_192_sample.py \
 > /home/omnisky/lstm6_half_patchsize8_col_mixer.log 2>&1"


PATCH_SIZE=8 试验3：COL_EVERY=4 + MIXER_EVERY=4 + BRANCH
tmux new -s train -d \
"PRETRAIN_CKPT=lstm6_half_patchsize8_col_mixer_branch.pth SUBSET_CLASSES=500 PATCH_SIZE=8 STRIDE=8 EPOCHS=80 PYRAMID=half MERGE_KIND=patch USE_DWT=0 BRANCH_ALPHA_MAX=1e-3 BRANCH_START=25 BRANCH_RAMP=60 STAGE_DIMS=256,384 STAGE_DEPTHS=0,8 COL_EVERY=4 MIXER_EVERY=4 \
torchrun --nproc_per_node=8 lstm6_stage1_pretrain_192_sample.py \
 > /home/omnisky/lstm6_half_patchsize8_col_mixer_branch.log 2>&1"

试验A0：
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
export DWT_FUSE=add
export DISABLE_BRANCH=1

export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=smoke_tiny_A1

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/smoke_tiny_A1.log 2>&1
'"



#==============================================
# Patch Size: 16 Baseline：A1（没有 PSWF）测试
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
export FEAT_CH=32
export PATCH_SIZE=16
export STRIDE=16
export AUTO_PATCH_DWT=1

export ABLATION=A1
export DWT_FUSE=add
export DISABLE_BRANCH=1

export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=tiny_vil_A1_ch32

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_A1_ch32.log 2>&1


#==============================================
# (2) PSWF 主线：W3 + add（先用轻量版本）
#==============================================
export ABLATION=W3
export DWT_FUSE=add
export FEAT_CH=32
export PATCH_SIZE=16
export STRIDE=16
export AUTO_PATCH_DWT=1
export RUN_TAG=tiny_vil_W3_add_ch32

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_W3_add_ch32.log 2>&1

#==============================================
# (3) 关键消融：W3_POOL_ONLY（证明 wavelet 分支是否真的贡献）
#==============================================
export ABLATION=W3_POOL_ONLY
export DWT_FUSE=none
export FEAT_CH=32
export PATCH_SIZE=16
export STRIDE=16
export AUTO_PATCH_DWT=1
export RUN_TAG=tiny_vil_W3_poolonly_ch32

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_W3_poolonly_ch32.log 2>&1


#==============================================
# add vs gated（第二优先级，但建议跑）
#==============================================
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
export PATCH_SIZE=16
export STRIDE=16
export AUTO_PATCH_DWT=1

export ABLATION=W3
export DWT_FUSE=gated
export DISABLE_BRANCH=1

export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=tiny_vil_W3_gated_ch32

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_W3_gated_ch32.log 2>&1


#==============================================
# “重/轻 PSWF 的 compute knob”
#==============================================
export ABLATION=W3
export PATCH_SIZE=16
export STRIDE=16
export DWT_FUSE=add
export FEAT_CH=32,64,64
export AUTO_PATCH_DWT=1
export RUN_TAG=tiny_vil_W3_add_ch326464

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_W3_add_ch326464.log 2>&1





#==============================================
# Plug-and-Play：ViT-Tiny（Tiny 上先跑两条就够）
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
export PATCH_SIZE=16
export STRIDE=16
export AUTO_PATCH_DWT=1

export ABLATION=A3
export DWT_FUSE=add
export DISABLE_BRANCH=1

export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=tiny_vit_A3_ch32

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vit_A3_ch32.log 2>&1

#==============================================
# ViT + PSWF（只要把 ABLATION 改成 W3 即可启用 PSWF tokenizer）
#==============================================
export AUTO_PATCH_DWT=1
export DISABLE_BRANCH=1
export PATCH_SIZE=16
export STRIDE=16

export ABLATION=W3
export DWT_FUSE=add

export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=tiny_vit_pswf_W3_add

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vit_pswf_W3_add.log 2>&1


#==============================================
# ViT + Pool Only
#==============================================
export AUTO_PATCH_DWT=1
export DISABLE_BRANCH=1
export PATCH_SIZE=16
export STRIDE=16

export ABLATION=W3_POOL_ONLY
export DWT_FUSE=none

export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=tiny_vit_W3_poolonly_ch32

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vit_W3_poolonly_ch32.log 2>&1
'"


#==============================================
# Patch Size: 8 Baseline：A1（没有 PSWF）测试
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
export FEAT_CH=32
export PATCH_SIZE=8
export STRIDE=8
export AUTO_PATCH_DWT=1

export ABLATION=A1
export DWT_FUSE=add
export DISABLE_BRANCH=1

export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=tiny_vil_A1_ch32_patch8

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_A1_ch32_patch8.log 2>&1


#==============================================
# (2) PSWF 主线：W3 + add（先用轻量版本）
#==============================================
export ABLATION=W3
export DWT_FUSE=add
export FEAT_CH=32
export PATCH_SIZE=8
export STRIDE=8
export AUTO_PATCH_DWT=1
export RUN_TAG=tiny_vil_W3_add_ch32_patch8

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_W3_add_ch32_patch8.log 2>&1

#==============================================
# (3) 关键消融：W3_POOL_ONLY（证明 wavelet 分支是否真的贡献）
#==============================================
export ABLATION=W3_POOL_ONLY
export DWT_FUSE=none
export FEAT_CH=32
export PATCH_SIZE=8
export STRIDE=8
export AUTO_PATCH_DWT=1
export RUN_TAG=tiny_vil_W3_poolonly_ch32_patch8

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_W3_poolonly_ch32_patch8.log 2>&1


#==============================================
# add vs gated（第二优先级，但建议跑）
#==============================================
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

export ABLATION=W3
export DWT_FUSE=gated
export DISABLE_BRANCH=1

export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=tiny_vil_W3_gated_ch32_patch8

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_W3_gated_ch32_patch8.log 2>&1


#==============================================
# “重/轻 PSWF 的 compute knob”
#==============================================
export ABLATION=W3
export PATCH_SIZE=8
export STRIDE=8
export DWT_FUSE=add
export FEAT_CH=32,64,64
export AUTO_PATCH_DWT=1
export RUN_TAG=tiny_vil_W3_add_ch32_patch86464

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_W3_add_ch32_patch86464.log 2>&1





#==============================================
# Plug-and-Play：ViT-Tiny（Tiny 上先跑两条就够）
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

export ABLATION=A3
export DWT_FUSE=add
export DISABLE_BRANCH=1

export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=tiny_vit_A3_ch32_patch8

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vit_A3_ch32_patch8.log 2>&1

#==============================================
# ViT + PSWF（只要把 ABLATION 改成 W3 即可启用 PSWF tokenizer）
#==============================================
export AUTO_PATCH_DWT=1
export DISABLE_BRANCH=1
export PATCH_SIZE=8
export STRIDE=8

export ABLATION=W3
export DWT_FUSE=add

export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=tiny_vit_W3_add_ch32_patch8

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vit_W3_add_ch32_patch8.log 2>&1


#==============================================
# ViT + Pool Only
#==============================================
export AUTO_PATCH_DWT=1
export DISABLE_BRANCH=1
export PATCH_SIZE=8
export STRIDE=8

export ABLATION=W3_POOL_ONLY
export DWT_FUSE=none

export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=tiny_vit_W3_poolonly_ch32_patch8

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vit_W3_poolonly_ch32_patch8.log 2>&1
'"




#==============================================
# 正则化测试： Baseline：A1（没有 PSWF）测试
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

export DATA_SEED=4321
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

export ABLATION=A1
export DWT_FUSE=add
export DISABLE_BRANCH=1

export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=tiny_vil_A1_ch32_patch8_reg

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_A1_ch32_patch8_reg.log 2>&1


#==============================================
# (2) PSWF 主线：W3 + add（先用轻量版本）
#==============================================
export ABLATION=W3
export DWT_FUSE=add
export FEAT_CH=32
export PATCH_SIZE=8
export STRIDE=8
export AUTO_PATCH_DWT=1
export LABEL_SMOOTH=0.1
export MIXUP_PROB=1.0
export CUTMIX_ALPHA=1.0
export MIXUP_ALPHA=0.0
export SWITCH_PROB=1.0
export RUN_TAG=tiny_vil_W3_add_ch32_patch8_reg

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_W3_add_ch32_patch8_reg.log 2>&1

#==============================================
# (3) 关键消融：W3_POOL_ONLY（证明 wavelet 分支是否真的贡献）
#==============================================
export ABLATION=W3_POOL_ONLY
export DWT_FUSE=none
export FEAT_CH=32
export PATCH_SIZE=8
export STRIDE=8
export AUTO_PATCH_DWT=1
export LABEL_SMOOTH=0.1
export MIXUP_PROB=1.0
export CUTMIX_ALPHA=1.0
export MIXUP_ALPHA=0.0
export SWITCH_PROB=1.0
export RUN_TAG=tiny_vil_W3_poolonly_ch32_patch8_reg

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_W3_poolonly_ch32_patch8_reg.log 2>&1

#==============================================
# (4) 改进版：W3_IMPROVED（scale=0初始化 + 乘性融合，无warmup）
#==============================================
export ABLATION=W3_IMPROVED
export DWT_FUSE=add
export FEAT_CH=32
export PATCH_SIZE=8
export STRIDE=8
export AUTO_PATCH_DWT=1
export LABEL_SMOOTH=0.1
export MIXUP_PROB=1.0
export CUTMIX_ALPHA=1.0
export MIXUP_ALPHA=0.0
export SWITCH_PROB=1.0
export RUN_TAG=tiny_vil_W3_improved_ch32_patch8_reg

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_W3_improved_ch32_patch8_reg.log 2>&1

#==============================================
# (5) 改进版：W3_IMPROVED_WARMUP（scale=0初始化 + 乘性融合 + warmup前5000步）
#==============================================
export ABLATION=W3_IMPROVED_WARMUP
export DWT_FUSE=add
export FEAT_CH=32
export PATCH_SIZE=8
export STRIDE=8
export AUTO_PATCH_DWT=1
export LABEL_SMOOTH=0.1
export MIXUP_PROB=1.0
export CUTMIX_ALPHA=1.0
export MIXUP_ALPHA=0.0
export SWITCH_PROB=1.0
export RUN_TAG=tiny_vil_W3_improved_warmup_ch32_patch8_reg

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_W3_improved_warmup_ch32_patch8_reg.log 2>&1


#==============================================
# Plug-and-Play：ViT-Tiny（Tiny 上先跑两条就够）
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
# ViT + PSWF（只要把 ABLATION 改成 W3 即可启用 PSWF tokenizer）
#==============================================
export AUTO_PATCH_DWT=1
export DISABLE_BRANCH=1
export PATCH_SIZE=8
export STRIDE=8

export ABLATION=W3
export DWT_FUSE=add
export LABEL_SMOOTH=0.1
export MIXUP_PROB=1.0
export CUTMIX_ALPHA=1.0
export MIXUP_ALPHA=0.0
export SWITCH_PROB=1.0
export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=tiny_vit_W3_add_ch32_patch8_reg

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vit_W3_add_ch32_patch8_reg.log 2>&1


#==============================================
# ViT + Pool Only
#==============================================
export AUTO_PATCH_DWT=1
export DISABLE_BRANCH=1
export PATCH_SIZE=8
export STRIDE=8
export LABEL_SMOOTH=0.1
export MIXUP_PROB=1.0
export CUTMIX_ALPHA=1.0
export MIXUP_ALPHA=0.0
export SWITCH_PROB=1.0
export ABLATION=W3_POOL_ONLY
export DWT_FUSE=none

export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=tiny_vit_W3_poolonly_ch32_patch8_reg

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vit_W3_poolonly_ch32_patch8_reg.log 2>&1

#==============================================
# ViT + W3_RESIDUAL（主路 pool-only，小波单独残差调 CLS）
#==============================================
export AUTO_PATCH_DWT=1
export DISABLE_BRANCH=1
export PATCH_SIZE=8
export STRIDE=8
export LABEL_SMOOTH=0.1
export MIXUP_PROB=1.0
export CUTMIX_ALPHA=1.0
export MIXUP_ALPHA=0.0
export SWITCH_PROB=1.0
export ABLATION=W3_RESIDUAL
export DWT_FUSE=add

export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=tiny_vit_W3_residual_ch32_patch8_reg

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vit_W3_residual_ch32_patch8_reg.log 2>&1
'"



#==============================================
# 第二阶段：ImageNet-1K 上做“主表最少点位”
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

export DATASET=imagenet
export DATA_ROOT=../data/imagenet_dataset
export MODEL_KIND=vil

export IMG_SIZE=192
export EPOCHS=100
export PER_GPU_BATCH=32
export ACCUM_STEPS=1
export AMP_DTYPE=bf16

export DIM=192
export DEPTH=12
export PATCH_SIZE=16
export STRIDE=16
export AUTO_PATCH_DWT=1

export BASE_LR=2e-4
export WARMUP_EPOCHS=5
export WEIGHT_DECAY=0.05
export EMA_DECAY=0.9997

# 为了让“时间对比”更干净，建议先关掉 mixup/ls
export LABEL_SMOOTH=0.1
export MIXUP_PROB=1.0
export CUTMIX_ALPHA=1.0
export MIXUP_ALPHA=0.0
export SWITCH_PROB=1.0

export DISABLE_BRANCH=1
export OUT_DIR=./outputs_pswf_paper

#=============================================
# 3.1 Baseline（A1）
#=============================================
export ABLATION=A1
export DWT_FUSE=add
export FEAT_CH=32
export RUN_TAG=in1k192_vil_A1_ch32

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/in1k192_vil_A1_ch32.log 2>&1

#=============================================
# 3.2 轻 PSWF（W3）
#=============================================
export ABLATION=W3
export DWT_FUSE=add
export FEAT_CH=32
export RUN_TAG=in1k192_vil_W3_add_ch32

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/in1k192_vil_W3_add_ch32.log 2>&1

#=============================================
# 3.3 pool-only（W3_POOL_ONLY）
#=============================================
export ABLATION=W3_POOL_ONLY
export DWT_FUSE=none
export FEAT_CH=32
export RUN_TAG=in1k192_vil_W3_poolonly_ch32
python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/in1k192_vil_W3_poolonly_ch32.log 2>&1

#=============================================
# （可选但建议）重 PSWF（展示训练加速 knob）
#=============================================
export ABLATION=W3
export DWT_FUSE=add
export FEAT_CH=32,64,64
export RUN_TAG=in1k192_vil_W3_add_ch326464
python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/in1k192_vil_W3_add_ch326464.log 2>&1

#=============================================
# ImageNet-1K ViT 配置
#=============================================
export MODEL_KIND=vit_tiny

#=============================================
# 4.1 ViT Baseline（A3）
#=============================================
export ABLATION=A3
export DWT_FUSE=add
export FEAT_CH=32
export RUN_TAG=in1k192_vit_A3_ch32

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/in1k192_vit_A3_ch32.log 2>&1

#=============================================
# 4.2 ViT + Pool Only（W3_POOL_ONLY）
#=============================================
export ABLATION=W3_POOL_ONLY
export DWT_FUSE=none
export FEAT_CH=32
export RUN_TAG=in1k192_vit_W3_poolonly_ch32

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/in1k192_vit_W3_poolonly_ch32.log 2>&1

#=============================================
# 4.3 ViT + W3_RESIDUAL（主路 pool-only，小波单独残差调 CLS）
#=============================================
export ABLATION=W3_RESIDUAL
export DWT_FUSE=add
export FEAT_CH=32
export RUN_TAG=in1k192_vit_W3_residual_ch32

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/in1k192_vit_W3_residual_ch32.log 2>&1

'"


#==============================================
# 第三阶段：Tiny-ImageNet-C VIL测试
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
export PATCH_SIZE=16
export STRIDE=16
export AUTO_PATCH_DWT=1
export DISABLE_BRANCH=1


#=============================================
# 评 Baseline：Tiny-ViL A1 ch32
#=============================================
export ABLATION=A1
export DWT_FUSE=add
export RUN_TAG=eval_tinyc_vil_A1_ch32
export CKPT=outputs_pswf_paper/tiny_vil_A1_ch32/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/eval_tinyc_vil_A1_ch32.log 2>&1



#=============================================
# 评 PSWF：Tiny-ViL W3 add ch32
#=============================================
export ABLATION=W3
export DWT_FUSE=add
export FEAT_CH=32
export RUN_TAG=eval_tinyc_vil_W3_add_ch32
export CKPT=outputs_pswf_paper/tiny_vil_W3_add_ch32/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/eval_tinyc_vil_W3_add_ch32.log 2>&1


#=============================================
# 评 Pool-only：Tiny-ViL W3_poolonly ch32
#=============================================
export ABLATION=W3_POOL_ONLY
export DWT_FUSE=none
export FEAT_CH=32
export RUN_TAG=eval_tinyc_vil_W3_poolonly_ch32
export CKPT=outputs_pswf_paper/tiny_vil_W3_poolonly_ch32/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/eval_tinyc_vil_W3_poolonly_ch32.log 2>&1

'"


#==============================================
# 第三阶段：Tiny-ImageNet-C VIT测试
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
export PATCH_SIZE=16
export STRIDE=16
export AUTO_PATCH_DWT=1
export DISABLE_BRANCH=1


#=============================================
# 评 Baseline：Tiny-ViT A3 ch32
#=============================================
export ABLATION=A3
export DWT_FUSE=add
export RUN_TAG=eval_tinyc_vit_A3_ch32
export CKPT=outputs_pswf_paper/tiny_vit_A3_ch32/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/eval_tinyc_vit_A3_ch32.log 2>&1


#=============================================
# 评 PSWF：Tiny-ViL W3 add ch32
#=============================================
export ABLATION=W3
export DWT_FUSE=add
export FEAT_CH=32
export RUN_TAG=eval_tinyc_vit_W3_add_ch32
export CKPT=outputs_pswf_paper/tiny_vit_pswf_W3_add/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/eval_tinyc_vit_W3_add_ch32.log 2>&1


#=============================================
# 评 Pool-only：Tiny-ViL W3_poolonly ch32
#=============================================
export ABLATION=W3_POOL_ONLY
export DWT_FUSE=none
export FEAT_CH=32
export RUN_TAG=eval_tinyc_vit_W3_poolonly_ch32
export CKPT=outputs_pswf_paper/tiny_vit_W3_poolonly_ch32/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/eval_tinyc_vit_W3_poolonly_ch32.log 2>&1

'"

#==============================================
# Plug-and-Play：ViT-Tiny LL 训练
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
export MODEL_KIND=vit_tiny
export IMG_SIZE=64
export EPOCHS=300
export PER_GPU_BATCH=128
export ACCUM_STEPS=1
export AMP_DTYPE=bf16

export DIM=192
export DEPTH=12
export FEAT_CH=32
export PATCH_SIZE=16
export STRIDE=16

export AUTO_PATCH_DWT=1
export DISABLE_BRANCH=1

export ABLATION=W3
export DWT_FUSE=LL

export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=tiny_vit_W3_LL

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vit_W3_LL.log 2>&1
'"

#==============================================
# Tiny-ImageNet-C VIT LL测试
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
export PATCH_SIZE=16
export STRIDE=16
export AUTO_PATCH_DWT=1
export DISABLE_BRANCH=1

export ABLATION=W3
export DWT_FUSE=LL
export FEAT_CH=32
export RUN_TAG=eval_tinyc_vit_W3_LL_ch32
export CKPT=outputs_pswf_paper/tiny_vit_W3_LL/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/eval_tinyc_vit_W3_LL_ch32.log 2>&1
'"


#==============================================
# Tiny-ImageNet-C VIL LL测试
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
export FEAT_CH=32
export PATCH_SIZE=16
export STRIDE=16
export AUTO_PATCH_DWT=1

#==============================================
# (2) PSWF 主线：W3 + add（先用轻量版本）
#==============================================
export ABLATION=W3
export DWT_FUSE=LL
export FEAT_CH=32
export RUN_TAG=tiny_vil_W3_LL_ch32

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_W3_LL_ch32.log 2>&1
'"


#==============================================
# Tiny-ImageNet-C VIL LL测试
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
export PATCH_SIZE=16
export STRIDE=16
export AUTO_PATCH_DWT=1
export DISABLE_BRANCH=1

#=============================================
# 评 PSWF：Tiny-ViL W3 add ch32
#=============================================
export ABLATION=W3
export DWT_FUSE=LL
export FEAT_CH=32
export RUN_TAG=eval_tinyc_vil_W3_LL_ch32
export CKPT=outputs_pswf_paper/tiny_vil_W3_add_ch32/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/eval_tinyc_vil_W3_LL_ch32.log 2>&1
'"


#### tiny image net 上训练 参数修改，patch_size=8; AUTO_PATCH_DWT=1; 另外还要补充label_smooth=0.1 + cutmix的试验。


#==============================================
# 第三阶段：PATCHSIZE 8 Tiny-ImageNet-C VIL测试
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
# 评 Baseline：Tiny-ViL A1 ch32
#=============================================
export ABLATION=A1
export DWT_FUSE=add
export RUN_TAG=eval_tinyc_vil_A1_ch32_patch8_reg
export CKPT=test/tiny_vil_A1_ch32_patch8_reg/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/eval_tinyc_vil_A1_ch32_patch8_reg.log 2>&1



#=============================================
# 评 PSWF：Tiny-ViL W3 add ch32
#=============================================
export ABLATION=W3
export DWT_FUSE=add
export FEAT_CH=32
export RUN_TAG=eval_tinyc_vil_W3_add_ch32_patch8_reg
export CKPT=test/tiny_vil_W3_add_ch32_patch8_reg/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/eval_tinyc_vil_W3_add_ch32_patch8_reg.log 2>&1


#=============================================
# 评 Pool-only：Tiny-ViL W3_poolonly ch32
#=============================================
export ABLATION=W3_POOL_ONLY
export DWT_FUSE=none
export FEAT_CH=32
export RUN_TAG=eval_tinyc_vil_W3_poolonly_ch322_patch8_reg
export CKPT=test/tiny_vil_W3_poolonly_ch32_patch8_reg/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/eval_tinyc_vil_W3_poolonly_ch322_patch8_reg.log 2>&1


#==============================================
# 第三阶段：Tiny-ImageNet-C VIT测试
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
# 评 Baseline：Tiny-ViT A3 ch32
#=============================================
export ABLATION=A3
export DWT_FUSE=add
export RUN_TAG=eval_tinyc_vit_A3_ch32_patch8_reg
export CKPT=test/tiny_vit_A3_ch32_patch8_reg/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/eval_tinyc_vit_A3_ch32_patch8_reg.log 2>&1


#=============================================
# 评 PSWF：Tiny-ViL W3 add ch32
#=============================================
export ABLATION=W3
export DWT_FUSE=add
export FEAT_CH=32
export RUN_TAG=eval_tinyc_vit_W3_add_ch32_patch8_reg
export CKPT=test/tiny_vit_W3_add_ch32_patch8_reg/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/eval_tinyc_vit_W3_add_ch32_patch8_reg.log 2>&1


#=============================================
# 评 Pool-only：Tiny-ViL W3_poolonly ch32
#=============================================
export ABLATION=W3_POOL_ONLY
export DWT_FUSE=none
export FEAT_CH=32
export RUN_TAG=eval_tinyc_vit_W3_poolonly_ch32_patch8_reg
export CKPT=test/tiny_vit_W3_poolonly_ch32_patch8_reg/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/eval_tinyc_vit_W3_poolonly_ch32_patch8_reg.log 2>&1

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
export DWT_FUSE=add
export DISABLE_BRANCH=1

export OUT_DIR=./outputs_pswf_paper
export RUN_TAG=tiny_vil_A1_ch326464_patch8

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_A1_ch326464_patch8.log 2>&1


#==============================================
# (3) 关键消融：W3_POOL_ONLY（证明 wavelet 分支是否真的贡献）
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
# Plug-and-Play：ViT-Tiny（Tiny 上先跑两条就够）
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
# ViT + PSWF（只要把 ABLATION 改成 W3 即可启用 PSWF tokenizer）
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