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
export MODE=eval

export IMG_SIZE=192
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
export MIXUP_ALPHA=0.0
export CUTMIX_ALPHA=0.0
export MIXUP_PROB=0.0
export LABEL_SMOOTH=0.0

export DISABLE_BRANCH=1

#=============================================
# 3.1 Baseline（A1）
#=============================================
export ABLATION=A1
export DWT_FUSE=add
export FEAT_CH=32
export RUN_TAG=in1k192_vil_A1_ch32_val
export CKPT=outputs_pswf_paper/in1k192_vil_A1_ch32/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/in1k192_vil_A1_ch32_val.log 2>&1

#=============================================
# 3.2 轻 PSWF（W3）
#=============================================
export ABLATION=W3
export DWT_FUSE=add
export FEAT_CH=32
export RUN_TAG=in1k192_vil_W3_add_ch32_val
export CKPT=outputs_pswf_paper/in1k192_vil_W3_add_ch32/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/in1k192_vil_W3_add_ch32_val.log 2>&1

#=============================================
# 3.3 pool-only（W3_POOL_ONLY）
#=============================================
export ABLATION=W3_POOL_ONLY
export DWT_FUSE=none
export FEAT_CH=32
export RUN_TAG=in1k192_vil_W3_poolonly_ch32_val
export CKPT=outputs_pswf_paper/in1k192_vil_W3_poolonly_ch32/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/in1k192_vil_W3_poolonly_ch32_val.log 2>&1

#=============================================
# （可选但建议）重 PSWF（展示训练加速 knob）
#=============================================
export ABLATION=W3
export DWT_FUSE=add
export FEAT_CH=32,64,64
export RUN_TAG=in1k192_vil_W3_add_ch326464_val
export CKPT=outputs_pswf_paper/in1k192_vil_W3_add_ch326464/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/in1k192_vil_W3_add_ch326464_val.log 2>&1
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

export DATASET=tiny_imagenet
export DATA_ROOT=../data/tiny-imagenet-200
export MODEL_KIND=vil
export MODE=eval
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

export RUN_TAG=tiny_vil_A1_ch32_patch8_reg_val
export CKPT=outputs_pswf_paper/tiny_vil_A1_ch32_patch8_reg/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_A1_ch32_patch8_reg_val.log 2>&1


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
export RUN_TAG=tiny_vil_W3_add_ch32_patch8_reg_val
export CKPT=outputs_pswf_paper/tiny_vil_W3_add_ch32_patch8_reg/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_W3_add_ch32_patch8_reg_val.log 2>&1

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
export RUN_TAG=tiny_vil_W3_poolonly_ch32_patch8_reg_val
export CKPT=outputs_pswf_paper/tiny_vil_W3_poolonly_ch32_patch8_reg/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_W3_poolonly_ch32_patch8_reg_val.log 2>&1


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

export LABEL_SMOOTH=0.1
export MIXUP_PROB=1.0
export CUTMIX_ALPHA=1.0
export MIXUP_ALPHA=0.0
export SWITCH_PROB=1.0

export ABLATION=W3
export DWT_FUSE=gated
export DISABLE_BRANCH=1

export RUN_TAG=tiny_vil_W3_gated_ch32_patch8_reg_val
export CKPT=outputs_pswf_paper/tiny_vil_W3_gated_ch32_patch8_reg/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_W3_gated_ch32_patch8_reg_val.log 2>&1


#==============================================
# “重/轻 PSWF 的 compute knob”
#==============================================
export ABLATION=W3
export PATCH_SIZE=8
export STRIDE=8
export DWT_FUSE=add
export FEAT_CH=32,64,64
export AUTO_PATCH_DWT=1
export LABEL_SMOOTH=0.1
export MIXUP_PROB=1.0
export CUTMIX_ALPHA=1.0
export MIXUP_ALPHA=0.0
export SWITCH_PROB=1.0
export RUN_TAG=tiny_vil_W3_add_ch32_patch86464_reg_val
export CKPT=outputs_pswf_paper/tiny_vil_W3_add_ch32_patch86464_reg/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vil_W3_add_ch32_patch86464_reg_val.log 2>&1

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

export RUN_TAG=tiny_vit_A3_ch32_patch8_reg_val
export CKPT=outputs_pswf_paper/tiny_vit_A3_ch32_patch8_reg/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vit_A3_ch32_patch8_reg_val.log 2>&1

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
export RUN_TAG=tiny_vit_W3_add_ch32_patch8_reg_val
export CKPT=outputs_pswf_paper/tiny_vit_W3_add_ch32_patch8_reg/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vit_W3_add_ch32_patch8_reg_val.log 2>&1


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

export RUN_TAG=tiny_vit_W3_poolonly_ch32_patch8_reg_val
export CKPT=outputs_pswf_paper/tiny_vit_W3_poolonly_ch32_patch8_reg/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/tiny_vit_W3_poolonly_ch32_patch8_reg_val.log 2>&1
'"