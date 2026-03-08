

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
export OUT_DIR=./output_pswf_continue

#=============================================
# 3.1 VIL Baseline(A1)
#=============================================
export ABLATION=A1
export DWT_FUSE=none
export RUN_TAG=in1k192_vil_A1_ch32_reg_continue
export RESUME_CKPT=outputs_pswf_paper/in1k192_vil_A1_ch32_reg_continue/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/in1k192_vil_A1_ch32_reg_continue.log 2>&1

#=============================================
# 3.3 VIL + Pool Only(W3_POOL_ONLY)
#=============================================
export ABLATION=W3_POOL_ONLY
export DWT_FUSE=none
unset WAVELET_WARMUP_STEPS
export RUN_TAG=in1k192_vil_W3_poolonly_ch32_reg_continue
export RESUME_CKPT=outputs_pswf_paper/in1k192_vil_W3_poolonly_ch32_reg/ema_best.pth

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/in1k192_vil_W3_poolonly_ch32_reg_continue.log 2>&1


#=============================================
# 3.2b 小波融合模式对比(add vs multiply, 需单独跑两次对比)
# 须设 WAVELET_SCALE_INIT 非 0，否则 add/multiply 等价、结果一致
#=============================================
# 加性融合
export ABLATION=W3_IMPROVED_WARMUP
export DWT_FUSE=add
export WAVELET_WARMUP_STEPS=0
export WAVELET_SCALE_INIT=0.1
export WAVELET_FUSE_MODE=multiply

export RESUME_CKPT=outputs_pswf_paper/in1k192_vil_W3_improved_warmup_ch32_reg_fuse_multiply/ema_best.pth
export RUN_TAG=in1k192_vil_W3_improved_warmup_ch32_reg_fuse_multiply_continue

python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/in1k192_vil_W3_improved_warmup_ch32_reg_fuse_multiply_continue.log 2>&1

#=============================================
# 3.2d VIL + W3_RESIDUALONLY(主路 pool-only, 单独开 head wavelet residual)
#=============================================
export ABLATION=W3_RESIDUALONLY
export DWT_FUSE=none
unset WAVELET_WARMUP_STEPS
unset WAVELET_SCALE_INIT

export RESUME_CKPT=outputs_pswf_paper/in1k192_vil_W3_residualonly_ch32_reg/ema_best.pth
export RUN_TAG=in1k192_vil_W3_residualonly_ch32_reg_continue
python -m torch.distributed.run --nproc_per_node=8 lstm5_stage1_pretrain_192_sample_ablation_paper.py \
    > /home/omnisky/in1k192_vil_W3_residualonly_ch32_reg_continue.log 2>&1

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
