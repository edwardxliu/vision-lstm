@echo off

call E:\miniconda3\Scripts\activate.bat vixlpy39
python --version
where python

cd /d E:\workspace\vision-lstm

goto :COMMENT_END
set PYRAMID=full
set MERGE_KIND=conv_patch
python -u lstm6_train_tinyimagenet_merge.py > train_full_conv_patch.log 2>&1

set PYRAMID=full
set MERGE_KIND=patch
python -u lstm6_train_tinyimagenet_merge.py > train_full_patch.log 2>&1

set PYRAMID=full
set MERGE_KIND=conv
python -u lstm6_train_tinyimagenet_merge.py > train_full_conv.log 2>&1


set PYRAMID=half
set MERGE_KIND=conv_patch
python -u lstm6_train_tinyimagenet_merge.py > train_half_conv_patch.log 2>&1

set PYRAMID=half
set MERGE_KIND=patch
python -u lstm6_train_tinyimagenet_merge.py > train_half_patch.log 2>&1

set PYRAMID=half
set MERGE_KIND=conv
python -u lstm6_train_tinyimagenet_merge.py > train_half_conv.log 2>&1

set PYRAMID=half2
set MERGE_KIND=conv_patch
python -u lstm6_train_tinyimagenet_merge.py > train_half2_conv_patch.log 2>&1

set PYRAMID=half2
set MERGE_KIND=patch
python -u lstm6_train_tinyimagenet_merge.py > train_half2_patch.log 2>&1

set PYRAMID=half2
set MERGE_KIND=conv
python -u lstm6_train_tinyimagenet_merge.py > train_half2_conv.log 2>&1


set PYRAMID=half
set MERGE_KIND=patch
set BRANCH_ALPHA_MAX=0
set USE_DWT=0
set COL_EVERY=0
set MIXER_EVERY=9999
set STAGE_DIMS=384,384
set STAGE_DEPTHS=2,6
python -u lstm6_train_tinyimagenet_merge.py > train_half_only.log 2>&1

set PYRAMID=none
set MERGE_KIND=patch
set BRANCH_ALPHA_MAX=0
set USE_DWT=0
set COL_EVERY=0
set MIXER_EVERY=9999
python -u lstm6_train_tinyimagenet_merge.py > train_none_pyramid.log 2>&1

set PYRAMID=half
set MERGE_KIND=patch
set BRANCH_ALPHA_MAX=0
set USE_DWT=0
set COL_EVERY=0
set MIXER_EVERY=9999
set STAGE_DIMS=256,384
set STAGE_DEPTHS=0,8
set PATCH_SIZE=8
SET STRID=8
python -u lstm6_train_tinyimagenet_merge.py > train_half_patchsize8.log 2>&1

set USE_DWT=1
set POOLING=attn
set OUT_CKPT=lstm5_attn_dwt_best.pth
python -u lstm5_train_tinyimagenet.py > lstm5_attn_dwt.log 2>&1
:COMMENT_END

set USE_DWT=1
set POOLING=global
set OUT_CKPT=lstm5_global_dwt_best.pth
python -u lstm5_train_tinyimagenet.py > lstm5_global_dwt.log 2>&1
