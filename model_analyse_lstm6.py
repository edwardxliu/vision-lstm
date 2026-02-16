import torch
# from vision_lstm5 import VisionLSTM2
from vision_lstm6 import VisionLSTM2
import os

# ----------------------------------
# 修改成你的 best_model.pth 路径
# ----------------------------------
#ckpt_path = "base_model.pth"
ckpt_path = "pth_copy/lstm6_half_only.pth"
#ckpt_path = "vil_stage2_224.pth"


# ----------------------------------
# 创建与你训练时一致的模型结构
# ----------------------------------

NUM_CLASSES = 150

pyramid      = os.environ.get("PYRAMID", "half")  # none/half/half2/full
pair_fusion  = os.environ.get("PAIR_FUSION", "parallel_gated")
col_every    = int(os.environ.get("COL_EVERY", "0"))
gamma_init   = float(os.environ.get("GAMMA_INIT", "1e-4"))
mixer_every  = int(os.environ.get("MIXER_EVERY", "9999"))
patch_size = int(os.environ.get("PATCH_SIZE", "8"))
stride     = int(os.environ.get("STRIDE", str(patch_size)))
merge_kind = os.environ.get("MERGE_KIND", "patch").lower()
merge_kinds = env_list("MERGE_KINDS", default=None)
if merge_kinds is not None:
    merge_kinds = [s.lower() for s in merge_kinds]

last_merge_kind = os.environ.get("LAST_MERGE_KIND", "patch").strip().lower()
if last_merge_kind in ("", "none", "null"):
    last_merge_kind = None

model = VisionLSTM2(
    dim=384,
    input_shape=(3, 192, 192),
    patch_size=16,
    depth=8,
    output_shape=(NUM_CLASSES,),
    mode="classifier",
    pooling="global",
    stride=16,
    legacy_norm=True,
    drop_path_rate=0.1,
    drop_path_decay=True,
    conv_kind="2d",
    conv_kernel_size=3,
    proj_bias=True,
    norm_bias=True,
    feature_extractor_channels=[32, 64],
    use_dwt=False,

    # ===== 金字塔/结构相关 =====
    pyramid=pyramid,
    mixer_every=mixer_every,
    pair_fusion=pair_fusion,
    col_every=col_every,
    gamma_init=gamma_init,

    # ===== merge ablation =====
    merge_kind=merge_kind,
    merge_kinds=patch,
    last_merge_kind=last_merge_kind,

    stage_dims=[384,384],
    stage_depths=[2,6]
)

# ----------------------------------
# 加载权重（注意 strict=False）
# ----------------------------------
state = torch.load(ckpt_path, map_location="cpu")
missing, unexpected = model.load_state_dict(state, strict=False)

print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))

# ----------------------------------
# 统计参数量
# ----------------------------------
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nTotal parameters: {total_params:,d}")
print(f"Trainable parameters: {trainable_params:,d}")
print(f"Total params (in Millions): {total_params/1e6:.3f} M")

