import os
import torch
from vision_lstm5 import VisionLSTM2

NUM_CLASSES = 150

pyramid      = os.environ.get("PYRAMID", "half")  # none/half/half2/full
pair_fusion  = os.environ.get("PAIR_FUSION", "parallel_gated")
col_every    = int(os.environ.get("COL_EVERY", "0"))
gamma_init   = float(os.environ.get("GAMMA_INIT", "1e-4"))
mixer_every  = int(os.environ.get("MIXER_EVERY", "9999"))
patch_size = int(os.environ.get("PATCH_SIZE", "8"))
stride     = int(os.environ.get("STRIDE", str(patch_size)))
# merge_kind = os.environ.get("MERGE_KIND", "patch").lower()
# merge_kinds = env_list("MERGE_KINDS", default=None)
# if merge_kinds is not None:
#     merge_kinds = [s.lower() for s in merge_kinds]

# last_merge_kind = os.environ.get("LAST_MERGE_KIND", "patch").strip().lower()
# if last_merge_kind in ("", "none", "null"):
#     last_merge_kind = None
@torch.no_grad()
def export_onnx(model, input_shape=(1,3,224,224), device="cpu",
                out_path="model.onnx", opset=17):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    model = model.to(device).eval()
    x = torch.randn(*input_shape, device=device)

    torch.onnx.export(
        model, x, out_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "B"}, "output": {0: "B"}},
        do_constant_folding=True
    )
    return out_path

model = VisionLSTM2(
    dim=384,
    input_shape=(3, 192, 192),
    patch_size=8,
    depth=8,
    output_shape=(NUM_CLASSES,),
    mode="classifier",
    pooling="global",
    stride=8,
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
    # pyramid=pyramid,
    # mixer_every=mixer_every,
    # pair_fusion=pair_fusion,
    # col_every=col_every,
    # gamma_init=gamma_init,

    # ===== merge ablation =====
    # merge_kind='patch',

    # stage_dims=[384,384],
    # stage_depths=[2,6]
)

# 用法：
export_onnx(model, input_shape=(1,3,192,192), device="cpu", out_path="onnx/vision_lstm5.onnx")
# export_onnx(m6, input_shape=(1,3,192,192), device="cpu", out_path="onnx/vision_lstm6.onnx")

# 然后用 Netron 打开（桌面/浏览器都行）
# pip install netron
# netron onnx/vision_lstm5.onnx
