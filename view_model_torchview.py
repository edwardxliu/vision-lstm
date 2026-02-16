import os
import torch
from torchview import draw_graph
from vision_lstm6 import VisionLSTM2


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
def _pick_tensor_output(out):
    """把模型输出规整成一个 Tensor（兼容 tuple/list/dict）"""
    if torch.is_tensor(out):
        return out
    if isinstance(out, (list, tuple)):
        for x in out:
            t = _pick_tensor_output(x)
            if t is not None:
                return t
    if isinstance(out, dict):
        for _, x in out.items():
            t = _pick_tensor_output(x)
            if t is not None:
                return t
    return None

@torch.no_grad()
def draw_model_graph(model, input_shape=(1, 3, 224, 224), device="cpu",
                     out_dir="model_graphs", name="model", depth=6):
    os.makedirs(out_dir, exist_ok=True)
    model = model.to(device).eval()

    # torchview 内部会做一次 forward；有些自定义 forward 可能需要显式 dummy forward 预热
    dummy = torch.randn(*input_shape, device=device)
    out = model(dummy)
    t = _pick_tensor_output(out)
    if t is None:
        raise RuntimeError("模型输出里没找到 Tensor，无法绘图；请检查 forward 返回值。")

    g = draw_graph(
        model,
        input_size=input_shape,          # 或 input_data=dummy
        device=device,
        depth=depth,                     # 控制展开层级，模型太大就调小
        expand_nested=True,              # 展开子模块
        graph_name=name
    )

    # 保存 png / pdf
    g.visual_graph.render(os.path.join(out_dir, name), format="png", cleanup=True)
    g.visual_graph.render(os.path.join(out_dir, name), format="pdf", cleanup=True)
    return g

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
    pyramid=pyramid,
    mixer_every=mixer_every,
    pair_fusion=pair_fusion,
    col_every=col_every,
    gamma_init=gamma_init,

    # ===== merge ablation =====
    merge_kind='patch',

    # stage_dims=[384,384],
    # stage_depths=[2,6]
)

# 示例：如果你已经有 m5/m6
draw_model_graph(model, input_shape=(1,3,192,192), device="cuda", name="vision_lstm7")
# draw_model_graph(m6, input_shape=(1,3,192,192), device="cuda", name="vision_lstm6")
