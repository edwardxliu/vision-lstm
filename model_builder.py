import os
from typing import Optional, List, Dict, Tuple

import torch
from torch import nn

from env_ddp import env_str, env_int, env_bool, env_float, env_list_int
from vision_lstm_util import interpolate_sincos, to_ntuple, DropPath  # noqa: F401 - re-exported for users


def _abs_mean(x: torch.Tensor) -> float:
    if x is None or x.numel() == 0:
        return 0.0
    return float(x.detach().abs().mean().item())


def _safe_ratio(num: float, den: float, eps: float = 1e-8) -> float:
    return float(num / max(float(den), eps))


# ----------------- Minimal ViT-T (for plug-and-play demos) -----------------
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, attn_drop=0.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViTTiny(nn.Module):
    """
    A minimal ViT-T/16 style model:
      dim=192, depth=12, heads=3, patch=16
    If pswf_embed is provided, it should be a module that returns a feature map (B,C,H',W') before patch embedding.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        num_classes: int,
        dim: int = 192,
        depth: int = 12,
        heads: int = 3,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        pswf_embed: Optional[nn.Module] = None,
        patch_embed: Optional[nn.Module] = None,
        pswf_gate: Optional[nn.Module] = None,
        wavelet_warmup_steps: int = 0,
        wavelet_fuse_mode: str = "add",
        wavelet_scale_init: float = 0.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.dim = dim
        self.pswf_embed = pswf_embed
        self.pswf_gate = pswf_gate
        self._wavelet_monitor_stats = {}

        if pswf_gate is not None:
            self.wavelet_scale = nn.Parameter(torch.tensor(float(wavelet_scale_init)))
            self.wavelet_warmup_steps = int(wavelet_warmup_steps) if wavelet_warmup_steps > 0 else 0
            self.wavelet_fuse_mode = str(wavelet_fuse_mode)
            self.register_buffer("_wavelet_step", torch.tensor(0, dtype=torch.long))
            self.register_buffer("_current_global_step", torch.tensor(-1, dtype=torch.long))
        else:
            self.wavelet_scale = None
            self.wavelet_warmup_steps = 0
            self.wavelet_fuse_mode = "add"

        if patch_embed is None:
            # standard patch embedding from RGB
            self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size, bias=True)
            self.pre_ch = 3
            self.ds_factor = 1
            pe_img = img_size
            pe_patch = patch_size
        else:
            # external patch embedding expects feature map already produced
            self.patch_embed = patch_embed
            self.pre_ch = None
            self.ds_factor = 2  # typical PSWF uses /2 (post-stem)
            pe_img = img_size // self.ds_factor
            pe_patch = patch_size // self.ds_factor

        grid = pe_img // pe_patch
        self.num_patches = grid * grid

        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, 1 + self.num_patches, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

        self.blocks = nn.Sequential(
            *[ViTBlock(dim, heads, mlp_ratio=4.0, attn_drop=attn_drop, drop=drop) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def _collect_embed_monitor_stats(self) -> Dict[str, float]:
        stats = {}
        if self.pswf_embed is None:
            return stats
        seen = set()
        for name, module in self.pswf_embed.named_modules():
            if module is self.pswf_embed or id(module) in seen:
                continue
            getter = getattr(module, "get_wavelet_monitor_stats", None)
            if not callable(getter):
                continue
            child_stats = getter()
            if not child_stats:
                continue
            seen.add(id(module))
            prefix = f"embed_{name.replace('.', '_')}" if name else "embed"
            for key, value in child_stats.items():
                stats[f"{prefix}_{key}"] = float(value)
        return stats

    def get_wavelet_monitor_stats(self) -> Dict[str, float]:
        return dict(self._wavelet_monitor_stats)

    def set_wavelet_global_step(self, global_step: int):
        """
        Provide a VIL-compatible interface for controlling wavelet warmup in ViT+PSWF.
        Only effective when pswf_gate is enabled and wavelet_warmup_steps > 0.
        """
        if self.pswf_embed is not None:
            seen = set()
            for _, module in self.pswf_embed.named_modules():
                if module is self.pswf_embed or id(module) in seen:
                    continue
                setter = getattr(module, "set_wavelet_global_step", None)
                if callable(setter):
                    setter(global_step)
                    seen.add(id(module))
        if self.wavelet_scale is not None and self.wavelet_warmup_steps > 0:
            self._current_global_step.fill_(global_step)

    def forward(self, x):
        gate_vec = None
        monitor_stats = {}
        if self.pswf_embed is None:
            x = self.patch_embed(x)  # (B, D, H/p, W/p)
        else:
            out = self.pswf_embed(x)
            if isinstance(out, tuple):
                feat, wav_feat = out
                if self.pswf_gate is not None:
                    gate_vec = self.pswf_gate(wav_feat)
            else:
                feat = out
                if self.pswf_gate is not None:
                    gate_vec = self.pswf_gate(feat)
            x = self.patch_embed(feat)
            monitor_stats.update(self._collect_embed_monitor_stats())

        # VitPatchEmbed returns (B, H, W, D); Conv2d returns (B, D, H, W)
        if x.ndim == 4 and x.shape[-1] == self.dim and x.shape[1] != self.dim:
            x = x.permute(0, 3, 1, 2).contiguous()
        x = x.flatten(2).transpose(1, 2)

        b = x.size(0)
        cls = self.cls.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos[:, : x.size(1)]
        x = self.blocks(x)
        x = self.norm(x)
        cls = x[:, 0]

        if gate_vec is not None and self.wavelet_scale is not None:
            cls_pre_wavelet = cls
            if self.wavelet_warmup_steps > 0 and self.training:
                if self._current_global_step.item() >= 0:
                    global_step = self._current_global_step.item()
                else:
                    global_step = self._wavelet_step.item()
                    self._wavelet_step += 1
                warmup_factor = min(1.0, max(0.0, float(global_step) / self.wavelet_warmup_steps))
            else:
                warmup_factor = 1.0

            effective_scale = self.wavelet_scale * warmup_factor
            gate_vec_t = torch.tanh(gate_vec.to(cls.dtype))
            gate_scale = effective_scale * gate_vec_t
            if self.wavelet_fuse_mode == "multiply":
                delta = cls_pre_wavelet * gate_scale
                cls = cls_pre_wavelet + delta
            else:
                delta = gate_scale
                cls = cls_pre_wavelet + delta
            cls_abs = _abs_mean(cls_pre_wavelet)
            delta_abs = _abs_mean(delta)
            monitor_stats.update(
                {
                    "head_input_abs_mean": cls_abs,
                    "head_gate_abs_mean": _abs_mean(gate_vec_t),
                    "head_delta_abs_mean": delta_abs,
                    "head_delta_over_input": _safe_ratio(delta_abs, cls_abs),
                    "head_effective_scale": float(effective_scale.detach().item()),
                }
            )
        else:
            monitor_stats.update(
                {
                    "head_input_abs_mean": _abs_mean(cls),
                    "head_gate_abs_mean": 0.0,
                    "head_delta_abs_mean": 0.0,
                    "head_delta_over_input": 0.0,
                    "head_effective_scale": 0.0,
                }
            )
        self._wavelet_monitor_stats = monitor_stats
        return self.head(cls)


# ----------------- Ablation config -----------------
def get_ablation_cfg(ablation_id: str, baseline_ablation: str = "A0") -> dict:
    """
    Map an ABLATION id to VisionLSTM2 toggles.
    You can treat W3 as your "PSWF" mainline.

    - W3_POOL_ONLY: post-stem downsample path only, no wavelet in token path, no head residual.
    - W3_TOKENONLY: post-stem pooled tokens plus wavelet residual, head wavelet residual off.
    - W3_RESIDUALONLY: main path pool-only, head wavelet residual on (DWT only for CLS modulation).
    - W3_BOTH: both token wavelet and head residual (same as W3).
    """
    ablation_id = (ablation_id or "A0").strip().upper()
    baseline_ablation = (baseline_ablation or "A0").strip().upper()

    a = {
        "A0": dict(
            use_conv_stem=True,
            use_dwt=True,
            pre_patch_dwt=False,
            disable_branch=True,
            pooling="bilateral_flatten",
            head_inject_gated=True,
        ),
        "A1": dict(
            use_conv_stem=True,
            use_dwt=False,
            pre_patch_dwt=False,
            disable_branch=True,
            pooling="bilateral_flatten",
            head_inject_gated=True,
        ),
        "A2": dict(
            use_conv_stem=False,
            use_dwt=False,
            pre_patch_dwt=True,
            disable_branch=True,
            pooling="bilateral_flatten",
            head_inject_gated=True,
        ),
        "A3": dict(
            use_conv_stem=False,
            use_dwt=False,
            pre_patch_dwt=False,
            disable_branch=True,
            pooling="bilateral_flatten",
            head_inject_gated=True,
        ),
    }

    w = {
        "W3": {
            **a["A1"],
            "post_stem_dwt": True,
            "post_stem_merge": "concat",
            "disable_branch": True,
            "wavelet_warmup_steps": 0,
            "wavelet_fuse_mode": "add",
            "head_wavelet_residual": True,
        },
        "W4": {**a["A1"], "post_stem_dwt": True, "post_stem_merge": "concat", "disable_branch": False},
        "W3_POOL_ONLY": {
            **a["A1"],
            "post_stem_dwt": True,
            "post_stem_merge": "concat",
            "disable_branch": True,
            "pool_only": True,
            "head_wavelet_residual": False,
        },
        "W3_TOKENONLY": {
            **a["A1"],
            "post_stem_dwt": True,
            "post_stem_merge": "concat",
            "disable_branch": True,
            "wavelet_warmup_steps": 0,
            "wavelet_fuse_mode": "add",
            "head_wavelet_residual": False,
        },
        "W3_RESIDUALONLY": {
            **a["A1"],
            "post_stem_dwt": True,
            "post_stem_merge": "concat",
            "disable_branch": True,
            "pool_only": True,
            "head_wavelet_residual": True,
            "wavelet_warmup_steps": 0,
            "wavelet_fuse_mode": "add",
        },
        "W3_IMPROVED_WARMUP": {
            **a["A1"],
            "post_stem_dwt": True,
            "post_stem_merge": "concat",
            "disable_branch": True,
            "wavelet_warmup_steps": 5000,
            "wavelet_fuse_mode": "add",
            "head_wavelet_residual": True,
        },
    }

    def resolve_base(base_id: str) -> dict:
        base_id = (base_id or "A0").strip().upper()
        if base_id in a:
            cfg = dict(a[base_id])
            cfg.setdefault("post_stem_dwt", False)
            cfg.setdefault("post_stem_merge", "replace")
            cfg.setdefault("pool_only", False)
            return cfg
        if base_id in w:
            cfg = dict(w[base_id])
            cfg.setdefault("head_inject_gated", True)
            cfg.setdefault("pool_only", cfg.get("pool_only", False))
            cfg.setdefault("head_wavelet_residual", True)
            return cfg
        raise KeyError(f"Unknown baseline ablation: {base_id}")

    if ablation_id in a:
        return resolve_base(ablation_id)
    if ablation_id in w:
        return resolve_base(ablation_id)

    return resolve_base(baseline_ablation)


def _infer_num_classes_for_builder() -> int:
    """
    Infer num_classes from env when building without loading dataset
    (e.g. for model_compute_lstm5_paper).
    """
    v = os.environ.get("NUM_CLASSES")
    if v is not None and str(v).strip() != "":
        try:
            return int(v)
        except ValueError:
            pass
    v = os.environ.get("SUBSET_CLASSES")
    if v is not None and str(v).strip() != "":
        try:
            k = int(v)
            if k > 0:
                return k
        except ValueError:
            pass
    ds = (os.environ.get("DATASET") or "").strip().lower()
    if ds in ("tiny_imagenet", "tiny-imagenet", "tinyimagenet", "tiny_imagenet_200"):
        return 200
    return 1000


def build_model_from_env(num_classes: Optional[int] = None, img_size: Optional[int] = None):
    """
    Build model from env vars (no DDP, no data loaders).
    Used by model_compute_lstm5_paper.py and model_analyse_lstm5_paper.py.

    Returns (model, cfg_dict); cfg_dict is a small dict with keys like
    model_kind, ablation_id, img_size, num_classes for reporting.
    """
    model_kind = env_str("MODEL_KIND", "vil").lower()
    _img_size = img_size if img_size is not None else env_int("IMG_SIZE", 192)
    _num_classes = num_classes if num_classes is not None else _infer_num_classes_for_builder()
    dim = env_int("DIM", 192)
    depth = env_int("DEPTH", 12)
    feat_ch = env_list_int("FEAT_CH", default=[32, 64, 64]) or [32, 64, 64]
    patch_size = env_int("PATCH_SIZE", 16)
    stride = env_int("STRIDE", patch_size)
    auto_patch_dwt = env_bool("AUTO_PATCH_DWT", True)
    ablation_id = env_str("ABLATION", "W3")
    dwt_fuse = env_str("DWT_FUSE", "add")
    disable_branch_env = env_bool("DISABLE_BRANCH", True)
    drop_path = env_float("DROP_PATH", 0.0)
    drop_path_decay = env_bool("DROP_PATH_DECAY", False)
    legacy_norm = env_bool("LEGACY_NORM", False)
    conv_kind = env_str("CONV_KIND", "2d")
    conv_kernel = env_int("CONV_KERNEL", 3)
    proj_bias = env_bool("PROJ_BIAS", True)
    norm_bias = env_bool("NORM_BIAS", True)
    token_wavelet_scale_init = env_float("TOKEN_WAVELET_SCALE_INIT", 0.1)
    token_wavelet_inner_scale_init = env_float("TOKEN_WAVELET_INNER_SCALE_INIT", token_wavelet_scale_init)
    token_wavelet_outer_scale_init = env_float("TOKEN_WAVELET_OUTER_SCALE_INIT", token_wavelet_scale_init)
    token_wavelet_shrink = env_float("TOKEN_WAVELET_SHRINK", 0.02)
    token_wavelet_hf_only = env_bool("TOKEN_WAVELET_HF_ONLY", True)
    token_wavelet_per_channel = env_bool("TOKEN_WAVELET_PER_CHANNEL", True)
    token_wavelet_hidden_ch = max(0, env_int("TOKEN_WAVELET_HIDDEN_CH", 0))
    token_wavelet_side_ch = max(0, env_int("TOKEN_WAVELET_SIDE_CH", 0))
    token_wavelet_side_mode = env_str("TOKEN_WAVELET_SIDE_MODE", "concat").strip().lower()
    token_wavelet_side_beta_init = env_float("TOKEN_WAVELET_SIDE_BETA_INIT", 0.1)
    token_wavelet_outer_gate = env_bool("TOKEN_WAVELET_OUTER_GATE", False)
    token_wavelet_split_bands = env_bool("TOKEN_WAVELET_SPLIT_BANDS", False)
    wavelet_input_image = env_bool("WAVELET_INPUT_IMAGE", False)

    #cfg = dict(model_kind=model_kind, ablation_id=ablation_id, img_size=_img_size, num_classes=_num_classes)

    cfg = dict(
        model_kind=model_kind,
        ablation_id=ablation_id,
        img_size=_img_size,
        num_classes=_num_classes,
        token_wavelet_scale_init=token_wavelet_scale_init,
        token_wavelet_inner_scale_init=token_wavelet_inner_scale_init,
        token_wavelet_outer_scale_init=token_wavelet_outer_scale_init,
        token_wavelet_shrink=token_wavelet_shrink,
        token_wavelet_hf_only=token_wavelet_hf_only,
        token_wavelet_per_channel=token_wavelet_per_channel,
        token_wavelet_hidden_ch=token_wavelet_hidden_ch,
        token_wavelet_side_ch=token_wavelet_side_ch,
        token_wavelet_side_mode=token_wavelet_side_mode,
        token_wavelet_side_beta_init=token_wavelet_side_beta_init,
        token_wavelet_outer_gate=token_wavelet_outer_gate,
        token_wavelet_split_bands=token_wavelet_split_bands,
        wavelet_input_image=wavelet_input_image,
    )

    if model_kind == "vil":
        from model_vil import VisionLSTM2

        abl_cfg = get_ablation_cfg(ablation_id)
        if "DISABLE_BRANCH" in os.environ:
            abl_cfg["disable_branch"] = bool(disable_branch_env)
        pool_only = bool(abl_cfg.get("pool_only", False))
        dwt_fuse_eff = "none" if pool_only else dwt_fuse
        model = VisionLSTM2(
            dim=dim,
            depth=depth,
            input_shape=(3, _img_size, _img_size),
            output_shape=(_num_classes,),
            mode="classifier",
            pooling=abl_cfg.get("pooling", "bilateral_flatten"),
            drop_path_rate=drop_path,
            drop_path_decay=drop_path_decay,
            legacy_norm=legacy_norm,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel,
            proj_bias=proj_bias,
            norm_bias=norm_bias,
            patch_size=patch_size,
            stride=stride,
            feature_extractor_channels=feat_ch,
            use_conv_stem=abl_cfg.get("use_conv_stem", True),
            use_dwt=abl_cfg.get("use_dwt", False),
            pre_patch_dwt=abl_cfg.get("pre_patch_dwt", False),
            post_stem_dwt=abl_cfg.get("post_stem_dwt", False),
            post_stem_merge=abl_cfg.get("post_stem_merge", "replace"),
            disable_branch=abl_cfg.get("disable_branch", True),
            auto_patch_dwt=auto_patch_dwt,
            dwt_fuse=dwt_fuse_eff,
            wavelet_warmup_steps=int(os.environ["WAVELET_WARMUP_STEPS"])
            if os.environ.get("WAVELET_WARMUP_STEPS")
            else abl_cfg.get("wavelet_warmup_steps", 0),
            wavelet_fuse_mode=os.environ.get("WAVELET_FUSE_MODE") or abl_cfg.get("wavelet_fuse_mode", "multiply"),
            head_wavelet_residual=abl_cfg.get("head_wavelet_residual", True),
            wavelet_scale_init=env_float("WAVELET_SCALE_INIT", 0.0),
            
            token_wavelet_scale_init = token_wavelet_scale_init,
            token_wavelet_inner_scale_init = token_wavelet_inner_scale_init,
            token_wavelet_outer_scale_init = token_wavelet_outer_scale_init,
            token_wavelet_shrink = token_wavelet_shrink,
            token_wavelet_hf_only = token_wavelet_hf_only,
            token_wavelet_per_channel = token_wavelet_per_channel,
            token_wavelet_hidden_channels = token_wavelet_hidden_ch,
            token_wavelet_side_channels = token_wavelet_side_ch,
            token_wavelet_side_mode = token_wavelet_side_mode,
            token_wavelet_side_beta_init = token_wavelet_side_beta_init,
            token_wavelet_outer_gate = token_wavelet_outer_gate,
            token_wavelet_split_bands = token_wavelet_split_bands,
            wavelet_input_image = wavelet_input_image,
        )
        return model, cfg

    if model_kind == "vit_tiny":
        ab_u = (ablation_id or "").strip().upper()
        use_pswf = ab_u.startswith("W3") or ab_u.startswith("W4")
        pool_only = ("POOL_ONLY" in ab_u) or ("POOLONLY" in ab_u)
        if use_pswf:
            if token_wavelet_side_mode == "patch":
                raise NotImplementedError(
                    "TOKEN_WAVELET_SIDE_MODE=patch is only supported for MODEL_KIND=vil."
                )
            from model_vil import (
                FeatureExtractor,
                PostStemWaveletMerge,
                VitPatchEmbed,
                WaveletGlobalGate,
                StemWithWaveletResidual,
                StemWithImageWavelet,
                DWTPreprocessor,
            )

            vit_wavelet_warmup_steps = 0
            vit_wavelet_fuse_mode = "add"
            vit_wavelet_scale_init = env_float("WAVELET_SCALE_INIT", 0.0)
            if ab_u == "W3_IMPROVED_WARMUP":
                vit_wavelet_warmup_steps = (
                    int(os.environ["WAVELET_WARMUP_STEPS"]) if os.environ.get("WAVELET_WARMUP_STEPS") else 5000
                )
                vit_wavelet_fuse_mode = os.environ.get("WAVELET_FUSE_MODE") or "multiply"
            stem = FeatureExtractor(input_channels=3, conv_channels=feat_ch, use_dwt=False, dwt_fuse="none")
            token_only = "TOKENONLY" in ab_u or ab_u == "W3_TOKENONLY"
            use_residual = ("RESIDUAL" in ab_u) or (ab_u == "W3_RESIDUAL")
            if use_residual:
                post_pool_only = PostStemWaveletMerge(
                    channels=stem.final_channels,
                    dwt_fuse="none",
                    merge="concat",
                    token_wavelet_scale_init=token_wavelet_scale_init,
                    token_wavelet_inner_scale_init=token_wavelet_inner_scale_init,
                    token_wavelet_outer_scale_init=token_wavelet_outer_scale_init,
                    token_wavelet_shrink=token_wavelet_shrink,
                    token_wavelet_hf_only=token_wavelet_hf_only,
                    token_wavelet_per_channel=token_wavelet_per_channel,
                    token_wavelet_warmup_steps=vit_wavelet_warmup_steps,
                    token_wavelet_hidden_channels=token_wavelet_hidden_ch,
                    token_wavelet_side_channels=token_wavelet_side_ch,
                    token_wavelet_side_mode=token_wavelet_side_mode,
                    token_wavelet_outer_gate=token_wavelet_outer_gate,
                    token_wavelet_split_bands=token_wavelet_split_bands,
                )

                dwt_module = DWTPreprocessor(
                    channels=stem.final_channels,
                    dwt_fuse="add",
                    token_wavelet_scale_init=token_wavelet_inner_scale_init,
                    token_wavelet_shrink=token_wavelet_shrink,
                    token_wavelet_hf_only=token_wavelet_hf_only,
                    token_wavelet_per_channel=token_wavelet_per_channel,
                )

                pswf_embed = StemWithWaveletResidual(stem, post_pool_only, dwt_module)
                main_ch = post_pool_only.out_channels
                pswf_gate = None if token_only else WaveletGlobalGate(in_channels=dwt_module.out_channels, dim=dim)
            else:
                dwt_fuse_eff = "none" if pool_only else dwt_fuse
                vit_image_input_ch = 3 if (wavelet_input_image and not pool_only) else 0
                post = PostStemWaveletMerge(
                    channels=stem.final_channels,
                    dwt_fuse=dwt_fuse_eff,
                    merge="concat",
                    token_wavelet_scale_init=token_wavelet_scale_init,
                    token_wavelet_inner_scale_init=token_wavelet_inner_scale_init,
                    token_wavelet_outer_scale_init=token_wavelet_outer_scale_init,
                    token_wavelet_shrink=token_wavelet_shrink,
                    token_wavelet_hf_only=token_wavelet_hf_only,
                    token_wavelet_per_channel=token_wavelet_per_channel,
                    token_wavelet_warmup_steps=vit_wavelet_warmup_steps,
                    token_wavelet_hidden_channels=token_wavelet_hidden_ch,
                    token_wavelet_side_channels=token_wavelet_side_ch,
                    token_wavelet_side_mode=token_wavelet_side_mode,
                    token_wavelet_outer_gate=token_wavelet_outer_gate,
                    token_wavelet_split_bands=token_wavelet_split_bands,
                    token_wavelet_image_input_channels=vit_image_input_ch,
                )
                if vit_image_input_ch > 0:
                    pswf_embed = StemWithImageWavelet(stem, post)
                else:
                    pswf_embed = nn.Sequential(stem, post)
                main_ch = post.out_channels
                pswf_gate = None if (token_only or pool_only) else WaveletGlobalGate(in_channels=main_ch, dim=dim)
                
            pe_res = (_img_size // 2, _img_size // 2)
            if bool(auto_patch_dwt):
                patch_eff = patch_size // 2
                stride_eff = stride // 2
            else:
                patch_eff = patch_size
                stride_eff = stride
            patch_embed = VitPatchEmbed(
                dim=dim,
                num_channels=main_ch,
                resolution=pe_res,
                patch_size=(patch_eff, patch_eff),
                stride=(stride_eff, stride_eff),
                init_weights="xavier_uniform",
            )
            model = ViTTiny(
                img_size=_img_size,
                patch_size=patch_size,
                num_classes=_num_classes,
                dim=dim,
                depth=depth,
                heads=max(1, dim // 64),
                pswf_embed=pswf_embed,
                patch_embed=patch_embed,
                pswf_gate=pswf_gate,
                wavelet_warmup_steps=vit_wavelet_warmup_steps,
                wavelet_fuse_mode=vit_wavelet_fuse_mode,
                wavelet_scale_init=vit_wavelet_scale_init,
            )
        else:
            model = ViTTiny(
                img_size=_img_size,
                patch_size=patch_size,
                num_classes=_num_classes,
                dim=dim,
                depth=depth,
                heads=max(1, dim // 64),
            )
        return model, cfg

    if model_kind == "mambavision":
        raise RuntimeError(
            "MODEL_KIND=mambavision is a stub. Provide a builder in-code or import your local implementation."
        )
    raise ValueError("MODEL_KIND must be vil | vit_tiny | mambavision")
