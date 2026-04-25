import warnings
import math
from enum import Enum
import einops
import torch
import torch.nn.functional as F
from torch import nn
from vision_lstm_util import interpolate_sincos, to_ntuple, DropPath

class SequenceConv2d(nn.Conv2d):
    """
    Applies 2D convolution to a sequence of flattened 2D patches.

    Args:
        *args: Arguments for nn.Conv2d.
        seqlens (tuple of int, optional): Spatial dimensions (height, width) of the input patches. 
                                           If None, assumes the input is square.
        **kwargs: Keyword arguments for nn.Conv2d.
    """
    def __init__(self, *args, seqlens=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.seqlens = seqlens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, channels).

        Returns:
            torch.Tensor: Output tensor after applying the convolution, reshaped back to (batch_size, seq_len, out_channels).
        """
        assert x.ndim == 3, "Input tensor must have 3 dimensions: (batch_size, seq_len, channels)."

        if self.seqlens is None:
            # Assumes square input
            h = math.sqrt(x.size(1))
            if not h.is_integer():
                raise ValueError(f"Input sequence length {x.size(1)} is not a perfect square.")
            h = int(h)
        else:
            if len(self.seqlens) != 2:
                raise ValueError("seqlens should be a tuple of length 2 (height, width).")
            h = self.seqlens[0]
        
        # Reshape input tensor from (batch_size, seq_len, channels) to (batch_size, channels, height, width)
        x = einops.rearrange(x, "b (h w) d -> b d h w", h=h)
        
        # Apply convolution
        x = super().forward(x)
        
        # Reshape output tensor back to (batch_size, seq_len, out_channels)
        x = einops.rearrange(x, "b d h w -> b (h w) d")
        
        return x

class SequenceTraversal(Enum):
    ROWWISE_FROM_TOP_LEFT = "rowwise_from_top_left"
    ROWWISE_FROM_BOT_RIGHT = "rowwise_from_bot_right"


def bias_linspace_init_(param: torch.Tensor, start: float = 3.4, end: float = 6.0) -> torch.Tensor:
    """Linearly spaced bias init across dimensions."""
    assert param.dim() == 1, f"param must be 1-dimensional (typically a bias), got {param.dim()}"
    n_dims = param.shape[0]
    init_vals = torch.linspace(start, end, n_dims)
    with torch.no_grad():
        param.copy_(init_vals)
    return param


def small_init_(param: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2019), using a normal distribution.
    Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py.
    """
    std = math.sqrt(2 / (5 * dim))
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


def wang_init_(param: torch.Tensor, dim: int, num_blocks: int):
    """ Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py. """
    std = 2 / num_blocks / math.sqrt(dim)
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


def _abs_mean(x: torch.Tensor) -> float:
    if x is None or x.numel() == 0:
        return 0.0
    return float(x.detach().abs().mean().item())


def _safe_ratio(num: float, den: float, eps: float = 1e-8) -> float:
    return float(num / max(float(den), eps))


def parallel_stabilized_simple(
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        igate_preact: torch.Tensor,
        fgate_preact: torch.Tensor,
        lower_triangular_matrix: torch.Tensor = None,
        stabilize_rowwise: bool = True,
        eps: float = 1e-6,
) -> torch.Tensor:
    """
    This is the mLSTM cell in parallel form.
    This version is stabilized. We control the range of exp() arguments by
    ensuring that they are always smaller than 0.0 by subtracting the maximum.

    Args:
        :param queries: (torch.Tensor) (B, NH, S, DH)
        :param keys: (torch.Tensor) (B, NH, S, DH)
        :param values: (torch.Tensor) (B, NH, S, DH)
        :param igate_preact: (torch.Tensor) (B, NH, S, 1)
        :param fgate_preact: (torch.Tensor) (B, NH, S, 1)
        :param lower_triangular_matrix: (torch.Tensor) (S,S). Defaults to None.
        :param stabilize_rowwise: (bool) Wether to stabilize the combination matrix C rowwise (take maximum per row).
            Alternative: Subtract the maximum over all rows. Defaults to True.
        :param eps: (float) small constant to avoid division by 0. Defaults to 1e-6.

    Returns:
        torch.Tensor: (B, NH, S, DH), h_tilde_state
    """

    orig_dtype = queries.dtype
    queries = queries.float()
    keys = keys.float()
    values = values.float()
    igate_preact = igate_preact.float()
    fgate_preact = fgate_preact.float()
    eps = 1e-6

    B, NH, S, DH = queries.shape
    _dtype, _device = queries.dtype, queries.device

    # forget gate matrix
    log_fgates = torch.nn.functional.logsigmoid(fgate_preact)  # (B, NH, S, 1)
    if lower_triangular_matrix is None or S < lower_triangular_matrix.size(-1):
        ltr = torch.tril(torch.ones((S, S), dtype=torch.bool, device=_device))
    else:
        ltr = lower_triangular_matrix
    assert ltr.dtype == torch.bool, f"lower_triangular_matrix must be of dtype bool, got {ltr.dtype}"

    log_fgates_cumsum = torch.cat(
        [
            torch.zeros((B, NH, 1, 1), dtype=_dtype, device=_device),
            torch.cumsum(log_fgates, dim=-2),
        ],
        dim=-2,
    )  # (B, NH, S+1, 1)
    # for each batch/head this is a matrix of shape (S+1, S+1) containing the cumsum of the log forget gate values
    # in the second dimension (colum dimension). Each row has the same is a copy of the first row.
    # First entry of each row is zero.
    rep_log_fgates_cumsum = log_fgates_cumsum.repeat(1, 1, 1, S + 1)  # (B, NH, S+1, S+1)
    # Now in each row cut off / subtract the forgetgate values of the later timesteps
    # where col j > row i
    _log_fg_matrix = rep_log_fgates_cumsum - rep_log_fgates_cumsum.transpose(-2, -1)  # (B, NH, S+1, S+1)
    # Causal masking & selection of the correct submatrix, such that forgetgate at timestep t is not applied
    # to the input at timestep t
    log_fg_matrix = torch.where(ltr, _log_fg_matrix[:, :, 1:, 1:], -float("inf"))  # (B, NH, S, S)

    # gate decay matrix D (combination of forget gate and input gate)
    log_D_matrix = log_fg_matrix + igate_preact.transpose(-2, -1)  # (B, NH, S, S)
    # D matrix stabilization
    if stabilize_rowwise:
        max_log_D, _ = torch.max(log_D_matrix, dim=-1, keepdim=True)  # (B, NH, S, 1)
    else:
        max_log_D = torch.max(log_D_matrix.view(B, NH, -1), dim=-1, keepdim=True)[0].unsqueeze(-1)
        # (B, NH, 1, 1)
    log_D_matrix_stabilized = log_D_matrix - max_log_D  # (B, NH, S, S)
    D_matrix = torch.exp(log_D_matrix_stabilized)  # (B, NH, S, S)

    keys_scaled = keys / math.sqrt(DH)

    # combination matrix C
    qk_matrix = queries @ keys_scaled.transpose(-2, -1)  # (B, NH, S, S)
    C_matrix = qk_matrix * D_matrix  # (B, NH, S, S)
    normalizer = torch.maximum(C_matrix.sum(dim=-1, keepdim=True).abs(), torch.exp(-max_log_D))  # (B, NH, S, 1)
    # (B, NH, S, S)
    C_matrix_normalized = C_matrix / (normalizer + eps)

    # retrieved values
    h_tilde_state = C_matrix_normalized @ values  # (B, NH, S, DH)

    h_tilde_state = h_tilde_state.to(orig_dtype)

    return h_tilde_state


class LinearHeadwiseExpand(nn.Module):
    """
    This is a structured projection layer that projects the input to a higher dimension.
    It only allows integer up-projection factors, i.e. the output dimension is a multiple of the input dimension.
    """

    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads

        dim_per_head = dim // num_heads
        self.weight = nn.Parameter(torch.empty(num_heads, dim_per_head, dim_per_head))
        if bias:
            self.bias = nn.Parameter(torch.empty(dim))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight.data, mean=0.0, std=math.sqrt(2 / 5 / self.weight.shape[-1]))
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "... (nh d) -> ... nh d", nh=self.num_heads)
        x = einops.einsum(
            x,
            self.weight,
            "... nh d, nh out_d d -> ... nh out_d",
        )
        x = einops.rearrange(x, "... nh out_d -> ... (nh out_d)")
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self):
        return (
            f"dim={self.dim}, "
            f"num_heads={self.num_heads}, "
            f"bias={self.bias is not None}, "
        )


class CausalConv1d(nn.Module):
    """
    Implements causal depthwise convolution for time series data.

    Args:
        dim (int): Number of features in the input tensor (i.e., the number of input channels).
        kernel_size (int): Size of the convolution kernel. Default is 4.
        bias (bool): Whether to use a bias term in the convolution. Default is True.
        channel_mixing (bool): Whether to mix features across channels. If True, uses groups=1. If False, uses groups=dim.
    """

    def __init__(self, dim, kernel_size=4, bias=True):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.bias = bias
        # Padding ensures the output is the same length as the input
        self.pad = kernel_size - 1

        # Depthwise convolution with padding to ensure causality
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=self.pad,
            groups=dim,
            bias=bias
        )
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize convolution parameters."""
        self.conv.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the causal convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, time_steps, feature_dim).

        Returns:
            torch.Tensor: Output tensor of the same shape as input (batch_size, time_steps, feature_dim).
        """
        # Ensure input is of shape (batch_size, feature_dim, time_steps)
        x = einops.rearrange(x, 'b t f -> b f t')
        
        # Apply causal depthwise convolution
        x = self.conv(x)
        
        # Remove padding to ensure output length matches input length
        x = x[:, :, :-self.pad]
        
        # Rearrange back to (batch_size, time_steps, feature_dim)
        x = einops.rearrange(x, 'b f t -> b t f')
        
        return x

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False. """

    def __init__(
            self,
            ndim: int = -1,
            weight: bool = True,
            bias: bool = False,
            eps: float = 1e-5,
            residual_weight: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(ndim)) if weight else None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps
        self.residual_weight = residual_weight
        self.ndim = ndim
        self.reset_parameters()

    @property
    def weight_proxy(self) -> torch.Tensor:
        if self.weight is None:
            return None
        if self.residual_weight:
            return 1.0 + self.weight
        else:
            return self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            normalized_shape=(self.ndim,),
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )

    def reset_parameters(self):
        if self.weight_proxy is not None:
            if self.residual_weight:
                nn.init.zeros_(self.weight)
            else:
                nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MultiHeadLayerNorm(LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, "Input must be 4D tensor (B, NH, S, DH)"
        B, NH, S, DH = x.shape

        gn_in_1 = x.transpose(1, 2)  # (B, S, NH, DH)
        gn_in_2 = gn_in_1.reshape(B * S, NH * DH)  # (B * S, NH * DH)
        out = F.group_norm(
            gn_in_2,
            num_groups=NH,
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )  # .to(x.dtype)
        # (B * S), (NH * DH) -> (B, S, NH, DH) -> (B, NH, S, DH)
        out = out.view(B, S, NH, DH).transpose(1, 2)
        return out


class MatrixLSTMCell(nn.Module):
    def __init__(self, dim, num_heads, norm_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.igate = nn.Linear(3 * dim, num_heads)
        self.fgate = nn.Linear(3 * dim, num_heads)
        self.outnorm = MultiHeadLayerNorm(ndim=dim, weight=True, bias=norm_bias)
        self.causal_mask_cache = {}
        self.reset_parameters()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, S, _ = q.shape  # (B, S, H)

        if_gate_input = torch.cat([q, k, v], dim=-1)
        q = q.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
        k = k.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
        v = v.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)

        q = q.transpose(1, 2)  # (B, NH, S, DH)
        k = k.transpose(1, 2)  # (B, NH, S, DH)
        v = v.transpose(1, 2)  # (B, NH, S, DH)

        # compute input and forget gate pre-activations
        igate_preact = self.igate(if_gate_input)  # (B, S, NH)
        igate_preact = igate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)
        fgate_preact = self.fgate(if_gate_input)  # (B, S, NH)
        fgate_preact = fgate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)#

        # cache causal mask to avoid memory allocation in every iteration
        key = (S, str(q.device))
        if key in self.causal_mask_cache:
            causal_mask = self.causal_mask_cache[key]
        else:
            causal_mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=q.device))
            self.causal_mask_cache[key] = causal_mask

        h_state = parallel_stabilized_simple(
            queries=q,
            keys=k,
            values=v,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
            lower_triangular_matrix=causal_mask,
        )  # (B, NH, S, DH)

        h_state_norm = self.outnorm(h_state)  # (B, NH, S, DH)
        h_state_norm = h_state_norm.transpose(1, 2).reshape(B, S, -1)  # (B, NH, S, DH) -> (B, S, NH, DH) -> (B, S, H)

        return h_state_norm

    def reset_parameters(self):
        self.outnorm.reset_parameters()
        # forget gate initialization
        torch.nn.init.zeros_(self.fgate.weight)
        bias_linspace_init_(self.fgate.bias, start=3.0, end=6.0)
        # input gate initialization
        torch.nn.init.zeros_(self.igate.weight)
        torch.nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)


class ViLLayer(nn.Module):
    def __init__(
            self,
            dim,
            direction,
            expansion=2,
            qkv_block_size=4,
            proj_bias=True,
            norm_bias=True,
            conv_bias=True,
            conv_kernel_size=4,
            conv_kind="2d",
            seqlens=None,
    ):
        super().__init__()
        assert dim % qkv_block_size == 0
        self.dim = dim
        self.direction = direction
        self.expansion = expansion
        self.qkv_block_size = qkv_block_size
        self.proj_bias = proj_bias
        self.conv_bias = conv_bias
        self.conv_kernel_size = conv_kernel_size
        self.conv_kind = conv_kind

        inner_dim = expansion * dim
        num_heads = inner_dim // qkv_block_size
        self.proj_up = nn.Linear(
            in_features=dim,
            out_features=2 * inner_dim,
            bias=proj_bias,
        )
        self.q_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )
        self.k_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )
        self.v_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )

        if conv_kind == "causal1d":
            self.conv = CausalConv1d(
                dim=inner_dim,
                kernel_size=conv_kernel_size,
                bias=conv_bias,
            )
        elif conv_kind == "2d":
            assert conv_kernel_size % 2 == 1, \
                f"same output shape as input shape is required -> even kernel sizes not supported"
            self.conv = SequenceConv2d(
                in_channels=inner_dim,
                out_channels=inner_dim,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2,
                groups=inner_dim,
                bias=conv_bias,
                seqlens=seqlens,
            )
        else:
            raise NotImplementedError
        self.mlstm_cell = MatrixLSTMCell(
            dim=inner_dim,
            num_heads=qkv_block_size,
            norm_bias=norm_bias,
        )
        self.learnable_skip = nn.Parameter(torch.ones(inner_dim))

        self.proj_down = nn.Linear(
            in_features=inner_dim,
            out_features=dim,
            bias=proj_bias,
        )
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape

        # alternate direction in successive layers
        if self.direction == SequenceTraversal.ROWWISE_FROM_TOP_LEFT:
            pass
        elif self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])
        else:
            raise NotImplementedError

        # up-projection
        x_inner = self.proj_up(x)
        x_mlstm, z = torch.chunk(x_inner, chunks=2, dim=-1)

        # mlstm branch
        x_mlstm_conv = self.conv(x_mlstm)
        x_mlstm_conv_act = F.silu(x_mlstm_conv)
        q = self.q_proj(x_mlstm_conv_act)
        k = self.k_proj(x_mlstm_conv_act)
        v = self.v_proj(x_mlstm)
        h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)
        h_tilde_state_skip = h_tilde_state + (self.learnable_skip * x_mlstm_conv_act)

        # output / z branch
        h_state = h_tilde_state_skip * F.silu(z)

        # down-projection
        x = self.proj_down(h_state)

        # reverse alternating flip
        if self.direction == SequenceTraversal.ROWWISE_FROM_TOP_LEFT:
            pass
        elif self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])
        else:
            raise NotImplementedError

        return x

    def reset_parameters(self):
        # init inproj
        small_init_(self.proj_up.weight, dim=self.dim)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)
        # init outproj (original mLSTM uses num_blocks=1)
        wang_init_(self.proj_down.weight, dim=self.dim, num_blocks=1)
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)

        nn.init.ones_(self.learnable_skip)

        def _init_qkv_proj(qkv_proj: LinearHeadwiseExpand):
            # use the embedding dim instead of the inner embedding dim
            small_init_(qkv_proj.weight, dim=self.dim)
            if qkv_proj.bias is not None:
                nn.init.zeros_(qkv_proj.bias)

        _init_qkv_proj(self.q_proj)
        _init_qkv_proj(self.k_proj)
        _init_qkv_proj(self.v_proj)

        self.mlstm_cell.reset_parameters()


class ViLBlock(nn.Module):
    def __init__(
            self,
            dim,
            direction,
            drop_path=0.0,
            conv_kind="2d",
            conv_kernel_size=3,
            proj_bias=True,
            norm_bias=True,
            seqlens=None,
    ):
        super().__init__()
        self.dim = dim
        self.direction = direction
        self.drop_path = drop_path
        self.conv_kind = conv_kind
        self.conv_kernel_size = conv_kernel_size

        self.drop_path = DropPath(drop_prob=drop_path)
        self.norm = LayerNorm(ndim=dim, weight=True, bias=norm_bias)
        self.layer = ViLLayer(
            dim=dim,
            direction=direction,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            seqlens=seqlens,
            norm_bias=norm_bias,
            proj_bias=proj_bias,
        )
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-4)
        self.reset_parameters()

    def _forward_path(self, x):
        x = self.norm(x)
        x = self.layer(x)
        #return x
        return self.gamma * x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop_path(x, self._forward_path)
        return x

    def reset_parameters(self):
        self.layer.reset_parameters()
        self.norm.reset_parameters()


class VitPatchEmbed(nn.Module):
    def __init__(self, dim, num_channels, resolution, patch_size, stride=None, init_weights="xavier_uniform"):
        """
        Args:
            dim (int): Dimensionality of the output embeddings.
            num_channels (int): Number of input channels.
            resolution (tuple): Spatial resolution of the input image.
            patch_size (int or tuple): Size of each patch.
            stride (int or tuple, optional): Stride of the convolutional layer. Defaults to patch_size.
            init_weights (str, optional): Weight initialization method. Options are "xavier_uniform" or "torch". Defaults to "xavier_uniform".
        """
        super().__init__()
        self.resolution = resolution
        self.init_weights = init_weights
        self.ndim = len(resolution)
        self.patch_size = to_ntuple(patch_size, n=self.ndim)
        self.stride = to_ntuple(stride, n=self.ndim) if stride is not None else self.patch_size

        # Validate resolution and patch size
        for i in range(self.ndim):
            assert (resolution[i] - self.patch_size[i]) % self.stride[i] == 0, \
                f"Bad (resolution, patch, stride) at dim {i}: {resolution[i]}, {self.patch_size[i]}, {self.stride[i]}"

        self.seqlens = [
            (resolution[i] - self.patch_size[i]) // self.stride[i] + 1
            for i in range(self.ndim)
        ]

        # Choose appropriate convolution function
        if self.ndim == 1:
            conv_ctor = nn.Conv1d
        elif self.ndim == 2:
            conv_ctor = nn.Conv2d
        elif self.ndim == 3:
            conv_ctor = nn.Conv3d
        else:
            raise NotImplementedError("Dimension not supported.")

        self.proj = conv_ctor(num_channels, dim, kernel_size=self.patch_size, stride=self.stride)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights based on the specified method."""
        if self.init_weights == "torch":
            pass  # Default initialization
        elif self.init_weights == "xavier_uniform":
            w = self.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.zeros_(self.proj.bias)
        else:
            raise NotImplementedError("Initialization method not supported.")

    def forward(self, x):
        """
        Forward pass of the module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, H, W) or (batch_size, num_channels, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_patches, dim)
        """
        for i in range(self.ndim):
            s = x.size(i + 2)
            assert s >= self.patch_size[i] and (s - self.patch_size[i]) % self.stride[i] == 0, \
                f"x.shape={x.shape} incompatible with patch={self.patch_size} stride={self.stride}"

        
        # Apply convolution to extract patches and project them
        x = self.proj(x)
        
        # Rearrange tensor from (batch_size, dim, H', W') to (batch_size, num_patches, dim)
        x = einops.rearrange(x, "b c ... -> b ... c")
        
        return x


class VitPosEmbed2d(nn.Module):
    def __init__(self, seqlens, dim: int, allow_interpolation: bool = True):
        """
        Args:
            seqlens (tuple): Sequence lengths for each spatial dimension (height, width).
            dim (int): Dimensionality of the positional embeddings.
            allow_interpolation (bool): Whether to allow interpolation of positional embeddings.
        """
        super().__init__()
        self.seqlens = seqlens
        self.dim = dim
        self.allow_interpolation = allow_interpolation
        self.embed = nn.Parameter(torch.zeros(1, *seqlens, dim))
        self.reset_parameters()

    @property
    def _expected_x_ndim(self):
        return len(self.seqlens) + 2

    def reset_parameters(self):
        """Initialize positional embeddings with truncated normal distribution."""
        nn.init.trunc_normal_(self.embed, std=.02)

    def forward(self, x):
        """
        Forward pass of the module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_patches, dim) or similar.

        Returns:
            torch.Tensor: Output tensor with added positional embeddings.
        """
        assert x.ndim == self._expected_x_ndim, f"Expected input with {self._expected_x_ndim} dimensions, got {x.ndim}"
        
        if x.shape[1:] != self.embed.shape[1:]:
            if not self.allow_interpolation:
                raise ValueError("Shape mismatch and interpolation not allowed.")
            # Interpolation function should be defined elsewhere
            embed = interpolate_sincos(embed=self.embed, seqlens=x.shape[1:-1])
        else:
            embed = self.embed
        
        return x + embed


class ViLBlockPair(nn.Module):
    def __init__(
            self,
            dim,
            drop_path=0.0,
            conv_kind="2d",
            conv_kernel_size=3,
            proj_bias=True,
            norm_bias=True,
            seqlens=None,
    ):
        super().__init__()
        self.rowwise_from_top_left = ViLBlock(
            dim=dim,
            direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT,
            drop_path=drop_path,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            proj_bias=proj_bias,
            norm_bias=norm_bias,
            seqlens=seqlens
        )
        self.rowwise_from_bot_right = ViLBlock(
            dim=dim,
            direction=SequenceTraversal.ROWWISE_FROM_BOT_RIGHT,
            drop_path=drop_path,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            proj_bias=proj_bias,
            norm_bias=norm_bias,
            seqlens=seqlens
        )

    def forward(self, x):
        x = self.rowwise_from_top_left(x)
        x = self.rowwise_from_bot_right(x)
        return x

class HaarDWT2d(nn.Module):
    """
    Fixed-weight 2D Haar DWT (depthwise conv + stride=2) producing four sub-bands: LL, LH, HL, HH.
    GPU-friendly and differentiable (weights are fixed and not trained).
    """
    def __init__(self, channels: int, padding: str = "reflect"):
        super().__init__()
        self.channels = channels
        self.padding = padding
        h = torch.tensor([1.0, 1.0]) / (2.0 ** 0.5)
        g = torch.tensor([1.0, -1.0]) / (2.0 ** 0.5)
        LL = torch.einsum('i,j->ij', h, h)  # 2x2
        LH = torch.einsum('i,j->ij', h, g)
        HL = torch.einsum('i,j->ij', g, h)
        HH = torch.einsum('i,j->ij', g, g)
        weight = torch.zeros(4 * channels, 1, 2, 2)
        for c in range(channels):
            weight[c*4 + 0, 0] = LL
            weight[c*4 + 1, 0] = LH
            weight[c*4 + 2, 0] = HL
            weight[c*4 + 3, 0] = HH
        self.register_buffer('weight', weight)

    def forward(self, x):  # x: (B,C,H,W)
        B, C, H, W = x.shape
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            # Only pad on the right/bottom to avoid shifting the feature map.
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        y = F.conv2d(x, self.weight, stride=2, padding=0, groups=self.channels)

        B, OC, H2, W2 = y.shape
        C = self.channels
        y = y.view(B, C, 4, H2, W2)
        LL, LH, HL, HH = y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]  # (B,C,H/2,W/2)
        return LL, LH, HL, HH

class HeadResidualAdapter(nn.Module):
    """
    Residual adapter that projects a branch vector to `head_dim` and injects it into the main vector
    with a learnable per-channel scale and an optional sample-wise gate.

    Base form:
        out = main + alpha ⊙ proj(branch)

    With gating (default):
        delta = proj(branch)
        g = sigmoid(Gate([main, delta]))  # scalar gate (B, 1)
        out = main + g * (alpha ⊙ delta)

    Notes:
    - `alpha` follows a LayerScale-style per-channel scaling (initialized small to avoid strong early injection).
    - The gate provides a data-dependent switch so that noisy/unstable branch features are down-weighted.
    """
    def __init__(
        self,
        head_dim: int,
        branch_dim: int,
        init_scale: float = 1e-2,
        use_gate: bool = True,
        gate_hidden_ratio: float = 0.0,
        gate_init_bias: float = -2.0,
    ):
        super().__init__()
        self.proj = nn.Linear(branch_dim, head_dim, bias=True)
        self.alpha = nn.Parameter(torch.ones(head_dim) * init_scale)
        self.alpha.requires_grad_(False)

        self.use_gate = bool(use_gate)
        if self.use_gate:
            in_dim = head_dim * 2  # concat([main, delta])
            if gate_hidden_ratio and gate_hidden_ratio > 0:
                hidden = max(8, int(in_dim * gate_hidden_ratio))
                self.gate = nn.Sequential(
                    nn.LayerNorm(in_dim),
                    nn.Linear(in_dim, hidden, bias=True),
                    nn.SiLU(),
                    nn.Linear(hidden, 1, bias=True),
                )
            else:
                self.gate = nn.Sequential(
                    nn.LayerNorm(in_dim),
                    nn.Linear(in_dim, 1, bias=True),
                )
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.constant_(self.gate[-1].bias, gate_init_bias)
        else:
            self.gate = None

    def forward(self, main_vec: torch.Tensor, branch_vec: torch.Tensor) -> torch.Tensor:
        delta = self.proj(branch_vec)  # (B, head_dim)
        if self.use_gate:
            g = torch.sigmoid(self.gate(torch.cat([main_vec, delta], dim=-1)))  # (B,1)
            return main_vec + g * (self.alpha * delta)
        return main_vec + (self.alpha * delta)


class StemWithWaveletResidual(nn.Module):
    """
    Stem wrapper that returns both pooled features and separate wavelet features:
        stem(x) -> (pool_only(feat), dwt(feat)).
    Typically used for a ViT-style "pool-only stem + wavelet residual modulation of the head".
    """
    def __init__(self, stem: nn.Module, post_pool_only: nn.Module, dwt_module: nn.Module):
        super().__init__()
        self.stem = stem
        self.post_pool_only = post_pool_only
        self.dwt_module = dwt_module

    def forward(self, x: torch.Tensor):
        feat = self.stem(x)
        main_feat = self.post_pool_only(feat)
        residual_forward = getattr(self.dwt_module, "forward_residual", None)
        wav_feat = residual_forward(feat) if callable(residual_forward) else self.dwt_module(feat)
        return (main_feat, wav_feat)


class StemWithImageWavelet(nn.Module):
    """
    Wrap a conv stem and a PostStemWaveletMerge such that the merge's wavelet branch
    sees the raw image (RGB), while the main path still goes through the conv stem.
    """
    def __init__(self, stem: nn.Module, post: nn.Module):
        super().__init__()
        self.stem = stem
        self.post = post

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem_out = self.stem(x)
        return self.post(stem_out, image=x)


class WaveletGlobalGate(nn.Module):
    """
    Extracts a global vector from wavelet feature maps to lightly modulate the main head.

    Design goals:
      - Depend only on (B, C, H, W) feature maps, so it can plug into arbitrary ViT/ViL-style backbones.
      - Keep parameters and FLOPs very small (typically in the thousands) so overhead is negligible.
    """
    def __init__(self, in_channels: int, dim: int, reduction: int = 4):
        super().__init__()
        in_channels = int(in_channels)
        dim = int(dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(8, in_channels // max(1, int(reduction)))
        self.mlp = nn.Sequential(
            nn.Flatten(),                # (B,C,1,1) -> (B,C)
            nn.Linear(in_channels, hidden, bias=True),
            nn.SiLU(),
            nn.Linear(hidden, dim, bias=True),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: Tensor of shape (B, C, H, W) from the wavelet/conv branch.

        Returns:
            Tensor of shape (B, dim) representing a global modulation vector.
        """
        x = self.pool(feat)
        return self.mlp(x)


class AttnPool(nn.Module):
    """
    Lightweight attention pooling that uses a learnable query to read out a global vector from tokens.

    It is a 1×N cross-attention (not full N×N self-attention), so the computational cost is small.
    Output shape: (B, D).
    """
    def __init__(self, dim: int, num_heads: int = 4, qkv_bias: bool = True):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, bias=qkv_bias, batch_first=True)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, x_tokens: torch.Tensor) -> torch.Tensor:
        # x_tokens: (B, N, D)
        B = x_tokens.shape[0]
        q = self.query.expand(B, -1, -1)  # (B,1,D)
        out, _ = self.attn(q, x_tokens, x_tokens, need_weights=False)  # (B,1,D)
        return out[:, 0, :]


class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.same = (in_ch == out_ch and s == 1)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.act1 = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, k, stride=1, padding=p, bias=False)
        self.short = nn.Identity() if self.same else nn.Conv2d(in_ch, out_ch, 1, stride=s, bias=False)
        self.gamma = nn.Parameter(torch.ones(1) * 1e-4)

    def forward(self, x):
        identity = self.short(x)
        y = self.conv1(self.act1(self.bn1(x)))
        y = self.conv2(self.act2(self.bn2(y)))
        return identity + self.gamma * y


class ResidualDepthwiseMix(nn.Module):
    """
    Lightweight local residual mixer using depthwise separable 3×3 convolution between token sequences and grids.

    Assumes grid = H_patches = W_patches (e.g., 8 or 4 when using patch=4 on CIFAR-10).
    """
    def __init__(self, d_model: int, grid: int):
        super().__init__()
        self.grid = grid
        self.dw = nn.Conv2d(d_model, d_model, 3, padding=1, groups=d_model, bias=False)
        self.pw = nn.Conv2d(d_model, d_model, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):  # x: (B, L, D)
        B, L, D = x.shape
        g = self.grid
        x2 = x.transpose(1, 2).reshape(B, D, g, g)
        y = self.pw(self.dw(x2)).reshape(B, D, g*g).transpose(1, 2)
        return x + self.gamma * y

class FeatureExtractor(nn.Module):
    """
    Optional Haar DWT followed by lightweight convolutional blocks to extract local features.

    dwt_fuse: 'none' | 'LL' | 'concat' | 'add' | 'gated'
    """
    def __init__(self, input_channels: int, conv_channels: list,
                 use_dwt: bool = False, dwt_fuse: str = "LL"):
        super().__init__()
        self.use_dwt = bool(use_dwt)
        self.dwt_fuse = dwt_fuse
        C = input_channels

        self.dwt = None
        self.reduce = None
        self.hf_reduce = None
        self.hf_gate = None

        if self.use_dwt:
            self.dwt = None if (self.dwt_fuse == "none") else HaarDWT2d(C)

            if dwt_fuse == "LL":
                first_in = C

            elif dwt_fuse == "concat":
                first_in = 4 * C

            elif dwt_fuse == "add":
                first_in = C
                self.reduce = nn.Conv2d(4 * C, C, kernel_size=1, bias=False)
                nn.init.zeros_(self.reduce.weight)

            elif dwt_fuse == "gated":
                first_in = C
                self.hf_reduce = nn.Conv2d(3 * C, C, kernel_size=1, bias=True)
                self.hf_gate = nn.Sequential(
                    nn.Conv2d(2 * C, C, kernel_size=1, bias=True),
                    nn.Sigmoid()
                )
                nn.init.zeros_(self.hf_reduce.weight)
                nn.init.zeros_(self.hf_reduce.bias)
                gate_conv = self.hf_gate[0]
                nn.init.zeros_(gate_conv.weight)
                nn.init.constant_(gate_conv.bias, -2.0)

            else:
                raise ValueError("dwt_fuse must be one of {'LL','concat','add','gated'}")
        else:
            first_in = C

        blocks = []
        in_ch = first_in
        for out_ch in conv_channels:
            blocks.append(ResidualConvBlock(in_ch, out_ch, k=3, s=1, p=1))
            in_ch = out_ch
        self.conv_features = nn.Sequential(*blocks)
        self.final_channels = conv_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_dwt:
            z = x
        else:
            ll, lh, hl, hh = self.dwt(x)

            if self.dwt_fuse == "LL":
                z = ll

            elif self.dwt_fuse == "concat":
                z = torch.cat([ll, lh, hl, hh], dim=1)

            elif self.dwt_fuse == "add":
                all4 = torch.cat([ll, lh, hl, hh], dim=1)      # 4C
                z = ll + self.reduce(all4)                     # C

            elif self.dwt_fuse == "gated":
                hf = torch.cat([lh, hl, hh], dim=1)            # 3C
                hf3 = self.hf_reduce(hf)                       # C
                gate = self.hf_gate(torch.cat([ll, hf3], dim=1))  # C
                z = ll + gate * hf3

            else:
                z = ll

        return self.conv_features(z)

class DWTPreprocessor(nn.Module):
    """
    Optional Haar DWT before PatchEmbed (typically used when conv stem is disabled).

    Performs a fixed-weight 2D Haar DWT (stride=2) and fuses sub-bands into a single tensor.
    dwt_fuse: 'none' | 'LL' | 'concat' | 'add' | 'gated'
      - 'LL'     : output LL only, channels = C
      - 'concat' : concat [LL,LH,HL,HH], channels = 4C
      - 'add'    : LL + 1x1( concat 4 bands ) -> channels = C
      - 'gated'  : LL + gate * 1x1( concat high-freq ) -> channels = C

    Output resolution is halved (H/2, W/2).
    """
    def __init__(
        self,
        channels: int,
        dwt_fuse: str = "LL",
        token_wavelet_scale_init: float = 0.1,
        token_wavelet_shrink: float = 0.02,
        token_wavelet_hf_only: bool = True,
        token_wavelet_per_channel: bool = True,
        token_wavelet_warmup_steps: int = 0,
        token_wavelet_hidden_channels: int = 0,
        token_wavelet_split_bands: bool = False,
    ):
        super().__init__()
        self.channels = int(channels)
        self.dwt_fuse = str(dwt_fuse)
        C = self.channels
        self.dwt = None if (self.dwt_fuse == "none") else HaarDWT2d(C)
        self._wavelet_monitor_stats = {}

        self.token_wavelet_hf_only = bool(token_wavelet_hf_only)
        self.token_wavelet_shrink = float(token_wavelet_shrink)
        self.token_wavelet_per_channel = bool(token_wavelet_per_channel)
        self.token_wavelet_warmup_steps = int(token_wavelet_warmup_steps) if token_wavelet_warmup_steps > 0 else 0
        self.token_wavelet_hidden_channels = int(token_wavelet_hidden_channels) if token_wavelet_hidden_channels > 0 else C
        self.token_wavelet_split_bands = bool(token_wavelet_split_bands) and bool(token_wavelet_hf_only)
        self.register_buffer("_wavelet_step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_current_global_step", torch.tensor(-1, dtype=torch.long))

        self.reduce = None
        self.hf_reduce = None
        self.hf_band_reduce = None
        self.hf_gate = None
        self.hf_scale = None
        hidden_channels = self.token_wavelet_hidden_channels

        if self.dwt_fuse == "none":
            # Pool-only ablation / no wavelet branch.
            # We still conceptually downsample by 2 (handled by the caller), but output has 0 channels.
            self.out_channels = 0
            self.residual_out_channels = 0
        elif self.dwt_fuse == "concat":
            self.out_channels = 4 * C
            self.residual_out_channels = self.out_channels
        elif self.dwt_fuse == "LL":
            self.out_channels = C
            self.residual_out_channels = self.out_channels
        elif self.dwt_fuse in ("add", "gated"):
            self.out_channels = C
            self.residual_out_channels = hidden_channels
        else:
            raise ValueError("dwt_fuse must be one of {'none','LL','concat','add','gated'}")

        if self.dwt_fuse == "add":
            if self.token_wavelet_hf_only:
                if self.token_wavelet_split_bands:
                    band_hidden = self._split_hidden_channels(hidden_channels)
                    self.hf_band_reduce = nn.ModuleList(
                        [nn.Conv2d(C, out_ch, kernel_size=1, bias=True) for out_ch in band_hidden]
                    )
                    for layer in self.hf_band_reduce:
                        nn.init.trunc_normal_(layer.weight, std=1e-2)
                        nn.init.zeros_(layer.bias)
                else:
                    self.hf_reduce = nn.Conv2d(3 * C, hidden_channels, kernel_size=1, bias=True)
                    nn.init.trunc_normal_(self.hf_reduce.weight, std=1e-2)
                    nn.init.zeros_(self.hf_reduce.bias)

                if self.token_wavelet_per_channel:
                    self.hf_scale = nn.Parameter(torch.ones(hidden_channels) * float(token_wavelet_scale_init))
                else:
                    self.hf_scale = nn.Parameter(torch.tensor(float(token_wavelet_scale_init)))
            else:
                self.reduce = nn.Conv2d(4 * C, hidden_channels, kernel_size=1, bias=True)
                # safe init: start from identity (out = LL)
                nn.init.zeros_(self.reduce.weight)
                nn.init.zeros_(self.reduce.bias)
        elif self.dwt_fuse == "gated":
            if self.token_wavelet_hf_only:
                if self.token_wavelet_split_bands:
                    band_hidden = self._split_hidden_channels(hidden_channels)
                    self.hf_band_reduce = nn.ModuleList(
                        [nn.Conv2d(C, out_ch, kernel_size=1, bias=True) for out_ch in band_hidden]
                    )
                    for layer in self.hf_band_reduce:
                        nn.init.zeros_(layer.weight)
                        nn.init.zeros_(layer.bias)
                else:
                    self.hf_reduce = nn.Conv2d(3 * C, hidden_channels, kernel_size=1, bias=True)
                    nn.init.zeros_(self.hf_reduce.weight)
                    nn.init.zeros_(self.hf_reduce.bias)
            else:
                self.reduce = nn.Conv2d(4 * C, hidden_channels, kernel_size=1, bias=True)
                nn.init.zeros_(self.reduce.weight)
                nn.init.zeros_(self.reduce.bias)
            self.hf_gate = nn.Sequential(
                nn.Conv2d(C + hidden_channels, hidden_channels, kernel_size=1, bias=True),
                nn.Sigmoid()
            )
            # safe init: suppress HF at start (out ≈ LL)
            gate_conv = self.hf_gate[0]
            nn.init.zeros_(gate_conv.weight)
            nn.init.constant_(gate_conv.bias, -2.0)

    def get_wavelet_monitor_stats(self):
        return dict(self._wavelet_monitor_stats)

    @staticmethod
    def _split_hidden_channels(total_channels: int):
        total_channels = int(total_channels)
        if total_channels < 3:
            raise ValueError("token_wavelet_split_bands=True requires token_wavelet_hidden_channels >= 3.")
        base = total_channels // 3
        rem = total_channels % 3
        return [base + (1 if i < rem else 0) for i in range(3)]

    def set_wavelet_global_step(self, global_step: int):
        if self.token_wavelet_warmup_steps > 0:
            self._current_global_step.fill_(int(global_step))

    def _get_token_warmup_factor(self) -> float:
        if self.token_wavelet_warmup_steps <= 0 or not self.training:
            return 1.0
        if self._current_global_step.item() >= 0:
            global_step = int(self._current_global_step.item())
        else:
            global_step = int(self._wavelet_step.item())
            self._wavelet_step += 1
        return min(1.0, max(0.0, float(global_step) / float(self.token_wavelet_warmup_steps)))

    def _record_wavelet_stats(self, base_abs: float, delta_abs: float, out_abs: float, extra=None):
        stats = {
            "base_abs_mean": base_abs,
            "delta_abs_mean": delta_abs,
            "delta_over_base": _safe_ratio(delta_abs, base_abs),
            "out_abs_mean": out_abs,
        }
        if extra:
            stats.update(extra)
        self._wavelet_monitor_stats = stats

    def _reduce_high_freq(self, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
        if self.token_wavelet_shrink > 0:
            lh = F.softshrink(lh, lambd=self.token_wavelet_shrink)
            hl = F.softshrink(hl, lambd=self.token_wavelet_shrink)
            hh = F.softshrink(hh, lambd=self.token_wavelet_shrink)
        if self.hf_band_reduce is not None:
            return torch.cat(
                [
                    self.hf_band_reduce[0](lh),
                    self.hf_band_reduce[1](hl),
                    self.hf_band_reduce[2](hh),
                ],
                dim=1,
            )
        hf = torch.cat([lh, hl, hh], dim=1)
        return self.hf_reduce(hf)

    def _project_hidden_delta(self, delta: torch.Tensor, residual_only: bool) -> torch.Tensor:
        if delta.shape[1] == self.channels or residual_only:
            return delta
        raise RuntimeError(
            "token_wavelet_hidden_channels requires residual-only usage in DWTPreprocessor; "
            "use it through PostStemWaveletMerge instead of direct DWTPreprocessor.forward()."
        )

    def _forward_impl(self, x: torch.Tensor, residual_only: bool = False) -> torch.Tensor:
        if self.dwt_fuse == "none":
            # Return an empty tensor with downsampled spatial size (H/2, W/2).
            B, _, H, W = x.shape
            out = x.new_zeros((B, 0, H // 2, W // 2))
            self._record_wavelet_stats(
                base_abs=0.0,
                delta_abs=0.0,
                out_abs=0.0,
                extra={"token_warmup_factor": 1.0, "residual_only": float(residual_only)},
            )
            return out
        ll, lh, hl, hh = self.dwt(x)
        warmup_factor = self._get_token_warmup_factor()
        if self.dwt_fuse == "LL":
            base_abs = _abs_mean(ll)
            out = torch.zeros_like(ll) if residual_only else ll
            self._record_wavelet_stats(
                base_abs=base_abs,
                delta_abs=0.0,
                out_abs=_abs_mean(out),
                extra={"token_warmup_factor": 1.0, "residual_only": float(residual_only)},
            )
            return out
        if self.dwt_fuse == "concat":
            hf = torch.cat([lh, hl, hh], dim=1) * warmup_factor
            ll_part = torch.zeros_like(ll) if residual_only else ll
            out = torch.cat([ll_part, hf], dim=1)
            base_abs = _abs_mean(ll)
            delta_abs = _abs_mean(hf)
            self._record_wavelet_stats(
                base_abs=base_abs,
                delta_abs=delta_abs,
                out_abs=_abs_mean(out),
                extra={"token_warmup_factor": warmup_factor, "residual_only": float(residual_only)},
            )
            return out
        if self.dwt_fuse == "add":
            if self.token_wavelet_hf_only:
                hf3 = self._reduce_high_freq(lh, hl, hh)

                if self.token_wavelet_per_channel:
                    scale = self.hf_scale.view(1, -1, 1, 1)
                else:
                    scale = self.hf_scale
                effective_scale = scale * warmup_factor
                hidden_delta = effective_scale * hf3
                delta = self._project_hidden_delta(hidden_delta, residual_only=residual_only)
                out = delta if residual_only else (ll + delta)
                base_abs = _abs_mean(ll)
                delta_abs = _abs_mean(delta)
                self._record_wavelet_stats(
                    base_abs=base_abs,
                    delta_abs=delta_abs,
                    out_abs=_abs_mean(out),
                    extra={
                        "token_warmup_factor": warmup_factor,
                        "token_effective_scale_mean": float(effective_scale.detach().abs().mean().item()),
                        "residual_only": float(residual_only),
                        "hidden_channels": float(self.token_wavelet_hidden_channels),
                        "split_bands": float(self.token_wavelet_split_bands),
                    },
                )
                return out

            ll_for_reduce = torch.zeros_like(ll) if residual_only else ll
            all4 = torch.cat([ll_for_reduce, lh, hl, hh], dim=1)
            hidden_delta = warmup_factor * self.reduce(all4)
            delta = self._project_hidden_delta(hidden_delta, residual_only=residual_only)
            out = delta if residual_only else (ll + delta)
            base_abs = _abs_mean(ll)
            delta_abs = _abs_mean(delta)
            self._record_wavelet_stats(
                base_abs=base_abs,
                delta_abs=delta_abs,
                out_abs=_abs_mean(out),
                extra={
                    "token_warmup_factor": warmup_factor,
                    "residual_only": float(residual_only),
                    "hidden_channels": float(self.token_wavelet_hidden_channels),
                    "split_bands": float(self.token_wavelet_split_bands),
                },
            )
            return out
        # gated
        if self.token_wavelet_hf_only:
            wave_feat = self._reduce_high_freq(lh, hl, hh)
        else:
            hf = torch.cat([lh, hl, hh], dim=1)
            if self.token_wavelet_shrink > 0:
                hf = F.softshrink(hf, lambd=self.token_wavelet_shrink)
            wave_feat = self.reduce(torch.cat([ll, hf], dim=1))
        gate = self.hf_gate(torch.cat([ll, wave_feat], dim=1))
        hidden_delta = warmup_factor * (gate * wave_feat)
        delta = self._project_hidden_delta(hidden_delta, residual_only=residual_only)
        out = delta if residual_only else (ll + delta)
        base_abs = _abs_mean(ll)
        delta_abs = _abs_mean(delta)
        self._record_wavelet_stats(
            base_abs=base_abs,
            delta_abs=delta_abs,
            out_abs=_abs_mean(out),
            extra={
                "token_warmup_factor": warmup_factor,
                "residual_only": float(residual_only),
                "gate_abs_mean": _abs_mean(gate),
                "hf_only": float(self.token_wavelet_hf_only),
                "hidden_channels": float(self.token_wavelet_hidden_channels),
                "split_bands": float(self.token_wavelet_split_bands),
            },
        )
        return out

    def forward_residual(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x, residual_only=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x, residual_only=False)



class PostStemWaveletMerge(nn.Module):
    """
    Apply Haar DWT after conv stem and merge with downsampled conv features.

    merge="replace": just return DWTPreprocessor(x)  (handled outside)
    merge="concat":  y = AvgPool2d(x) + scale * proj(DWT_residual(x))
    """
    def __init__(
        self,
        channels: int,
        dwt_fuse: str = "add",
        merge: str = "concat",
        token_wavelet_scale_init: float = 0.1,
        token_wavelet_inner_scale_init=None,
        token_wavelet_outer_scale_init=None,
        token_wavelet_shrink: float = 0.02,
        token_wavelet_hf_only: bool = True,
        token_wavelet_per_channel: bool = True,
        token_wavelet_warmup_steps: int = 0,
        token_wavelet_hidden_channels: int = 0,
        token_wavelet_side_channels: int = 0,
        token_wavelet_side_mode: str = "concat",
        token_wavelet_outer_gate: bool = False,
        token_wavelet_split_bands: bool = False,
        token_wavelet_image_input_channels: int = 0,
    ):
        super().__init__()
        self.channels = int(channels)
        self.merge = str(merge)
        self.dwt_fuse = str(dwt_fuse)
        self.token_wavelet_per_channel = bool(token_wavelet_per_channel)
        self.token_wavelet_hidden_channels = int(token_wavelet_hidden_channels) if token_wavelet_hidden_channels > 0 else 0
        self.token_wavelet_side_channels = int(token_wavelet_side_channels) if token_wavelet_side_channels > 0 else 0
        self.token_wavelet_side_mode = str(token_wavelet_side_mode or "concat").strip().lower()
        if self.token_wavelet_side_mode not in ("concat", "patch"):
            raise ValueError("token_wavelet_side_mode must be 'concat' or 'patch'.")
        self.token_wavelet_outer_gate = bool(token_wavelet_outer_gate)
        self.token_wavelet_split_bands = bool(token_wavelet_split_bands)
        self.token_wavelet_image_input_channels = (
            int(token_wavelet_image_input_channels) if token_wavelet_image_input_channels > 0 else 0
        )
        self._wavelet_monitor_stats = {}
        self._last_side_feature = None
        shared_scale_init = float(token_wavelet_scale_init)
        self.token_wavelet_inner_scale_init = float(shared_scale_init if token_wavelet_inner_scale_init is None else token_wavelet_inner_scale_init)
        self.token_wavelet_outer_scale_init = float(shared_scale_init if token_wavelet_outer_scale_init is None else token_wavelet_outer_scale_init)
        self.token_wavelet_warmup_steps = int(token_wavelet_warmup_steps) if token_wavelet_warmup_steps > 0 else 0
        self.register_buffer("_wavelet_step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_current_global_step", torch.tensor(-1, dtype=torch.long))

        # When image_input_channels > 0, the inner DWT runs on the raw RGB image
        # instead of the conv-stem output. Its single-level Haar halves the image
        # spatial size; we then adaptively align it to match pool(stem_out) regardless
        # of whether the conv stem uses stride 1 or 2.
        dwt_in_channels = self.token_wavelet_image_input_channels if self.token_wavelet_image_input_channels > 0 else self.channels
        self.dwt = DWTPreprocessor(
            channels=dwt_in_channels,
            dwt_fuse=dwt_fuse,
            token_wavelet_scale_init=self.token_wavelet_inner_scale_init,
            token_wavelet_shrink=token_wavelet_shrink,
            token_wavelet_hf_only=token_wavelet_hf_only,
            token_wavelet_per_channel=token_wavelet_per_channel,
            token_wavelet_warmup_steps=0,
            token_wavelet_hidden_channels=self.token_wavelet_hidden_channels,
            token_wavelet_split_bands=self.token_wavelet_split_bands,
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        if self.merge != "concat":
            raise ValueError("PostStemWaveletMerge currently supports merge='concat' only.")

        residual_channels = int(getattr(self.dwt, "residual_out_channels", self.dwt.out_channels))
        self.wave_channels = residual_channels
        self.side_channels = self.token_wavelet_side_channels if residual_channels > 0 else 0
        self.side_uses_patch_embed = self.side_channels > 0 and self.token_wavelet_side_mode == "patch"
        self.output_channels = self.channels if self.side_uses_patch_embed else (self.channels + self.side_channels)
        if residual_channels > 0:
            self.wave_proj = nn.Conv2d(residual_channels, self.channels, kernel_size=1, bias=True)
            nn.init.trunc_normal_(self.wave_proj.weight, std=1e-2)
            nn.init.zeros_(self.wave_proj.bias)
            if self.token_wavelet_per_channel:
                self.wave_scale = nn.Parameter(torch.ones(self.channels) * self.token_wavelet_outer_scale_init)
            else:
                self.wave_scale = nn.Parameter(torch.tensor(self.token_wavelet_outer_scale_init))
            if self.side_channels > 0:
                self.wave_side_proj = nn.Conv2d(residual_channels, self.side_channels, kernel_size=1, bias=True)
                nn.init.trunc_normal_(self.wave_side_proj.weight, std=1e-2)
                nn.init.zeros_(self.wave_side_proj.bias)
                self.wave_side_norm = nn.InstanceNorm2d(self.side_channels, affine=True)
                if self.token_wavelet_per_channel:
                    self.wave_side_scale = nn.Parameter(torch.ones(self.side_channels) * self.token_wavelet_outer_scale_init)
                else:
                    self.wave_side_scale = nn.Parameter(torch.tensor(self.token_wavelet_outer_scale_init))
            else:
                self.wave_side_proj = None
                self.wave_side_norm = None
                self.wave_side_scale = None
            if self.token_wavelet_outer_gate:
                gate_out_channels = self.channels if self.side_uses_patch_embed else (self.channels + self.side_channels)
                self.outer_gate = nn.Sequential(
                    nn.Conv2d(self.channels, gate_out_channels, kernel_size=1, bias=True),
                    nn.Sigmoid(),
                )
                gate_conv = self.outer_gate[0]
                nn.init.zeros_(gate_conv.weight)
                nn.init.constant_(gate_conv.bias, -2.0)
            else:
                self.outer_gate = None
        else:
            self.wave_proj = None
            self.wave_scale = None
            self.wave_side_proj = None
            self.wave_side_norm = None
            self.wave_side_scale = None
            self.outer_gate = None

    @property
    def out_channels(self):
        return self.output_channels

    def get_wavelet_monitor_stats(self):
        stats = dict(self._wavelet_monitor_stats)
        if hasattr(self.dwt, "get_wavelet_monitor_stats"):
            child_stats = self.dwt.get_wavelet_monitor_stats()
            for key, value in child_stats.items():
                stats[f"inner_{key}"] = value
        return stats

    def get_side_feature(self):
        return self._last_side_feature

    def set_wavelet_global_step(self, global_step: int):
        if self.token_wavelet_warmup_steps > 0:
            self._current_global_step.fill_(int(global_step))
        if hasattr(self.dwt, "set_wavelet_global_step"):
            self.dwt.set_wavelet_global_step(global_step)

    def _get_token_warmup_factor(self) -> float:
        if self.token_wavelet_warmup_steps <= 0 or not self.training:
            return 1.0
        if self._current_global_step.item() >= 0:
            global_step = int(self._current_global_step.item())
        else:
            global_step = int(self._wavelet_step.item())
            self._wavelet_step += 1
        return min(1.0, max(0.0, float(global_step) / float(self.token_wavelet_warmup_steps)))

    def forward(self, x: torch.Tensor, image: torch.Tensor = None) -> torch.Tensor:
        self._last_side_feature = None
        x_ds = self.pool(x)
        pool_abs = _abs_mean(x_ds)
        if self.dwt.out_channels == 0:
            self._wavelet_monitor_stats = {
                "pool_abs_mean": pool_abs,
                "wave_abs_mean": 0.0,
                "token_delta_abs_mean": 0.0,
                "token_delta_over_pool": 0.0,
                "out_abs_mean": pool_abs,
                "out_main_abs_mean": pool_abs,
                "token_warmup_factor": 1.0,
                "token_hidden_channels": float(self.wave_channels),
                "token_side_channels": float(self.side_channels),
                "token_side_abs_mean": 0.0,
                "token_side_mode_patch": float(self.side_uses_patch_embed),
                "token_alpha_abs_mean": 0.0,
                "token_outer_gate_abs_mean": 0.0,
                "token_outer_gate_enabled": float(self.token_wavelet_outer_gate),
                "token_split_bands": float(self.token_wavelet_split_bands),
                "token_image_input": float(self.token_wavelet_image_input_channels),
            }
            return x_ds
        if self.token_wavelet_image_input_channels > 0:
            assert image is not None, (
                "PostStemWaveletMerge built with token_wavelet_image_input_channels>0 "
                "but forward was called without an image argument."
            )
            assert image.shape[1] == self.token_wavelet_image_input_channels, (
                f"image has {image.shape[1]} channels but DWT expects "
                f"{self.token_wavelet_image_input_channels}."
            )
            dwt_input = image
        else:
            dwt_input = x
        residual_forward = getattr(self.dwt, "forward_residual", None)
        w_res = residual_forward(dwt_input) if callable(residual_forward) else self.dwt(dwt_input)
        if self.token_wavelet_image_input_channels > 0 and w_res.shape[-2:] != x_ds.shape[-2:]:
            w_res = F.adaptive_avg_pool2d(w_res, x_ds.shape[-2:])
        w_main = self.wave_proj(w_res)
        warmup_factor = self._get_token_warmup_factor()
        if self.token_wavelet_per_channel:
            scale_main = self.wave_scale.view(1, -1, 1, 1)
        else:
            scale_main = self.wave_scale
        effective_scale = scale_main * warmup_factor
        alpha = effective_scale * torch.tanh(w_main)
        side_feat = None
        gate_main = None
        gate_side = None
        if self.outer_gate is not None:
            gate_all = self.outer_gate(x_ds)
            gate_main = gate_all[:, : self.channels]
            if self.side_channels > 0:
                gate_side = gate_all[:, self.channels :]
        if gate_main is not None:
            alpha = gate_main * alpha
        delta = x_ds * alpha
        out_main = x_ds + delta
        if self.wave_side_proj is not None:
            w_side = self.wave_side_proj(w_res)
            if self.wave_side_norm is not None:
                w_side = self.wave_side_norm(w_side)
            if self.token_wavelet_per_channel:
                scale_side = self.wave_side_scale.view(1, -1, 1, 1)
            else:
                scale_side = self.wave_side_scale
            side_feat = (scale_side * warmup_factor) * w_side
            if (not self.side_uses_patch_embed) and gate_side is not None:
                side_feat = gate_side * side_feat
            if self.side_uses_patch_embed:
                self._last_side_feature = side_feat
                out = out_main
            else:
                out = torch.cat([out_main, side_feat], dim=1)
        else:
            out = out_main
        delta_abs = _abs_mean(delta)
        side_abs = _abs_mean(side_feat)
        self._wavelet_monitor_stats = {
            "pool_abs_mean": pool_abs,
            "wave_abs_mean": _abs_mean(w_main),
            "token_delta_abs_mean": delta_abs,
            "token_delta_over_pool": _safe_ratio(delta_abs, pool_abs),
            "out_abs_mean": _abs_mean(out),
            "out_main_abs_mean": _abs_mean(out_main),
            "token_warmup_factor": warmup_factor,
            "token_effective_scale_mean": float(effective_scale.detach().abs().mean().item()),
            "token_scale_per_channel": float(self.token_wavelet_per_channel),
            "token_hidden_channels": float(self.wave_channels),
            "token_side_channels": float(self.side_channels),
            "token_side_abs_mean": side_abs,
            "token_side_mode_patch": float(self.side_uses_patch_embed),
            "token_alpha_abs_mean": _abs_mean(alpha),
            "token_outer_gate_abs_mean": _abs_mean(gate_main) if gate_main is not None else 1.0,
            "token_outer_gate_enabled": float(self.outer_gate is not None),
            "token_split_bands": float(self.token_wavelet_split_bands),
            "token_image_input": float(self.token_wavelet_image_input_channels),
        }
        return out


class VisionLSTM2(nn.Module):
    def __init__(
    self,
    dim,
    input_shape,
    patch_size,
    depth,
    output_shape,
    mode,
    pooling,
    drop_path_rate,
    drop_path_decay,
    stride,
    legacy_norm,
    conv_kind,
    conv_kernel_size,
    proj_bias,
    norm_bias,
    feature_extractor_channels,
    use_dwt=False,
    dwt_fuse="gated",
    auto_patch_dwt=True,
    use_conv_stem=True,
    pre_patch_dwt=False,
    disable_branch=False,
    head_inject_gated=True,
    head_gate_hidden_ratio=0.0,
    head_gate_init_bias=-2.0,
    attn_pool_heads=4,
    post_stem_dwt=False,
    post_stem_merge="replace",
    wavelet_warmup_steps=0,
    wavelet_fuse_mode="multiply",
    head_wavelet_residual=True,
    wavelet_scale_init=0.0,

    # ----- new token-wavelet args -----
    token_wavelet_scale_init=0.1,
    token_wavelet_inner_scale_init=None,
    token_wavelet_outer_scale_init=None,
    token_wavelet_shrink=0.02,
    token_wavelet_hf_only=True,
    token_wavelet_per_channel=True,
    token_wavelet_hidden_channels=0,
    token_wavelet_side_channels=0,
    token_wavelet_side_mode="concat",
    token_wavelet_side_beta_init=0.1,
    token_wavelet_outer_gate=False,
    token_wavelet_split_bands=False,
    wavelet_input_image=False,
):
        super(VisionLSTM2, self).__init__()

        self.dim = dim
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.depth = depth
        self.output_shape = output_shape
        self.mode = mode
        self.pooling = pooling
        self.drop_path_rate = drop_path_rate
        self.drop_path_decay = drop_path_decay
        self.stride = stride
        self.legacy_norm = legacy_norm
        self.conv_kind = conv_kind
        self.conv_kernel_size = conv_kernel_size
        self.proj_bias = proj_bias
        self.norm_bias = norm_bias
        
        # --- Stem / (optional) pre-patch DWT configuration ---
        self.use_conv_stem = bool(use_conv_stem)
        self.pre_patch_dwt = bool(pre_patch_dwt)
        self.disable_branch = bool(disable_branch)
        self.post_stem_dwt = bool(post_stem_dwt)
        self.post_stem_merge = str(post_stem_merge)
        
        self.token_wavelet_scale_init = float(token_wavelet_scale_init)
        self.token_wavelet_inner_scale_init = float(self.token_wavelet_scale_init if token_wavelet_inner_scale_init is None else token_wavelet_inner_scale_init)
        self.token_wavelet_outer_scale_init = float(self.token_wavelet_scale_init if token_wavelet_outer_scale_init is None else token_wavelet_outer_scale_init)
        self.token_wavelet_shrink = float(token_wavelet_shrink)
        self.token_wavelet_hf_only = bool(token_wavelet_hf_only)
        self.token_wavelet_per_channel = bool(token_wavelet_per_channel)
        self.token_wavelet_hidden_channels = int(token_wavelet_hidden_channels) if token_wavelet_hidden_channels > 0 else 0
        self.token_wavelet_side_channels = int(token_wavelet_side_channels) if token_wavelet_side_channels > 0 else 0
        self.token_wavelet_side_mode = str(token_wavelet_side_mode or "concat").strip().lower()
        self.token_wavelet_side_beta_init = float(token_wavelet_side_beta_init)
        self.token_wavelet_outer_gate = bool(token_wavelet_outer_gate)
        self.token_wavelet_split_bands = bool(token_wavelet_split_bands)
        self.wavelet_input_image = bool(wavelet_input_image)
        self._wavelet_monitor_stats = {}


        if self.use_conv_stem and self.pre_patch_dwt:
            raise ValueError("pre_patch_dwt=True is intended for the no-stem ablation (use_conv_stem=False).")
        if (not self.use_conv_stem) and bool(use_dwt):
            raise ValueError("use_dwt=True applies to the conv stem. For DWT-only ablation, set pre_patch_dwt=True and use_conv_stem=False.")

        if self.post_stem_dwt and (not self.use_conv_stem):
            raise ValueError("post_stem_dwt=True requires use_conv_stem=True (it applies after the conv stem).")
        if self.post_stem_dwt and (bool(use_dwt) or self.pre_patch_dwt):
            raise ValueError("post_stem_dwt=True expects no other DWT downsample. Set use_dwt=False and pre_patch_dwt=False.")

        dwt_before_patch = (bool(use_dwt) if self.use_conv_stem else self.pre_patch_dwt)
        ds = 1
        if dwt_before_patch:
            ds *= 2
        if self.post_stem_dwt:
            ds *= 2

        # Optional DWT before patch embedding (only when conv stem is disabled)
        if self.pre_patch_dwt:
            self.pre_patch = DWTPreprocessor(
                channels=input_shape[0],
                dwt_fuse=dwt_fuse,
                token_wavelet_scale_init=self.token_wavelet_inner_scale_init,
                token_wavelet_shrink=self.token_wavelet_shrink,
                token_wavelet_hf_only=self.token_wavelet_hf_only,
                token_wavelet_per_channel=self.token_wavelet_per_channel,
                token_wavelet_warmup_steps=wavelet_warmup_steps,
            )
            pre_patch_channels = self.pre_patch.out_channels
        else:
            self.pre_patch = nn.Identity()
            pre_patch_channels = input_shape[0]

        # Optional conv stem (FeatureExtractor)
        if self.use_conv_stem:
            assert feature_extractor_channels is not None and len(feature_extractor_channels) > 0, "feature_extractor_channels must be a non-empty list when use_conv_stem=True"
            self.feature_extractor = FeatureExtractor(
                input_channels=input_shape[0],
                conv_channels=feature_extractor_channels,
                use_dwt=use_dwt,
                dwt_fuse=dwt_fuse,
            )
            num_channels = self.feature_extractor.final_channels
        else:
            self.feature_extractor = nn.Identity()
            num_channels = pre_patch_channels

        # Optional post-stem Haar/DWT downsample (after conv stem, before patch embedding)
        stem_out_channels = int(num_channels)
        if self.post_stem_dwt:
            if self.post_stem_merge == "replace":
                self.post_stem = DWTPreprocessor(
                    channels=num_channels,
                    dwt_fuse=dwt_fuse,
                    token_wavelet_scale_init=self.token_wavelet_inner_scale_init,
                    token_wavelet_shrink=self.token_wavelet_shrink,
                    token_wavelet_hf_only=self.token_wavelet_hf_only,
                    token_wavelet_per_channel=self.token_wavelet_per_channel,
                    token_wavelet_warmup_steps=wavelet_warmup_steps,
                )
            elif self.post_stem_merge == "concat":
                self.post_stem = PostStemWaveletMerge(
                    channels=num_channels,
                    dwt_fuse=dwt_fuse,
                    merge="concat",
                    token_wavelet_scale_init=self.token_wavelet_scale_init,
                    token_wavelet_inner_scale_init=self.token_wavelet_inner_scale_init,
                    token_wavelet_outer_scale_init=self.token_wavelet_outer_scale_init,
                    token_wavelet_shrink=self.token_wavelet_shrink,
                    token_wavelet_hf_only=self.token_wavelet_hf_only,
                    token_wavelet_per_channel=self.token_wavelet_per_channel,
                    token_wavelet_warmup_steps=wavelet_warmup_steps,
                    token_wavelet_hidden_channels=self.token_wavelet_hidden_channels,
                    token_wavelet_side_channels=self.token_wavelet_side_channels,
                    token_wavelet_side_mode=self.token_wavelet_side_mode,
                    token_wavelet_outer_gate=self.token_wavelet_outer_gate,
                    token_wavelet_split_bands=self.token_wavelet_split_bands,
                    token_wavelet_image_input_channels=int(input_shape[0]) if self.wavelet_input_image else 0,
                )
            else:
                raise ValueError("post_stem_merge must be 'replace' or 'concat'")
            num_channels = self.post_stem.out_channels
        else:
            self.post_stem = nn.Identity()

        # Record channels for branch construction
        self.branch_in_channels = int(num_channels)

        # PatchEmbed resolution and (optional) auto-adjust of patch/stride under DWT downsampling
        pe_res = (input_shape[1] // ds, input_shape[2] // ds)
        patch_t = to_ntuple(patch_size, n=2)
        stride_t = to_ntuple(stride if stride is not None else patch_size, n=2)
        patch_eff, stride_eff = patch_t, stride_t
        if bool(auto_patch_dwt) and ds == 2:
            # Interpret the provided patch/stride as the 'base' values at original resolution.
            if any((p % 2) != 0 for p in patch_t) or any((s % 2) != 0 for s in stride_t):
                raise ValueError("auto_patch_dwt=True requires even patch_size/stride when DWT downsamples by 2.")
            patch_eff = tuple(p // 2 for p in patch_t)
            stride_eff = tuple(s // 2 for s in stride_t)

        self.patch_size_eff = patch_eff
        self.stride_eff = stride_eff

        self.patch_embed = VitPatchEmbed(
            dim=dim,
            num_channels=num_channels,
            resolution=pe_res,
            patch_size=self.patch_size_eff,
            stride=self.stride_eff,
            init_weights="xavier_uniform"
        )
        side_patch_channels = 0
        if hasattr(self.post_stem, "side_uses_patch_embed") and self.post_stem.side_uses_patch_embed:
            side_patch_channels = int(getattr(self.post_stem, "side_channels", 0))
        if side_patch_channels > 0:
            self.patch_embed_side = VitPatchEmbed(
                dim=dim,
                num_channels=side_patch_channels,
                resolution=pe_res,
                patch_size=self.patch_size_eff,
                stride=self.stride_eff,
                init_weights="xavier_uniform"
            )
            if self.patch_embed_side.seqlens != self.patch_embed.seqlens:
                raise ValueError("patch_embed_side seqlens must match main patch_embed seqlens.")
            self.side_token_beta = nn.Parameter(torch.tensor(self.token_wavelet_side_beta_init))
        else:
            self.patch_embed_side = None
            self.side_token_beta = None

        # Initialize learnable positional embedding
        self.pos_embed = VitPosEmbed2d(seqlens=self.patch_embed.seqlens, dim=dim)

        # Initialize blocks
        if drop_path_decay and drop_path_rate > 0.:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        else:
            dpr = [drop_path_rate] * depth

        self.blocks = nn.ModuleList(
            [
                ViLBlockPair(
                    dim=dim,
                    drop_path=dpr[i],
                    conv_kind=conv_kind,
                    seqlens=self.patch_embed.seqlens,
                    proj_bias=proj_bias,
                    norm_bias=norm_bias,
                )
                for i in range(depth)
            ],
        )
        if pooling == "bilateral_flatten":
            head_dim = dim * 2
        else:
            head_dim = dim
        self.norm = LayerNorm(dim, bias=norm_bias, eps=1e-6)
        if legacy_norm:
            self.legacy_norm = nn.LayerNorm(head_dim)
        else:
            self.legacy_norm = nn.Identity()

        # Optional attention pooling (1×N cross-attn)
        self.attn_pool = None
        if self.pooling == "attn":
            self.attn_pool = AttnPool(dim=dim, num_heads=int(attn_pool_heads))

        # Classification head
        if mode == "features":
            assert self.output_shape is None
            self.head = None
            if self.pooling is None:
                self.output_shape = (self.patch_embed.num_patches, dim)
            elif self.pooling == "to_image":
                self.output_shape = (dim, *self.patch_embed.seqlens)
            elif self.pooling == "attn":
                self.output_shape = (dim,)
            else:
                raise NotImplementedError(f"invalid pooling '{pooling}' for mode '{mode}'")
        elif mode == "classifier":
            assert self.output_shape is not None and len(self.output_shape) == 1, \
                f"define number of classes via output_shape=(num_classes,) (e.g. output_shape=(1000,) for ImageNet-1K"
            self.head = nn.Linear(head_dim, self.output_shape[0])
            nn.init.trunc_normal_(self.head.weight, std=2e-5)
            nn.init.zeros_(self.head.bias)
            self.head_adapter = HeadResidualAdapter(head_dim=head_dim, branch_dim=dim, use_gate=bool(head_inject_gated), gate_hidden_ratio=float(head_gate_hidden_ratio), gate_init_bias=float(head_gate_init_bias)) if (not self.disable_branch) else None
        else:
            raise NotImplementedError

        # Branch for additional feature maps (optional)
        if self.disable_branch:
            self.feature_extractor_branch = None
        else:
            branch_in = self.branch_in_channels
            self.feature_extractor_branch = nn.Sequential(
                nn.Conv2d(in_channels=branch_in, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(8),
                nn.Flatten(),
                nn.Linear(32 * 8 * 8, 384),
                nn.SiLU(),
                nn.Linear(384, dim)
            )

        self.dwt_for_residual = None
        post_stem_dwt_uses_image = bool(getattr(self.post_stem, "token_wavelet_image_input_channels", 0) > 0)
        if bool(head_wavelet_residual) and self.post_stem_dwt and self.post_stem_merge == "concat":
            wav_ch = 0
            if (
                hasattr(self.post_stem, "dwt")
                and self.post_stem.dwt is not None
                and not post_stem_dwt_uses_image
            ):
                wav_ch = int(getattr(self.post_stem.dwt, "residual_out_channels", self.post_stem.dwt.out_channels))
            if wav_ch > 0:
                self.wavelet_residual = WaveletGlobalGate(in_channels=wav_ch, dim=head_dim)
                scale_init = float(wavelet_scale_init)
                self.wavelet_scale = nn.Parameter(torch.tensor(scale_init))
                self.wavelet_warmup_steps = int(wavelet_warmup_steps) if wavelet_warmup_steps > 0 else 0
                self.wavelet_fuse_mode = str(wavelet_fuse_mode)
                self.register_buffer("_wavelet_step", torch.tensor(0, dtype=torch.long))
                self.register_buffer("_current_global_step", torch.tensor(-1, dtype=torch.long))
            else:
                self.dwt_for_residual = DWTPreprocessor(channels=stem_out_channels, dwt_fuse="add")
                wav_ch = self.dwt_for_residual.out_channels
                self.wavelet_residual = WaveletGlobalGate(in_channels=wav_ch, dim=head_dim)
                scale_init = float(wavelet_scale_init)
                self.wavelet_scale = nn.Parameter(torch.tensor(scale_init))
                self.wavelet_warmup_steps = int(wavelet_warmup_steps) if wavelet_warmup_steps > 0 else 0
                self.wavelet_fuse_mode = str(wavelet_fuse_mode)
                self.register_buffer("_wavelet_step", torch.tensor(0, dtype=torch.long))
                self.register_buffer("_current_global_step", torch.tensor(-1, dtype=torch.long))
        else:
            self.wavelet_residual = None
            self.wavelet_scale = None
            self.wavelet_warmup_steps = 0
            self.wavelet_fuse_mode = "add"

    
        g_h, g_w = self.patch_embed.seqlens
        assert g_h == g_w
        self.mixer_every = 2
        self.mixers = nn.ModuleList([ResidualDepthwiseMix(d_model=self.dim, grid=g_h) for _ in range((self.depth + self.mixer_every - 1)//self.mixer_every)])

    def load_state_dict(self, state_dict, strict=True):
        if "pos_embed.embed" in state_dict:
            old_pos_embed = state_dict["pos_embed.embed"]
            if old_pos_embed.shape != self.pos_embed.embed.shape:
                state_dict["pos_embed.embed"] = interpolate_sincos(embed=old_pos_embed, seqlens=self.pos_embed.seqlens)
        return super().load_state_dict(state_dict=state_dict, strict=strict)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = {"pos_embed.embed"}
        if self.side_token_beta is not None:
            nwd.add("side_token_beta")
        return nwd

    def get_wavelet_monitor_stats(self):
        return dict(self._wavelet_monitor_stats)
    
    def set_wavelet_global_step(self, global_step: int):
        """
        Set the current global_step used for wavelet warmup.

        Training code should call this once per optimization step.

        Args:
            global_step: Current training step (starting from 0).
        """
        for module in (self.pre_patch, self.post_stem):
            setter = getattr(module, "set_wavelet_global_step", None)
            if callable(setter):
                setter(global_step)
        if self.wavelet_residual is not None and self.wavelet_warmup_steps > 0:
            self._current_global_step.fill_(global_step)

    def forward(self, x):
        # Main branch
        x_raw = x
        x0 = self.pre_patch(x)
        stem_out = self.feature_extractor(x0)
        if self.wavelet_input_image and isinstance(self.post_stem, PostStemWaveletMerge):
            feature_maps = self.post_stem(stem_out, image=x_raw)
        else:
            feature_maps = self.post_stem(stem_out)
        monitor_stats = {}
        if hasattr(self.post_stem, "get_wavelet_monitor_stats"):
            for key, value in self.post_stem.get_wavelet_monitor_stats().items():
                monitor_stats[f"post_stem_{key}"] = float(value)
        x_main = self.patch_embed(feature_maps)
        if self.patch_embed_side is not None:
            side_maps_getter = getattr(self.post_stem, "get_side_feature", None)
            side_maps = side_maps_getter() if callable(side_maps_getter) else None
            if side_maps is not None:
                x_side = self.patch_embed_side(side_maps).to(dtype=x_main.dtype)
                side_delta = self.side_token_beta.to(dtype=x_main.dtype) * x_side
                x_main = x_main + side_delta
                monitor_stats.update(
                    {
                        "post_stem_side_token_abs_mean": _abs_mean(x_side),
                        "post_stem_side_token_delta_abs_mean": _abs_mean(side_delta),
                        "post_stem_side_token_beta": float(self.side_token_beta.detach().item()),
                    }
                )
            else:
                monitor_stats.update(
                    {
                        "post_stem_side_token_abs_mean": 0.0,
                        "post_stem_side_token_delta_abs_mean": 0.0,
                        "post_stem_side_token_beta": float(self.side_token_beta.detach().item()),
                    }
                )
        x_main = self.pos_embed(x_main)
        x_main = einops.rearrange(x_main, "b ... d -> b (...) d")

        midx = 0
        for i, block in enumerate(self.blocks):
            x_main = block(x_main)
            if (i % self.mixer_every) == 0:
                x_main = self.mixers[midx](x_main)
                midx += 1
        x_main = self.norm(x_main)

        if self.pooling is None:
            x_main = self.legacy_norm(x_main)
        elif self.pooling == "to_image":
            x_main = self.legacy_norm(x_main)
            seqlen_h, seqlen_w = self.patch_embed.seqlens
            x_main = einops.rearrange(
                x_main,
                "b (seqlen_h seqlen_w) dim -> b dim seqlen_h seqlen_w",
                seqlen_h=seqlen_h,
                seqlen_w=seqlen_w,
            )
        elif self.pooling == "global":
            # Global Average Pooling over tokens: [B, N, D] -> [B, D]
            x_main = x_main.mean(dim=1)
            x_main = self.legacy_norm(x_main)
        elif self.pooling == "bilateral_avg":
            x_main = (x_main[:, 0] + x_main[:, -1]) / 2
            x_main = self.legacy_norm(x_main)
        elif self.pooling == "bilateral_flatten":
            x_main = torch.concat([x_main[:, 0], x_main[:, -1]], dim=1)
            x_main = self.legacy_norm(x_main)
        elif self.pooling == "attn":
            assert self.attn_pool is not None, "attn_pool is not initialized (pooling='attn')"
            x_main = self.attn_pool(x_main)  # (B, dim)
            x_main = self.legacy_norm(x_main)
        else:
            raise NotImplementedError(f"pooling '{self.pooling}' is not implemented")

        if self.wavelet_residual is not None:
            x_main_pre_wavelet = x_main
            dwt_monitor = self.dwt_for_residual if self.dwt_for_residual is not None else self.post_stem.dwt
            residual_forward = getattr(dwt_monitor, "forward_residual", None)
            w = residual_forward(stem_out) if callable(residual_forward) else dwt_monitor(stem_out)
            if hasattr(dwt_monitor, "get_wavelet_monitor_stats"):
                for key, value in dwt_monitor.get_wavelet_monitor_stats().items():
                    monitor_stats[f"head_dwt_{key}"] = float(value)
            vec = self.wavelet_residual(w)
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
            gate_vec = torch.tanh(vec.to(x_main.dtype))
            gate_scale = effective_scale * gate_vec
            if self.wavelet_fuse_mode == "multiply":
                delta = x_main_pre_wavelet * gate_scale
                x_main = x_main_pre_wavelet + delta
            else:  # "add" - kept for backward compatibility
                delta = gate_scale
                x_main = x_main_pre_wavelet + delta
            head_input_abs = _abs_mean(x_main_pre_wavelet)
            head_delta_abs = _abs_mean(delta)
            monitor_stats.update(
                {
                    "head_input_abs_mean": head_input_abs,
                    "head_gate_abs_mean": _abs_mean(gate_vec),
                    "head_delta_abs_mean": head_delta_abs,
                    "head_delta_over_input": _safe_ratio(head_delta_abs, head_input_abs),
                    "head_effective_scale": float(effective_scale.detach().item()),
                }
            )
        else:
            monitor_stats.update(
                {
                    "head_input_abs_mean": _abs_mean(x_main),
                    "head_gate_abs_mean": 0.0,
                    "head_delta_abs_mean": 0.0,
                    "head_delta_over_input": 0.0,
                    "head_effective_scale": 0.0,
                }
            )
    
        # Optional feature branch + residual injection
        if (self.feature_extractor_branch is not None) and (self.head_adapter is not None):
            feature_branch_out = self.feature_extractor_branch(feature_maps)   # (B, dim)
            x_main = self.head_adapter(x_main, feature_branch_out)             # (B, head_dim)

        if self.head is None:
            self._wavelet_monitor_stats = monitor_stats
            return x_main
        combined_output = self.head(x_main)
        self._wavelet_monitor_stats = monitor_stats
        return combined_output
