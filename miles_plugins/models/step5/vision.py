"""Frozen PerceptionEncoder and text projector for Step5 training.

Follows SGLang's `step3_vl_10b.PerceptionEncoder` with BF16 weights and FP32 2D RoPE
angles. Checkpoint names are matched strictly; crops precede each image's overview.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import Tensor, nn

_TOWER_PREFIXES = ("vision_model.", "vit_large_projector.")
# SGLang's PerceptionEncoder fixes these instead of reading them from the config.
_LAYER_NORM_EPS = 1e-5
_ROPE_THETA = 10000.0


@dataclass(frozen=True)
class Step5VisionConfig:
    """The tower's dimensions; its architecture is fixed."""

    width: int
    layers: int
    heads: int
    patch_size: int
    image_size: int
    mlp_width: int
    text_hidden_size: int

    @classmethod
    def from_hf(cls, hf_config: Any) -> "Step5VisionConfig":
        """From a ``step5_vl`` config (SGLang's ``Step5VLConfig``), after checking it describes this tower."""
        vision = hf_config.vision_config
        if (vision.hidden_act, vision.use_rope2d) != ("quick_gelu", True):
            raise ValueError(
                f"Step5's vision tower is a QuickGELU ViT with 2D RoPE, got hidden_act={vision.hidden_act!r} "
                f"and use_rope2d={vision.use_rope2d!r}"
            )
        return cls(
            width=vision.width,
            layers=vision.layers,
            heads=vision.heads,
            patch_size=vision.patch_size,
            image_size=vision.image_size,
            # The config holds the MLP width as the ratio 8960 / 1536; this is SGLang's arithmetic.
            mlp_width=int(vision.width * vision.mlp_ratio),
            text_hidden_size=hf_config.text_config.hidden_size,
        )


class Step5VisionEncoder(nn.Module):
    """The vision tower and projector: normalized pixels in, text-width image tokens out."""

    def __init__(self, config: Step5VisionConfig) -> None:
        super().__init__()
        self.vision_model = _PerceptionEncoder(config)
        self.vit_large_projector = nn.Linear(4 * config.width, config.text_hidden_size, bias=False)

    @torch.no_grad()
    def features(self, pixel_values: Tensor, patch_pixel_values: Tensor, num_patches: Tensor) -> Tensor:
        """Image tokens ``[n_image_tokens, text_hidden_size]``: per image, its crops' tokens, then its overview's.

        ``pixel_values`` holds one overview per image, ``patch_pixel_values`` every image's crops in
        image order (possibly none), and ``num_patches`` each image's crop count.
        """
        crop_counts = num_patches.tolist()
        assert len(crop_counts) == len(pixel_values) and sum(crop_counts) == len(patch_pixel_values), (
            f"num_patches {crop_counts} does not match {len(pixel_values)} overview(s) "
            f"and {len(patch_pixel_values)} crop(s)"
        )
        overviews = self._encode(pixel_values)
        crops = self._encode(patch_pixel_values)
        order = feature_order(crop_counts, overview_tokens=overviews.shape[1], crop_tokens=crops.shape[1])
        tokens = torch.cat([overviews.flatten(0, 1), crops.flatten(0, 1)])
        return tokens[torch.tensor(order, dtype=torch.long, device=tokens.device)]

    def _encode(self, pixels: Tensor) -> Tensor:
        # contiguous as in SGLang: matmul runs a transposed input as a batched GEMM, which rounds differently
        return self.vit_large_projector(self.vision_model(pixels).contiguous())


def feature_order(crop_counts: list[int], *, overview_tokens: int, crop_tokens: int) -> list[int]:
    """Rows of ``cat([all overview tokens, all crop tokens])`` in feature order.

    Feature order is each image's crops (in order), then its overview, image after image; it is
    the order of the ``<im_patch>`` tokens in the prompt.
    """
    crops_start = len(crop_counts) * overview_tokens
    order: list[int] = []
    crops_before = 0
    for image, count in enumerate(crop_counts):
        order += range(crops_start + crops_before * crop_tokens, crops_start + (crops_before + count) * crop_tokens)
        order += range(image * overview_tokens, (image + 1) * overview_tokens)
        crops_before += count
    return order


def load_step5_vision_encoder(
    hf_checkpoint: str, config: Step5VisionConfig, *, device: torch.device
) -> Step5VisionEncoder:
    """The frozen tower on ``device``, holding the checkpoint's BF16 weights, matched to it strictly by name."""
    # Built on meta and filled by assignment, so no weight is initialized or allocated twice.
    with torch.device("meta"):
        encoder = Step5VisionEncoder(config)
    encoder.load_state_dict(_read_tower_weights(hf_checkpoint, device=device), strict=True, assign=True)
    return encoder.requires_grad_(False).eval()


def _read_tower_weights(hf_checkpoint: str, *, device: torch.device) -> dict[str, Tensor]:
    checkpoint = Path(hf_checkpoint)
    weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    names_by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if name.startswith(_TOWER_PREFIXES):
            names_by_shard.setdefault(shard, []).append(name)
    weights = {}
    for shard, names in sorted(names_by_shard.items()):
        with safe_open(str(checkpoint / shard), framework="pt", device=str(device)) as f:
            weights |= {name: f.get_tensor(name) for name in names}
    not_bf16 = sorted(name for name, weight in weights.items() if weight.dtype != torch.bfloat16)
    if not_bf16:
        raise ValueError(f"Step5 vision weights must be BF16, but these are not: {not_bf16[:8]}")
    return weights


class _PerceptionEncoder(nn.Module):
    """The ViT and its downsampling convolutions (checkpoint prefix ``vision_model.``)."""

    def __init__(self, config: Step5VisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.native_grid = config.image_size // config.patch_size
        self.width = config.width
        self.head_dim = config.width // config.heads
        self.conv1 = nn.Conv2d(3, config.width, kernel_size=config.patch_size, stride=config.patch_size, bias=False)
        self.positional_embedding = nn.Parameter(torch.empty(self.native_grid**2, config.width))
        self.ln_pre = nn.LayerNorm(config.width, eps=_LAYER_NORM_EPS)
        self.transformer = _Transformer(config)
        self.vit_downsampler1 = nn.Conv2d(config.width, 2 * config.width, kernel_size=3, stride=2, padding=1)
        self.vit_downsampler2 = nn.Conv2d(2 * config.width, 4 * config.width, kernel_size=3, stride=2, padding=1)

    def forward(self, pixels: Tensor) -> Tensor:
        """``[images, 3, height, width]`` -> ``[images, tokens, 4 * width]``, each conv halving the grid (rounding up)."""
        # sizes spelled out, not -1: a batch of no crops has zero elements
        batch, _, height, width = pixels.shape
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        patches = self.conv1(pixels.to(self.conv1.weight.dtype))
        hidden = patches.permute(0, 2, 3, 1).reshape(batch, grid_h * grid_w, self.width)
        hidden = self.ln_pre(hidden + self._position_embedding(grid_h, grid_w))
        angles = rope_angles(grid_h, grid_w, self.head_dim, device=hidden.device)
        hidden = self.transformer(hidden, cos=angles.cos(), sin=angles.sin())
        grid = hidden.transpose(1, 2).reshape(batch, self.width, grid_h, grid_w)
        return self.vit_downsampler2(self.vit_downsampler1(grid)).flatten(2).transpose(1, 2)

    def _position_embedding(self, grid_h: int, grid_w: int) -> Tensor:
        """The native grid's embedding resized to ``grid_h x grid_w`` (exactly itself at the native size)."""
        native = self.positional_embedding.reshape(1, self.native_grid, self.native_grid, -1)
        resized = F.interpolate(
            native.permute(0, 3, 1, 2).contiguous(), size=(grid_h, grid_w), mode="bilinear", align_corners=False
        )
        return resized.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, -1)


class _Transformer(nn.Module):
    def __init__(self, config: Step5VisionConfig) -> None:
        super().__init__()
        self.resblocks = nn.ModuleList(_Block(config) for _ in range(config.layers))

    def forward(self, hidden: Tensor, *, cos: Tensor, sin: Tensor) -> Tensor:
        for block in self.resblocks:
            hidden = block(hidden, cos=cos, sin=sin)
        return hidden


class _Block(nn.Module):
    def __init__(self, config: Step5VisionConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.width, eps=_LAYER_NORM_EPS)
        self.attn = _Attention(config.width, heads=config.heads)
        self.ls_1 = _LayerScale(config.width)
        self.ln_2 = nn.LayerNorm(config.width, eps=_LAYER_NORM_EPS)
        self.mlp = _Mlp(config.width, mlp_width=config.mlp_width)
        self.ls_2 = _LayerScale(config.width)

    def forward(self, hidden: Tensor, *, cos: Tensor, sin: Tensor) -> Tensor:
        hidden = hidden + self.ls_1(self.attn(self.ln_1(hidden), cos=cos, sin=sin))
        return hidden + self.ls_2(self.mlp(self.ln_2(hidden)))


class _Attention(nn.Module):
    def __init__(self, width: int, *, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * width, width))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * width))
        self.out_proj = nn.Linear(width, width)

    def forward(self, hidden: Tensor, *, cos: Tensor, sin: Tensor) -> Tensor:
        batch, seq, width = hidden.shape
        qkv = F.linear(hidden, self.in_proj_weight, self.in_proj_bias).view(batch, seq, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)  # each [batch, heads, seq, head_dim]
        query, key = _rotate(query, cos=cos, sin=sin), _rotate(key, cos=cos, sin=sin)
        attended = F.scaled_dot_product_attention(query, key, value, scale=self.head_dim**-0.5)
        return self.out_proj(attended.transpose(1, 2).reshape(batch, seq, width))


class _Mlp(nn.Module):
    def __init__(self, width: int, *, mlp_width: int) -> None:
        super().__init__()
        self.c_fc = nn.Linear(width, mlp_width)
        self.c_proj = nn.Linear(mlp_width, width)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.c_proj(quick_gelu(self.c_fc(hidden)))


class _LayerScale(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(width))

    def forward(self, hidden: Tensor) -> Tensor:
        return hidden * self.gamma


def quick_gelu(x: Tensor) -> Tensor:
    """Transformers' ``QuickGELUActivation``, which SGLang's tower applies in BF16."""
    return x * torch.sigmoid(1.702 * x)


def rope_angles(grid_h: int, grid_w: int, head_dim: int, *, device: torch.device) -> Tensor:
    """fp32 angles ``[grid_h * grid_w, head_dim]`` for row-major patches: the column's, then the row's.

    Each half is a ``head_dim // 2``-wide 1D RoPE whose frequencies each cover an adjacent channel
    pair. These are SGLang's ``PerceptionEncoderRope2D`` expressions, evaluated on the tower's device
    as SGLang builds its table, so the angles agree bit for bit; ``pow`` rounds differently on CPU.
    SGLang looks a smaller grid up in the 52x52 table, which yields these same values.
    """
    half = head_dim // 2
    inv_freq = 1.0 / (_ROPE_THETA ** (torch.arange(0, half, 2, device=device)[: half // 2].float() / half))

    def axis_angles(length: int) -> Tensor:
        positions = torch.arange(length, dtype=torch.float, device=device)
        return torch.einsum("..., f -> ... f", positions, inv_freq).repeat_interleave(2, dim=-1)

    columns = axis_angles(grid_w)[None, :].expand(grid_h, grid_w, half)
    rows = axis_angles(grid_h)[:, None].expand(grid_h, grid_w, half)
    return torch.cat([columns, rows], dim=-1).reshape(grid_h * grid_w, head_dim)


def rotate_half(x: Tensor) -> Tensor:
    """``(x0, x1) -> (-x1, x0)`` over adjacent channel pairs."""
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _rotate(heads: Tensor, *, cos: Tensor, sin: Tensor) -> Tensor:
    """``heads * cos + rotate_half(heads) * sin`` in fp32, rounded once (SGLang's ``apply_rotary_emb``)."""
    return (heads * cos + rotate_half(heads) * sin).to(heads.dtype)
