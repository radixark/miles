"""Pure-torch port of the Muse Spark NaViT vision tower.

Mirrors ``sglang/srt/layers/muse_spark_navit.py`` (dev) and the vision head of
``MuseSparkForConditionalGeneration`` operator by operator, with plain ``nn.Linear``
modules named after the pre-release export (``vision_model.*``,
``vision_projection.weight``); ``MuseSparkVisionTower.checkpoint_name`` maps them to the
released checkpoint (``model.vision_tower.*``, ``model.vision_adapter.*``,
``model.vision_projection.weight``).

Chain: patches ``[N, T*C*P*P]`` -> conv1_linear -> + interpolated learned
position embedding -> ln_pre -> 50 pre-norm blocks (windowed attention over
32x32 patch tiles except every ``sparse_attention_factor``-th layer and the
last, which attend over the whole image; 2D RoPE inside attention) -> ln_post
-> 2x2 pixel shuffle (channel interleaved) -> adapter (c_fc, GELU, c_proj,
GELU) -> vision_projection -> weightless RMSNorm.
"""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class MuseSparkVisionConfig:
    hidden_size: int = 1536
    num_hidden_layers: int = 50
    num_attention_heads: int = 16
    num_channels: int = 3
    intermediate_size: int = 8960
    vision_output_dim: int = 6144
    image_size: int = 448
    patch_size: int = 14
    patch_temporal: int = 2
    downsample_factor: int = 2
    sparse_attention_factor: int = 4
    use_rope: bool = True
    rope_theta: float = 10000.0
    grid_sample_mode: str = "bilinear"
    norm_eps: float = 1e-5
    projector_input_dim: int = 4096
    llm_hidden_size: int = 8192
    normalize_embeddings: bool = True
    embedding_norm_eps: float = 1e-5
    max_num_image_tokens: int = 4096
    min_num_image_tokens: int = 1
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5)

    @classmethod
    def from_hf(cls, hf_config: dict) -> MuseSparkVisionConfig:
        """Tower config of the released checkpoint (``vision_config``, ``text_config`` and the projector sizes at the
        top level). What that config leaves out -- bilinear position interpolation, 0.5 mean/std, 1..4096 image
        tokens, a weightless RMSNorm on the projected rows -- is what SGLang's NaViT, processor and omni model
        hard-code, i.e. the dataclass defaults."""
        vision, text = hf_config["vision_config"], hf_config["text_config"]
        hidden, merge, patch = int(vision["hidden_size"]), int(vision["merge_size"]), int(vision["patch_size"])
        rope = vision.get("rope_parameters") or {}
        if vision.get("hidden_act", "gelu") != "gelu" or hf_config.get("projector_hidden_act", "gelu") != "gelu":
            raise ValueError("Muse Spark NaViT and its projector use gelu only")
        if rope.get("rope_type", "axial") != "axial":
            raise ValueError("Muse Spark NaViT uses axial RoPE only")
        if int(vision["pos_emb_height"]) != int(vision["pos_emb_width"]):
            raise ValueError("Muse Spark NaViT position grid must be square")
        if int(hf_config["out_hidden_size"]) != hidden * merge**2:
            raise ValueError("out_hidden_size must equal vision hidden_size * merge_size^2")
        config = cls(
            hidden_size=hidden,
            num_hidden_layers=int(vision["num_hidden_layers"]),
            num_attention_heads=int(vision["num_attention_heads"]),
            intermediate_size=int(vision["intermediate_size"]),
            vision_output_dim=hidden * merge**2,
            image_size=int(vision["pos_emb_height"]) * patch,
            patch_size=patch,
            patch_temporal=int(vision["patch_temporal"]),
            downsample_factor=merge,
            rope_theta=float(rope.get("rope_theta", 10000.0)),
            norm_eps=float(vision.get("layer_norm_eps", 1e-5)),
            projector_input_dim=int(hf_config["projector_hidden_size"]),
            llm_hidden_size=int(text["hidden_size"]),
        )
        # layer_types lists window/full attention per block; NaViT expresses it as a sparse_attention_factor.
        full = [kind == "full_attention" for kind in vision["layer_types"]]
        if len(full) != config.num_hidden_layers or set(vision["layer_types"]) - {
            "window_attention",
            "full_attention",
        }:
            raise ValueError("vision layer_types must name window_attention/full_attention for every block")
        for factor in range(1, config.num_hidden_layers + 1):
            candidate = replace(config, sparse_attention_factor=factor)
            if full == [candidate.uses_full_attention(index) for index in range(config.num_hidden_layers)]:
                return candidate
        raise ValueError("vision layer_types is not a NaViT window/full attention pattern")

    def __post_init__(self) -> None:
        if self.vision_output_dim // self.downsample_factor**2 != self.hidden_size:
            raise ValueError("vision_output_dim / downsample_factor^2 must equal hidden_size")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("vision hidden size must be divisible by attention heads")
        if (self.hidden_size // self.num_attention_heads) % 4:
            raise ValueError("NaViT attention head size must be divisible by four")

    def uses_full_attention(self, index: int) -> bool:
        """Every ``sparse_attention_factor``-th block and the last attend over the whole image."""
        return (
            self.sparse_attention_factor <= 1
            or index == self.num_hidden_layers - 1
            or (index + 1) % self.sparse_attention_factor == 0
        )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def patch_input_size(self) -> int:
        return self.num_channels * self.patch_size * self.patch_size * self.patch_temporal

    @property
    def pos_grid(self) -> int:
        return self.image_size // self.patch_size

    @property
    def token_patch_size(self) -> int:
        return self.patch_size * self.downsample_factor


# --------------------------------------------------------------------------
# pure helpers (same math as the SGLang module)
# --------------------------------------------------------------------------


def patchify_clip(clip: torch.Tensor, patch_size: int, patch_temporal: int) -> tuple[torch.Tensor, int, int]:
    """``[3,H,W]`` or ``[T,3,H,W]`` -> ``([gh*gw, T*C*P*P], gh, gw)`` in NaViT patch order."""
    if clip.ndim == 3:
        clip = clip.unsqueeze(0)
    if clip.shape[0] == 1:
        clip = clip.expand(patch_temporal, -1, -1, -1)
    if clip.shape[0] != patch_temporal:
        raise ValueError("each NaViT clip must contain patch_temporal frames")
    _, channels, height, width = clip.shape
    if height % patch_size or width % patch_size:
        raise ValueError("vision inputs must be aligned to the patch size")
    grid_h, grid_w = height // patch_size, width // patch_size
    patches = (
        clip.reshape(patch_temporal, channels, grid_h, patch_size, grid_w, patch_size)
        .permute(2, 4, 0, 1, 3, 5)
        .reshape(grid_h * grid_w, -1)
    )
    return patches, grid_h, grid_w


def rope_frequencies(head_dim: int, grid_h: int, grid_w: int, theta: float, device) -> torch.Tensor:
    """``[gh*gw, head_dim//2, 2]`` of (cos, sin); width angles first, 1-based positions."""
    half_dim = head_dim // 2
    exponents = torch.arange(0, half_dim, 2, dtype=torch.float32, device=device)
    inverse = theta ** (-exponents / half_dim)
    pos_h = torch.arange(1, grid_h + 1, dtype=torch.float32, device=device)
    pos_w = torch.arange(1, grid_w + 1, dtype=torch.float32, device=device)
    mesh_h, mesh_w = torch.meshgrid(pos_h, pos_w, indexing="ij")
    freq_h = torch.einsum("hw,d->hwd", mesh_h, inverse)
    freq_w = torch.einsum("hw,d->hwd", mesh_w, inverse)
    angles = torch.cat((freq_w, freq_h), dim=-1)
    return torch.stack((angles.cos(), angles.sin()), dim=-1).reshape(-1, half_dim, 2)


def patch_grid_positions(grid_h: int, grid_w: int, device) -> torch.Tensor:
    """fp32 grid_sample coordinates of patch centers in patch order, ``[gh*gw, 2]``, for the checkpoint's xy
    (non-ij) position-table layout (SGLang #223)."""
    pos_h = torch.linspace(-1 + 1 / grid_h, 1 - 1 / grid_h, grid_h, device=device, dtype=torch.float32)
    pos_w = torch.linspace(-1 + 1 / grid_w, 1 - 1 / grid_w, grid_w, device=device, dtype=torch.float32)
    return torch.stack(torch.meshgrid(pos_h, pos_w, indexing="xy"), dim=-1).reshape(-1, 2)


def apply_rope(query: torch.Tensor, key: torch.Tensor, frequencies: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleaved-pair rotation in fp32; ``frequencies`` is ``[N, head_dim//2, 2]``."""
    cosine = frequencies[..., 0].unsqueeze(1)
    sine = frequencies[..., 1].unsqueeze(1)

    def rotate(value: torch.Tensor) -> torch.Tensor:
        pairs = value.float().reshape(*value.shape[:-1], -1, 2)
        first, second = pairs.unbind(dim=-1)
        rotated = torch.stack((first * cosine - second * sine, first * sine + second * cosine), dim=-1).flatten(-2)
        return rotated.to(value.dtype)

    return rotate(query), rotate(key)


def pixel_shuffle_permutation(grid_h: int, grid_w: int, factor: int, device) -> torch.Tensor:
    if grid_h % factor or grid_w % factor:
        raise ValueError("preprocessed image grid is not divisible by downsample factor")
    order = torch.arange(grid_h * grid_w, device=device)
    return order.reshape(grid_h // factor, factor, grid_w // factor, factor).permute(0, 2, 1, 3).reshape(-1)


def sparse_permutation(grid_h: int, grid_w: int, block_h: int, block_w: int, device) -> tuple[torch.Tensor, list[int]]:
    """Tile the patch grid into ``block_h x block_w`` windows; returns (order, per-window lengths)."""
    padded_h = math.ceil(grid_h / block_h) * block_h
    padded_w = math.ceil(grid_w / block_w) * block_w
    order = torch.arange(grid_h * grid_w, device=device).reshape(grid_h, grid_w)
    order = F.pad(order, (0, padded_w - grid_w, 0, padded_h - grid_h), value=-1)
    rows, cols = padded_h // block_h, padded_w // block_w
    order = order.reshape(rows, block_h, cols, block_w).permute(0, 2, 1, 3).reshape(rows * cols, block_h * block_w)
    lengths = (order != -1).sum(dim=1).tolist()
    return order[order != -1], lengths


def cu_seqlens_of(lengths: Sequence[int], device) -> torch.Tensor:
    cu = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    if lengths:
        cu[1:] = torch.tensor(lengths, dtype=torch.int32).cumsum(0)
    return cu.to(device)


def find_resize_target(
    image_h: int, image_w: int, patch_size: int, minimum_tokens: int, maximum_tokens: int
) -> tuple[int, int, int]:
    """Line-for-line port of the omni processor's resize policy: (target_h, target_w, token_count).

    ``patch_size`` is the LLM-token patch (vision patch * downsample factor, 28 px); both target
    sides are multiples of it. Kept identical to SGLang, including the set-based candidate
    ordering that breaks aspect ties.
    """
    ideal_h = image_h / patch_size
    ideal_w = image_w / patch_size
    ideal_count = ideal_h * ideal_w
    aspect = ideal_w / ideal_h
    if ideal_count > maximum_tokens:
        ideal_h = math.sqrt(maximum_tokens / aspect)
        ideal_w = ideal_h * aspect
    elif ideal_count < minimum_tokens:
        ideal_h = math.sqrt(minimum_tokens / aspect)
        ideal_w = ideal_h * aspect
    try:
        patches_h, patches_w = _least_distortion_fit(
            ideal_h, ideal_w, image_h, image_w, minimum_tokens, maximum_tokens
        )
    except ValueError:
        patches_h, patches_w = _least_distance_fit(ideal_h, ideal_w, minimum_tokens, maximum_tokens)
    return patches_h * patch_size, patches_w * patch_size, patches_h * patches_w


def _least_distortion_fit(ideal_h, ideal_w, image_h, image_w, minimum, maximum):
    candidates = set(
        itertools.product((math.floor(ideal_h), math.ceil(ideal_h)), (math.floor(ideal_w), math.ceil(ideal_w)))
    )
    candidates = {
        (height, width)
        for height, width in candidates
        if height > 0 and width > 0 and minimum <= height * width <= maximum
    }
    if not candidates:
        raise ValueError("no rounded resize target satisfies the token budget")
    aspect = image_h / image_w
    return min(candidates, key=lambda item: abs(item[0] / item[1] - aspect))


def _least_distance_fit(ideal_h, ideal_w, minimum, maximum):
    best = None
    for height in range(1, maximum + 1):
        lower = math.ceil(minimum / height)
        upper = math.floor(maximum / height)
        if lower > upper:
            continue
        width = min(max(round(ideal_w), lower), upper)
        distance = (height - ideal_h) ** 2 + (width - ideal_w) ** 2
        if best is None or distance < best[0]:
            best = (distance, height, width)
    if best is None:
        raise ValueError("no resize target satisfies the token budget")
    return best[1], best[2]


# --------------------------------------------------------------------------
# modules
# --------------------------------------------------------------------------


class NaViTAttention(nn.Module):
    """Separate q/k/v/out projections (checkpoint layout); varlen attention via SDPA per sequence
    or, when available, the flash varlen kernel."""

    def __init__(self, config: MuseSparkVisionConfig, attn_impl: str = "sdpa"):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.use_rope = config.use_rope
        self.attn_impl = attn_impl
        hidden = config.hidden_size
        self.q_proj = nn.Linear(hidden, hidden, bias=True)
        self.k_proj = nn.Linear(hidden, hidden, bias=True)
        self.v_proj = nn.Linear(hidden, hidden, bias=True)
        self.out_proj = nn.Linear(hidden, hidden, bias=True)

    def forward(
        self, value: torch.Tensor, frequencies: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int
    ) -> torch.Tensor:
        tokens = value.shape[0]
        q = self.q_proj(value).view(tokens, self.num_heads, self.head_dim)
        k = self.k_proj(value).view(tokens, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(tokens, self.num_heads, self.head_dim)
        if self.use_rope:
            q, k = apply_rope(q, k, frequencies)
        if self.attn_impl == "flash":
            from flash_attn import flash_attn_varlen_func

            out = flash_attn_varlen_func(
                q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, softmax_scale=self.scale, causal=False
            )
        else:
            out = torch.empty_like(q)
            bounds = cu_seqlens.tolist()
            for start, end in zip(bounds[:-1], bounds[1:], strict=True):
                if end <= start:
                    continue
                # [1, heads, len, head_dim]
                qs, ks, vs = (t[start:end].transpose(0, 1).unsqueeze(0) for t in (q, k, v))
                out[start:end] = F.scaled_dot_product_attention(qs, ks, vs, scale=self.scale)[0].transpose(0, 1)
        return self.out_proj(out.reshape(tokens, -1))


class NaViTMLP(nn.Module):
    def __init__(self, config: MuseSparkVisionConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.c_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(value)))


class NaViTBlock(nn.Module):
    def __init__(self, config: MuseSparkVisionConfig, attn_impl: str):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = NaViTAttention(config, attn_impl)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = NaViTMLP(config)

    def forward(self, value, frequencies, cu_seqlens, max_seqlen):
        value = value + self.attn(self.ln_1(value), frequencies, cu_seqlens, max_seqlen)
        return value + self.mlp(self.ln_2(value))


class NaViTEncoder(nn.Module):
    def __init__(self, config: MuseSparkVisionConfig, attn_impl: str):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(NaViTBlock(config, attn_impl) for _ in range(config.num_hidden_layers))

    def uses_full_attention(self, index: int) -> bool:
        return self.config.uses_full_attention(index)

    def forward(self, value, frequencies, full_cu, full_max, sparse_cu, sparse_max):
        for index, layer in enumerate(self.layers):
            if self.uses_full_attention(index):
                value = layer(value, frequencies, full_cu, full_max)
            else:
                value = layer(value, frequencies, sparse_cu, sparse_max)
        return value


class NaViTAdapter(nn.Module):
    def __init__(self, config: MuseSparkVisionConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.vision_output_dim, config.projector_input_dim, bias=False)
        self.c_proj = nn.Linear(config.projector_input_dim, config.projector_input_dim, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.c_proj(F.gelu(self.c_fc(value))))


class NaViT(nn.Module):
    """``vision_model.*`` of the checkpoint: patches in, adapter features out."""

    def __init__(self, config: MuseSparkVisionConfig, attn_impl: str = "sdpa"):
        super().__init__()
        self.config = config
        self.conv1_linear = nn.Linear(config.patch_input_size, config.hidden_size, bias=False)
        self.positional_embedding_vlm = nn.Parameter(
            torch.empty(config.pos_grid * config.pos_grid, config.hidden_size)
        )
        self.ln_pre = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.encoder = NaViTEncoder(config, attn_impl)
        self.ln_post = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.adapter = NaViTAdapter(config)

    def forward_patches(self, patches: torch.Tensor, grids: Sequence[tuple[int, int]]) -> torch.Tensor:
        """``patches`` ``[sum(gh*gw), T*C*P*P]`` in per-image patch order; returns ``[sum(gh*gw/f^2), projector_input_dim]``."""
        config = self.config
        device = patches.device
        param_dtype = self.conv1_linear.weight.dtype
        patches = patches.to(param_dtype)
        positions, frequencies, full_lengths, sparse_lengths, sparse_orders, downsample_orders = [], [], [], [], [], []
        offset = 0
        for grid_h, grid_w in grids:
            positions.append(patch_grid_positions(grid_h, grid_w, device))
            frequencies.append(rope_frequencies(config.head_dim, grid_h, grid_w, config.rope_theta, device))
            downsample_orders.append(
                pixel_shuffle_permutation(grid_h, grid_w, config.downsample_factor, device) + offset
            )
            if config.sparse_attention_factor > 1:
                order, lengths = sparse_permutation(grid_h, grid_w, config.pos_grid, config.pos_grid, device)
                sparse_orders.append(order + offset)
                sparse_lengths.extend(lengths)
            else:
                sparse_lengths.append(grid_h * grid_w)
            full_lengths.append(grid_h * grid_w)
            offset += grid_h * grid_w
        if offset != patches.shape[0]:
            raise ValueError(f"grids cover {offset} patches but {patches.shape[0]} rows were given")
        positions = torch.cat(positions)
        frequencies = torch.cat(frequencies)
        downsample_order = torch.cat(downsample_orders)

        value = self.conv1_linear(patches)
        # The table is sampled in fp32 and added in the activation dtype, as SGLang does.
        position_grid = (
            self.positional_embedding_vlm.float()
            .reshape(config.pos_grid, config.pos_grid, -1)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        interpolated = F.grid_sample(
            position_grid, positions[None, None], mode=config.grid_sample_mode, align_corners=False
        )[0, :, 0].transpose(0, 1)
        value = value + interpolated.to(value.dtype)
        value = self.ln_pre(value)

        full_cu = cu_seqlens_of(full_lengths, device)
        full_max = max(full_lengths)
        if sparse_orders:
            sparse_order = torch.cat(sparse_orders)
            value = value[sparse_order]
            frequencies = frequencies[sparse_order]
            sparse_cu, sparse_max = cu_seqlens_of(sparse_lengths, device), max(sparse_lengths)
        else:
            sparse_order, sparse_cu, sparse_max = None, full_cu, full_max
        value = self.encoder(value, frequencies, full_cu, full_max, sparse_cu, sparse_max)
        if sparse_order is not None:
            restored = torch.empty_like(value)
            restored.index_copy_(0, sparse_order, value)
            value = restored
        value = self.ln_post(value)

        value = value[downsample_order]
        factor_squared = config.downsample_factor**2
        value = (
            value.reshape(-1, factor_squared, value.shape[-1])
            .transpose(1, 2)
            .reshape(value.shape[0] // factor_squared, -1)
        )
        return self.adapter(value)

    def forward(self, clips: Sequence[torch.Tensor]) -> torch.Tensor:
        """Convenience path for whole images ``[3,H,W]`` (or ``[T,3,H,W]``): patchify then ``forward_patches``."""
        patches, grids = [], []
        for clip in clips:
            rows, grid_h, grid_w = patchify_clip(
                clip.to(self.conv1_linear.weight.dtype), self.config.patch_size, self.config.patch_temporal
            )
            patches.append(rows)
            grids.append((grid_h, grid_w))
        return self.forward_patches(torch.cat(patches), grids)


class RMSNormWithoutScale(nn.Module):
    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(value, (value.shape[-1],), weight=None, eps=self.eps)


class MuseSparkVisionTower(nn.Module):
    """The omni model's whole vision head: NaViT -> vision_projection -> perception_emb_norm.

    Output rows are LLM-width image embeddings (before the text model's ``m_emb`` multiplier,
    which the model wrapper applies to text and image rows alike).
    """

    def __init__(self, config: MuseSparkVisionConfig, attn_impl: str = "sdpa"):
        super().__init__()
        self.config = config
        self.vision_model = NaViT(config, attn_impl)
        self.vision_projection = nn.Linear(config.projector_input_dim, config.llm_hidden_size, bias=False)
        self.perception_emb_norm = (
            RMSNormWithoutScale(config.embedding_norm_eps) if config.normalize_embeddings else None
        )

    def _head(self, features: torch.Tensor) -> torch.Tensor:
        output = self.vision_projection(features)
        return self.perception_emb_norm(output) if self.perception_emb_norm is not None else output

    def forward_patches(self, patches: torch.Tensor, grids: Sequence[tuple[int, int]]) -> torch.Tensor:
        return self._head(self.vision_model.forward_patches(patches, grids))

    def forward(self, clips: Sequence[torch.Tensor]) -> torch.Tensor:
        return self._head(self.vision_model(clips))

    @staticmethod
    def checkpoint_name(state_key: str) -> str:
        """``state_dict`` key -> tensor name in the released checkpoint (also what the engine loads in a weight update).

        The checkpoint stores the q/k projection rows of every head as two consecutive rotary halves; this module
        wants them interleaved (``weight_mapping.rope_pairs_hf_to_kernel``, like SGLang's loader).
        """
        for pattern, replacement in _CHECKPOINT_NAMES:
            match = pattern.fullmatch(state_key)
            if match:
                return match.expand(replacement)
        raise KeyError(f"{state_key!r} is not a Muse Spark vision tower parameter")


_LAYER = r"vision_model\.encoder\.layers\.(\d+)\."
_CHECKPOINT_NAMES = tuple(
    (re.compile(pattern), replacement)
    for pattern, replacement in (
        (r"vision_model\.conv1_linear\.weight", "model.vision_tower.patch_embedder.patch_embedding.weight"),
        (
            r"vision_model\.positional_embedding_vlm",
            "model.vision_tower.patch_embedder.position_embedding_table.weight",
        ),
        (r"vision_model\.(ln_pre|ln_post)\.(weight|bias)", r"model.vision_tower.\1.\2"),
        (_LAYER + r"ln_1\.(weight|bias)", r"model.vision_tower.layers.\1.norm1.\2"),
        (_LAYER + r"ln_2\.(weight|bias)", r"model.vision_tower.layers.\1.norm2.\2"),
        (_LAYER + r"attn\.(q|k|v)_proj\.(weight|bias)", r"model.vision_tower.layers.\1.attn.\2_proj.\3"),
        (_LAYER + r"attn\.out_proj\.(weight|bias)", r"model.vision_tower.layers.\1.attn.proj.\2"),
        (_LAYER + r"mlp\.c_fc\.(weight|bias)", r"model.vision_tower.layers.\1.mlp.fc1.\2"),
        (_LAYER + r"mlp\.c_proj\.(weight|bias)", r"model.vision_tower.layers.\1.mlp.fc2.\2"),
        (r"vision_model\.adapter\.c_fc\.weight", "model.vision_adapter.fc1.weight"),
        (r"vision_model\.adapter\.c_proj\.weight", "model.vision_adapter.fc2.weight"),
        (r"vision_projection\.weight", "model.vision_projection.weight"),
    )
)
