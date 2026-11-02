"""Pure-torch port of the Qwen3.5 vision tower, computed the way SGLang's ``Qwen3VLMoeVisionModel`` computes it.

The tower is frozen and inference-only. Each step follows SGLang's serving path on CUDA (Hopper), so the trainer
merges the vision features the rollout engines saw:

- the patch embedding is ``Conv3dLayer``'s unfold + one GEMM, not cuDNN's Conv3d;
- the position-embedding table is interpolated with SGLang's bilinear weights, scaled and summed in the table dtype;
- the rotary cos/sin table is computed in fp32 on the GPU, as ``RotaryEmbedding`` builds its cache;
- attention applies SGLang's ``apply_rotary_pos_emb`` and runs on SGLang's FA3 varlen kernel.

Parameter names are the HF ones under ``model.visual.``, so the checkpoint loads unchanged.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN

from miles.backends.training_utils.weight_update.hf_weight_iterator.checkpoint_towers import load_checkpoint_towers


@dataclass(frozen=True)
class Qwen3_5VisionConfig:
    depth: int
    hidden_size: int
    num_heads: int
    intermediate_size: int
    hidden_act: str
    in_channels: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    out_hidden_size: int
    num_position_embeddings: int

    @classmethod
    def from_hf(cls, vision_config: dict) -> "Qwen3_5VisionConfig":
        if vision_config.get("deepstack_visual_indexes"):
            raise ValueError("the Qwen3.5 vision tower port has no deepstack mergers")
        return cls(**{name: vision_config[name] for name in cls.__dataclass_fields__})

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


class _PatchEmbed(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.config = config
        kernel = (config.temporal_patch_size, config.patch_size, config.patch_size)
        # Held as nn.Conv3d for the checkpoint layout; forward runs it as SGLang's Conv3dLayer does.
        self.proj = nn.Conv3d(config.in_channels, config.hidden_size, kernel, stride=kernel, bias=True)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        c = self.config
        x = pixel_values.view(-1, c.in_channels, c.temporal_patch_size, c.patch_size, c.patch_size)
        x = x.to(self.proj.weight.dtype)
        k1, k2, k3 = self.proj.kernel_size
        x = x.unfold(2, k1, k1).unfold(3, k2, k2).unfold(4, k3, k3)
        n, d, h, w = x.shape[0], x.shape[2], x.shape[3], x.shape[4]
        x = x.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(n, d, h, w, -1)
        x = F.linear(x, self.proj.weight.reshape(self.proj.out_channels, -1), self.proj.bias)
        return x.permute(0, 4, 1, 2, 3).reshape(-1, c.hidden_size)


class _Attention(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(
        self, x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        # Deferred: SGLang's kernels are only importable where SGLang is installed.
        from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func
        from sglang.srt.layers.rotary_embedding import apply_rotary_pos_emb

        q, k, v = self.qkv(x).split(self.num_heads * self.head_dim, dim=-1)
        q = q.reshape(-1, self.num_heads, self.head_dim)
        k = k.reshape(-1, self.num_heads, self.head_dim)
        v = v.reshape(-1, self.num_heads, self.head_dim)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        out = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=None,
            window_size=(-1, -1),
        )
        return self.proj(out.reshape(x.shape[0], -1))


class _MLP(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act(self.linear_fc1(x)))


class _Block(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = _Attention(config)
        self.mlp = _MLP(config)

    def forward(self, x, cu_seqlens, max_seqlen, cos, sin):
        x = x + self.attn(self.norm1(x), cu_seqlens, max_seqlen, cos, sin)
        return x + self.mlp(self.norm2(x))


class _Merger(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.merged_size = config.hidden_size * config.spatial_merge_size**2
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.merged_size, self.merged_size, bias=True)
        self.linear_fc2 = nn.Linear(self.merged_size, config.out_hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x).view(-1, self.merged_size)
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


class Qwen3_5VisionTower(nn.Module):
    """``model.visual.*`` of a Qwen3.5 checkpoint: patches in, merged token embeddings out."""

    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.config = config
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        self.patch_embed = _PatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.blocks = nn.ModuleList(_Block(config) for _ in range(config.depth))
        self.merger = _Merger(config)

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Embed every image or video of a batch.

        Args:
            pixel_values: [num_patches, patch_dim] patches of every item, in item order.
            grid_thw: [num_items, 3] patch grid of every item.

        Returns:
            [num_patches / spatial_merge_size**2, out_hidden_size] embeddings, one row per placeholder token.
        """
        grids = [tuple(int(v) for v in row) for row in grid_thw.tolist()]
        x = self.patch_embed(pixel_values.to(self.pos_embed.weight.device))
        x = x + self._interpolated_pos_embed(grids)
        cos, sin = self._rotary_cos_sin(grids)
        # One attention sequence per frame group, as SGLang's token_cu_seqlens.
        lengths = np.repeat([h * w for _, h, w in grids], [t for t, _, _ in grids])
        cu_seqlens = torch.tensor(np.concatenate([[0], lengths.cumsum()]), dtype=torch.int32, device=x.device)
        max_seqlen = int(lengths.max())
        for block in self.blocks:
            x = block(x, cu_seqlens, max_seqlen, cos, sin)
        return self.merger(x)

    def _interpolated_pos_embed(self, grids: list[tuple[int, int, int]]) -> torch.Tensor:
        # SGLang's fast_pos_embed_interpolate_from_list: weights derived by subtraction, cast to the table dtype,
        # and the four corners summed in that dtype.
        side, merge, hidden = self.num_grid_per_side, self.config.spatial_merge_size, self.config.hidden_size
        device, dtype = self.pos_embed.weight.device, self.pos_embed.weight.dtype
        outputs = []
        for t, h, w in grids:
            h_idxs = torch.linspace(0, side - 1, h, dtype=torch.float32, device=device)
            w_idxs = torch.linspace(0, side - 1, w, dtype=torch.float32, device=device)
            h_floor, w_floor = h_idxs.to(torch.long), w_idxs.to(torch.long)
            h_ceil = torch.clamp(h_floor + 1, max=side - 1)
            w_ceil = torch.clamp(w_floor + 1, max=side - 1)
            dh, dw = torch.meshgrid(h_idxs - h_floor, w_idxs - w_floor, indexing="ij")
            h_floor, w_floor = torch.meshgrid(h_floor, w_floor, indexing="ij")
            h_ceil, w_ceil = torch.meshgrid(h_ceil, w_ceil, indexing="ij")
            w11 = dh * dw
            w10 = dh - w11
            w01 = dw - w11
            w00 = 1 - dh - w01
            rows = torch.stack([h_floor, h_floor, h_ceil, h_ceil]) * side
            cols = torch.stack([w_floor, w_ceil, w_floor, w_ceil])
            weights = torch.stack([w00, w01, w10, w11]).reshape(4, -1, 1).to(dtype)
            embeds = self.pos_embed((rows + cols).reshape(4, -1)) * weights
            combined = embeds.sum(dim=0).reshape(h // merge, merge, w // merge, merge, hidden)
            combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden)
            outputs.append(combined.expand(t, -1, -1).reshape(-1, hidden))
        return torch.cat(outputs)

    def _rotary_cos_sin(self, grids: list[tuple[int, int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        # SGLang's rot_pos_emb over a RotaryEmbedding(head_dim, rotary_dim=head_dim // 2, base 10000) cache,
        # built in fp32 on the GPU; each half of a head rotates by the (row, col) frequencies.
        device = self.pos_embed.weight.device
        rotary_dim = self.config.head_dim // 2
        max_grid = max(max(h, w) for _, h, w in grids)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim))
        freqs = torch.einsum("i,j -> ij", torch.arange(max_grid, dtype=torch.float32, device=device), inv_freq)
        positions = torch.cat(
            [_rot_pos_ids(h, w, self.config.spatial_merge_size).repeat(t, 1) for t, h, w in grids]
        ).to(device)
        cos = freqs.cos()[positions].flatten(1)
        sin = freqs.sin()[positions].flatten(1)
        return torch.cat([cos, cos], dim=-1), torch.cat([sin, sin], dim=-1)


def _rot_pos_ids(h: int, w: int, merge: int) -> torch.Tensor:
    """[h * w, 2] (row, col) of every patch, in the spatial-merge block order the patches arrive in."""
    rows = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w)).reshape(h // merge, merge, w // merge, merge)
    cols = np.broadcast_to(np.arange(w).reshape(1, w), (h, w)).reshape(h // merge, merge, w // merge, merge)
    rows = rows.transpose(0, 2, 1, 3).flatten()
    cols = cols.transpose(0, 2, 1, 3).flatten()
    return torch.from_numpy(np.stack([rows, cols], axis=-1))


def load_vision_tower(hf_checkpoint: str, *, device: torch.device, dtype: torch.dtype) -> Qwen3_5VisionTower:
    """Build the frozen tower from a Qwen3.5 checkpoint's ``model.visual.*`` tensors."""
    tower = Qwen3_5VisionTower(
        Qwen3_5VisionConfig.from_hf(json.loads((Path(hf_checkpoint) / "config.json").read_text())["vision_config"])
    )
    state = {
        name.removeprefix("model.visual."): tensor
        for name, tensor in load_checkpoint_towers(hf_checkpoint)
        if name.startswith("model.visual.")
    }
    tower.load_state_dict(state, strict=True)
    tower.to(device=device, dtype=dtype)
    tower.requires_grad_(False)
    return tower.eval()
