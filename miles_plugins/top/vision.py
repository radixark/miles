"""The frozen Qwen3.5 vision tower under TOP: the tower's parameters run through serving's TOP tower program.

``sglang.srt.true_on_policy.vision`` owns the program (the biased BI projection, eager FP32 rotary, noncausal FA4);
serving's vision layers call the same functions, so a tower row does not depend on which side encoded it or on
what else shared its batch. Forward only: the tower is frozen, loaded from the checkpoint, and never trained.
"""

import functools

import numpy as np
import torch
import torch.nn.functional as F

from miles_plugins.models.qwen3_5.vision import Qwen3_5VisionTower, _rot_pos_ids, load_vision_tower


class TopVisionTower:
    """``Qwen3_5VisionTower``'s parameters and position tables, run by the TOP tower program.

    Not a module: the tower stays out of the optimizer, checkpoints and weight sync, as the frozen tower does.
    """

    def __init__(self, tower: Qwen3_5VisionTower):
        if tower.pos_embed.weight.dtype != torch.bfloat16:
            raise NotImplementedError("[top] the vision tower program runs BF16 parameters")
        self.tower = tower

    @torch.no_grad()
    def __call__(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """[num_patches, patch_dim] patches and the [num_items, 3] grids of a batch's images or videos ->
        [num_patches / spatial_merge_size**2, out_hidden_size] BF16, one row per placeholder token."""
        # Deferred: SGLang is only importable where it is installed.
        from sglang.srt.true_on_policy import vision as program

        tower, config = self.tower, self.tower.config
        device, dtype = tower.pos_embed.weight.device, tower.pos_embed.weight.dtype
        grids = [tuple(int(v) for v in row) for row in grid_thw.tolist()]

        # One patch is one kernel window, so the patch convolution is a projection of the flattened patch.
        proj = tower.patch_embed.proj
        x = program.linear(
            pixel_values.to(device=device, dtype=dtype).reshape(-1, proj.weight[0].numel()),
            proj.weight.reshape(proj.out_channels, -1),
            proj.bias,
        )
        x = x + tower._interpolated_pos_embed(grids)
        cos, sin = self._rotary_cos_sin(grids)
        # One attention sequence per frame, as serving packs them.
        lengths = np.repeat([h * w for _, h, w in grids], [t for t, _, _ in grids])
        cu_seqlens = torch.tensor(np.concatenate([[0], lengths.cumsum()]), dtype=torch.int32, device=device)
        max_seqlen = int(lengths.max())

        heads, head_dim = config.num_heads, config.head_dim
        for block in tower.blocks:
            attn = block.attn
            q, k, v = (
                part.reshape(-1, heads, head_dim)
                for part in program.linear(block.norm1(x), attn.qkv.weight, attn.qkv.bias).split(
                    heads * head_dim, dim=-1
                )
            )
            q, k = program.rotate(q, k, cos, sin)
            out = program.attention(q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, softmax_scale=None)
            x = x + program.linear(out.reshape(x.shape[0], -1), attn.proj.weight, attn.proj.bias)
            mlp = block.mlp
            hidden = mlp.act(program.linear(block.norm2(x), mlp.linear_fc1.weight, mlp.linear_fc1.bias))
            x = x + program.linear(hidden, mlp.linear_fc2.weight, mlp.linear_fc2.bias)

        merger = tower.merger
        x = merger.norm(x).view(-1, merger.merged_size)
        x = F.gelu(program.linear(x, merger.linear_fc1.weight, merger.linear_fc1.bias))
        return program.linear(x, merger.linear_fc2.weight, merger.linear_fc2.bias)

    def _rotary_cos_sin(self, grids: list[tuple[int, int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """[num_patches, head_dim] cos and sin: each half of a head rotates by its patch's (row, col) frequencies."""
        from sglang.srt.true_on_policy import vision as program

        config = self.tower.config
        cos, sin = program.rotary_table(head_dim=config.head_dim, max_grid=max(max(h, w) for _, h, w in grids))
        positions = torch.cat([_rot_pos_ids(h, w, config.spatial_merge_size).repeat(t, 1) for t, h, w in grids]).to(
            cos.device
        )
        cos, sin = cos[positions].flatten(1), sin[positions].flatten(1)
        return torch.cat([cos, cos], dim=-1), torch.cat([sin, sin], dim=-1)


def load_top_vision_tower(hf_checkpoint: str, *, device: torch.device) -> TopVisionTower:
    """The frozen tower of a Qwen3.5 checkpoint's ``model.visual.*`` tensors, bound to the TOP tower program."""
    return TopVisionTower(load_vision_tower(hf_checkpoint, device=device, dtype=torch.bfloat16))


@functools.cache
def top_vision_tower(hf_checkpoint: str, *, device: torch.device) -> TopVisionTower:
    """This process's tower of ``hf_checkpoint`` on ``device``, loaded at the first sample with media."""
    return load_top_vision_tower(hf_checkpoint, device=device)
