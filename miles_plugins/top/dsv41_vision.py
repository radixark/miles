"""The frozen DeepSeek-V4.1 vision tower under TOP: the checkpoint's ViT and aligner run serving's TOP tower program.

``sglang.srt.true_on_policy.vision`` owns the projections, the FP32-weight RMSNorm and the noncausal attention;
``sglang.srt.models.deepseek_v41_vit`` owns the FP32 2D rotary, the aligner's window downsample and the image span.
Serving's tower calls the same functions under TOP, so an image's rows do not depend on which side encoded it.
Forward only: the tower is frozen, loaded from the checkpoint, and never trained.
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

_PREFIXES = ("vision.", "aligner.", "image_")


class TopV41VisionTower:
    """The ViT, aligner and image boundary embeddings of a V4.1 checkpoint, bound to the TOP tower program.

    Not a module: the tower stays out of the optimizer, checkpoints and weight sync."""

    def __init__(self, tensors: dict[str, torch.Tensor], config: dict):
        vision = config["vision_config"]
        self.t = tensors
        self.n_layers = vision["num_hidden_layers"]
        self.heads = vision["num_attention_heads"]
        self.dim = vision["hidden_size"]
        self.rope_theta = vision["rope_theta"]
        self.ratio = vision["downsample_ratio"]
        self.eps = 1e-6

    @torch.no_grad()
    def __call__(self, patches: torch.Tensor, *, n_vit_h: int, n_vit_w: int) -> torch.Tensor:
        """[n_vit_h * n_vit_w, 3, p, p] BF16 patches of one image -> [num_image_tokens, hidden] BF16, the rows its
        image-token span takes in the LLM input."""
        from sglang.srt.models.deepseek_v41_vit import downsample, image_span

        t = self.t
        x = self._vit(patches, n_vit_h=n_vit_h, n_vit_w=n_vit_w)
        x = downsample(x, n_h=n_vit_h, n_w=n_vit_w, ratio=self.ratio)
        x = self._linear("aligner.w2", F.gelu(self._linear("aligner.w1", x)))
        return image_span(
            x,
            n_vit_h=n_vit_h,
            n_vit_w=n_vit_w,
            ratio=self.ratio,
            start=t["image_start"],
            newline=t["image_newline"],
            end=t["image_end"],
        )

    def _vit(self, patches: torch.Tensor, *, n_vit_h: int, n_vit_w: int) -> torch.Tensor:
        from sglang.srt.models.deepseek_v41_vit import apply_rotary, get_vision_cos_sin
        from sglang.srt.true_on_policy import vision as program

        device = self.t["vision.norm.weight"].device
        x = self._linear("vision.patch_embed.proj", patches.to(device=device, dtype=torch.bfloat16).flatten(1))
        tokens = x.shape[0]
        head_dim = self.dim // self.heads
        cos, sin = get_vision_cos_sin(n_vit_h, n_vit_w, head_dim // 2, self.rope_theta)
        cos, sin = cos.to(device), sin.to(device)
        cu_seqlens = torch.tensor([0, tokens], dtype=torch.int32, device=device)
        for i in range(self.n_layers):
            p = f"vision.blocks.{i}."
            qkv = self._linear(p + "attn.wqkv", self._norm(p + "norm1", x))
            q, k, v = (part.reshape(tokens, self.heads, head_dim) for part in qkv.split(self.dim, dim=-1))
            q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
            out = program.attention(q, k, v, cu_seqlens=cu_seqlens, max_seqlen=tokens, softmax_scale=None)
            x = x + self._linear(p + "attn.wo", out.reshape(tokens, -1))
            gate, up = self._linear(p + "mlp.w1", self._norm(p + "norm2", x)).chunk(2, dim=-1)
            x = x + self._linear(p + "mlp.w2", F.silu(gate) * up)
        return self._norm("vision.norm", x)

    def _linear(self, name: str, x: torch.Tensor) -> torch.Tensor:
        from sglang.srt.true_on_policy import vision as program

        return program.linear(x, self.t[name + ".weight"], self.t.get(name + ".bias"))

    def _norm(self, name: str, x: torch.Tensor) -> torch.Tensor:
        from sglang.srt.true_on_policy import vision as program

        return program.rms_norm(x, self.t[name + ".weight"], self.eps)


def load_top_v41_vision_tower(hf_checkpoint: str, *, device: torch.device) -> TopV41VisionTower:
    """The tower of a V4.1 checkpoint's ``vision.*``, ``aligner.*`` and ``image_*`` tensors on ``device``: norm
    weights in FP32 (serving upcasts them at load), everything else BF16 as stored."""
    path = Path(hf_checkpoint)
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    config = json.loads((path / "config.json").read_text())
    tensors = {}
    for shard in sorted({file for name, file in index.items() if name.startswith(_PREFIXES)}):
        with safe_open(path / shard, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                if name.startswith(_PREFIXES):
                    tensor = reader.get_tensor(name)
                    if tensor.dtype != torch.bfloat16:
                        raise NotImplementedError(f"[top] V4.1 tower tensor {name} is {tensor.dtype}, not BF16")
                    is_norm = re.search(r"\.norm\d?\.weight$", name) is not None
                    tensors[name] = tensor.to(device=device, dtype=torch.float32 if is_norm else torch.bfloat16)
    if not tensors:
        raise NotImplementedError(f"[top] {hf_checkpoint} carries no V4.1 vision tower")
    return TopV41VisionTower(tensors, config)


@functools.cache
def top_v41_vision_tower(hf_checkpoint: str, *, device: torch.device) -> TopV41VisionTower:
    """This process's tower of ``hf_checkpoint`` on ``device``, loaded at the first sample with images."""
    return load_top_v41_vision_tower(hf_checkpoint, device=device)
