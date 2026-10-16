"""Step5 HF names <-> torchtitan names, one to one: the checkpoint packs each MoE layer's routed experts
into ``[E, F, D]`` tensors, the layout of torchtitan's grouped experts.

The released checkpoint stores most routed experts in block-scaled FP8; the reader dequantizes them as
it reads, so the trainer holds every weight in high precision.
"""

import math
import re
from pathlib import Path

import torch
from safetensors import safe_open
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.checkpoint.planner import ReadItem
from torch.distributed.checkpoint.quantized_hf_storage import QuantizedHuggingFaceStorageReader
from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from miles_plugins.models.step5.region_selector import CHECKPOINT_NAMES as INDEXER_CHECKPOINT_NAMES

_DIRECT = {
    "tok_embeddings.weight": "model.embed_tokens.weight",
    "norm.rms.weight": "model.norm.weight",
    "lm_head.weight": "lm_head.weight",
}

_LAYER = {
    "attention_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    **{f"attention.{proj}.weight": f"self_attn.{proj}.weight" for proj in ("q_proj", "k_proj", "v_proj", "g_proj")},
    "attention.o_proj.weight": "self_attn.o_proj.weight",
    "attention.q_norm.weight": "self_attn.q_norm.weight",
    "attention.k_norm.weight": "self_attn.k_norm.weight",
    **{f"attention.indexer.{name}": f"self_attn.{hf}" for name, hf in INDEXER_CHECKPOINT_NAMES.items()},
    "feed_forward.w1.weight": "mlp.gate_proj.weight",
    "feed_forward.w3.weight": "mlp.up_proj.weight",
    "feed_forward.w2.weight": "mlp.down_proj.weight",
    "moe.router.gate.weight": "moe.gate.weight",
    "moe.expert_bias_E": "moe.router_bias",
    "moe.shared_experts.w1.weight": "share_expert.gate_proj.weight",
    "moe.shared_experts.w3.weight": "share_expert.up_proj.weight",
    "moe.shared_experts.w2.weight": "share_expert.down_proj.weight",
    "moe.routed_experts.inner_experts.w1_EFD": "moe.gate_proj.weight",
    "moe.routed_experts.inner_experts.w3_EFD": "moe.up_proj.weight",
    "moe.routed_experts.inner_experts.w2_EDF": "moe.down_proj.weight",
}

_TITAN_LAYER = re.compile(r"layers\.(\d+)\.(.+)")
_HF_LAYER = re.compile(r"model\.layers\.(\d+)\.(.+)")
_FROM_HF_LAYER = {hf: titan for titan, hf in _LAYER.items()}
_FROM_HF_DIRECT = {hf: titan for titan, hf in _DIRECT.items()}


def to_hf_name(name: str) -> str:
    if name in _DIRECT:
        return _DIRECT[name]
    match = _TITAN_LAYER.fullmatch(name)
    if match is None or match.group(2) not in _LAYER:
        raise ValueError(f"no Step5 checkpoint tensor for {name}")
    return f"model.layers.{match.group(1)}.{_LAYER[match.group(2)]}"


def from_hf_name(name: str) -> str:
    if name in _FROM_HF_DIRECT:
        return _FROM_HF_DIRECT[name]
    match = _HF_LAYER.fullmatch(name)
    if match is None or match.group(2) not in _FROM_HF_LAYER:
        raise ValueError(f"no torchtitan Step5 tensor for {name}")
    return f"layers.{match.group(1)}.{_FROM_HF_LAYER[match.group(2)]}"


class Step5StateDictAdapter(StateDictAdapter):
    def to_hf(self, state_dict: dict) -> dict:
        return {to_hf_name(name): tensor for name, tensor in state_dict.items()}

    def from_hf(self, hf_state_dict: dict) -> dict:
        return {from_hf_name(name): tensor for name, tensor in hf_state_dict.items()}

    def get_hf_storage_reader(self, path: str, from_quantized: bool = False) -> HuggingFaceStorageReader:
        # the released checkpoint is quantized whether or not the trainer was told so
        return PackedFp8StorageReader(path)


class PackedFp8StorageReader(QuantizedHuggingFaceStorageReader):
    """Reads block-scaled FP8 tensors of any rank, the blocks tiling the last two dimensions, as FP32.

    torch's reader dequantizes 2-D weights only; Step5 packs a layer's experts into one 3-D tensor
    with ``[E, F / 128, D / 128]`` scales.
    """

    def _read_quantized_tensor_with_block_alignment(self, req: ReadItem, safetensor_file) -> torch.Tensor:
        name = req.storage_index.fqn
        *leading, rows, cols = (
            slice(start, start + length) for start, length in zip(req.storage_offsets, req.lengths, strict=True)
        )
        weight = safetensor_file.get_slice(name)[(*leading, rows, cols)]
        block = self.block_size
        scale = self._scale_inv(name, safetensor_file)[
            (
                *leading,
                slice(rows.start // block, math.ceil(rows.stop / block)),
                slice(cols.start // block, math.ceil(cols.stop / block)),
            )
        ]
        scale = scale.repeat_interleave(block, dim=-2).repeat_interleave(block, dim=-1)
        row_offset, col_offset = rows.start % block, cols.start % block
        scale = scale[..., row_offset : row_offset + weight.shape[-2], col_offset : col_offset + weight.shape[-1]]
        return (weight.float() * scale.float()).to(self.target_dtype)

    def _scale_inv(self, name: str, safetensor_file) -> torch.Tensor:
        scale_name = self._weight_scale_mapping[name]
        if self._weight_map[scale_name] == self._weight_map[name]:
            return safetensor_file.get_tensor(scale_name)
        with safe_open(Path(self.path) / self._weight_map[scale_name], framework="pt", device="cpu") as handle:
            return handle.get_tensor(scale_name)
