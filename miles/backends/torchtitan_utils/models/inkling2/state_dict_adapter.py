import re

import torch
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard
from torchtitan.models.base.state_dict_adapter import StateDictAdapter

_HF = "model.llm."

_ROOT_MAP = {
    "tok_embeddings.weight": "embed.weight",
    "embed_norm.weight": "embed_norm.weight",
    "norm.weight": "norm.weight",
    "lm_head.weight": "unembed.weight",
    "output_query.query": "attn_res.output_query",
    "loop_queries.queries": "middle_cycle_cross_pass_queries",
    "loop_queries.kv_queries": "middle_cycle_cross_pass_kv_queries",
}

_LAYER_RENAMES = (
    ("feed_forward.router.gate.weight", "mlp.gate.weight"),
    ("feed_forward.expert_bias_E", "mlp.gate.bias"),
    ("feed_forward.routed_experts.w13.weight", "mlp.experts.w13_weight"),
    ("feed_forward.routed_experts.w2.weight", "mlp.experts.w2_weight"),
)

_LAYER_KEY = re.compile(r"layers\.(\d+)\.(.+)")
_HF_LAYER_KEY = re.compile(re.escape(_HF) + r"layers\.(\d+)\.(.+)")
_HF_QUERY_KEY = re.compile(re.escape(_HF) + r"attn_res\.(attn|mlp)_queries\.(\d+)")


def _per_shard(tensor: torch.Tensor, fn, shape: tuple[int, ...]) -> torch.Tensor:
    """Apply a reshape that leaves dim 0 alone to each rank's dim-0 shard of a DTensor."""
    if not isinstance(tensor, DTensor):
        return fn(tensor)
    if not all(
        isinstance(p, Replicate) or (isinstance(p, (Shard, _StridedShard)) and p.dim == 0) for p in tensor.placements
    ):
        raise NotImplementedError(f"cannot reshape an expert tensor placed as {tensor.placements}")
    local = fn(tensor.to_local())
    stride = torch.empty(shape, device="meta").stride()
    return DTensor.from_local(
        local, tensor.device_mesh, tensor.placements, run_check=False, shape=shape, stride=stride
    )


def _w13_to_hf(w13_E2FD: torch.Tensor) -> torch.Tensor:
    """titan stacks [gate; up] as [E, 2, F, D]; the checkpoint interleaves gate/up rows as [E, 2F, D]."""
    experts, _, hidden, dim = w13_E2FD.shape
    return _per_shard(
        w13_E2FD,
        lambda t: t.permute(0, 2, 1, 3).reshape(t.shape[0], 2 * hidden, dim),
        (experts, 2 * hidden, dim),
    )


def _w13_from_hf(w13_E2FD_hf: torch.Tensor) -> torch.Tensor:
    experts, two_hidden, dim = w13_E2FD_hf.shape
    hidden = two_hidden // 2
    return _per_shard(
        w13_E2FD_hf,
        lambda t: t.view(t.shape[0], hidden, 2, dim).permute(0, 2, 1, 3).contiguous(),
        (experts, 2, hidden, dim),
    )


class Inkling2StateDictAdapter(StateDictAdapter):
    """Maps the Inkling2 checkpoint (``model.llm.*``, interleaved fused w13) to the titan module tree."""

    def to_hf_key(self, key: str) -> str | None:
        if key in _ROOT_MAP:
            return _HF + _ROOT_MAP[key]
        match = _LAYER_KEY.fullmatch(key)
        if match is None:
            return None
        layer, rest = int(match.group(1)), match.group(2)
        if rest == "attn_query.query":
            return f"{_HF}attn_res.attn_queries.{layer - 1}"
        if rest == "mlp_query.query":
            return f"{_HF}attn_res.mlp_queries.{layer}"
        for titan, hf in _LAYER_RENAMES:
            if rest == titan:
                return f"{_HF}layers.{layer}.{hf}"
        return f"{_HF}layers.{layer}.{rest}"

    def from_hf_key(self, key: str) -> str | None:
        for titan, hf in _ROOT_MAP.items():
            if key == _HF + hf:
                return titan
        match = _HF_QUERY_KEY.fullmatch(key)
        if match is not None:
            kind, index = match.group(1), int(match.group(2))
            return f"layers.{index + 1}.attn_query.query" if kind == "attn" else f"layers.{index}.mlp_query.query"
        match = _HF_LAYER_KEY.fullmatch(key)
        if match is None:
            return None
        layer, rest = int(match.group(1)), match.group(2)
        for titan, hf in _LAYER_RENAMES:
            if rest == hf:
                return f"layers.{layer}.{titan}"
        return f"layers.{layer}.{rest}"

    def to_hf(self, state_dict: dict, **kwargs) -> dict:
        hf = {}
        for key, value in state_dict.items():
            hf_key = self.to_hf_key(key)
            if hf_key is None:
                continue
            hf[hf_key] = _w13_to_hf(value) if hf_key.endswith(".mlp.experts.w13_weight") else value
        return hf

    def from_hf(self, hf_state_dict: dict, **kwargs) -> dict:
        state = {}
        for key, value in hf_state_dict.items():
            titan_key = self.from_hf_key(key)
            if titan_key is None:
                continue
            state[titan_key] = _w13_from_hf(value) if key.endswith(".mlp.experts.w13_weight") else value
        return state
