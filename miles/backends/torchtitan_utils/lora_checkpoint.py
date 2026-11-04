"""torchtitan LoRA's adapter checkpoint members and the HF layout of its adapter elements.

The layout comes from the model's own state-dict adapter: ``sd_adapter.to_hf`` runs on index-valued tensors of each
adapter's global shape, so fused projections (wqkv, w13) split exactly as their base weights do; each rank keeps the
HF positions its DTensor shard holds. Collective: ranks agree on a probe failure, then gather every pipeline stage's
targets and shared-A groups.
"""

from collections.abc import Mapping

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from miles.backends.torchtitan_utils.lora_names import is_adapter_key, split_adapter_key
from miles.backends.training_utils.checkpoint.adapter.probed_slots import (
    HfPiece,
    ProbedAdapterSlots,
    RowMapping,
    agreed_row_mapping,
    finish_local_pieces,
    rectangles,
)
from miles.utils.lora.hf_lora_targets import HfLoraTargets


# a placed code is rank * _RANK_STRIDE + the element's offset in the rank's concatenated local shards
_RANK_STRIDE = 2**40


def _local_codes(param: torch.Tensor, start: int) -> torch.Tensor:
    """Collective for a DTensor. A global-shaped tensor: this rank's elements numbered from ``start`` in local
    row-major order, others -1."""
    if not isinstance(param, DTensor):
        return torch.arange(start, start + param.numel()).view(param.shape)
    local = param.to_local()
    rank = dist.get_rank()
    codes = torch.arange(start, start + local.numel(), device=local.device).view(local.shape) + rank * _RANK_STRIDE
    # DTensor places every rank's shard, so any placement (strided, uneven, multi-dim) lands on its global rows
    placed = DTensor.from_local(codes, param.device_mesh, param.placements, shape=param.shape, stride=param.stride())
    placed = placed.full_tensor().cpu()
    return torch.where(placed // _RANK_STRIDE == rank, placed % _RANK_STRIDE, -1)


def probe_torchtitan_row_mapping(params: Mapping[str, torch.Tensor], sd_adapter) -> RowMapping:
    """Collective. ``params``: the adapter parameters by titan key, in the order their local shards are
    concatenated."""
    # placing the codes is collective, so every rank does it for every parameter before any rank-local step can fail
    codes, start = {}, 0
    for key, param in params.items():
        codes[key] = _local_codes(param, start)
        start += (param.to_local() if isinstance(param, DTensor) else param).numel()
    error, pieces, groups, modules = None, [], [], set()
    try:
        pieces, groups, modules = _probe_local(codes, start, sd_adapter)
    except Exception as exc:
        error = exc
    return agreed_row_mapping(error, pieces, groups, modules)


def _probe_local(
    codes_by_key: Mapping[str, torch.Tensor], total: int, sd_adapter
) -> tuple[list[HfPiece], list[tuple[str, ...]], set[str]]:
    """Rank-local: the pieces of this rank's placed codes."""
    pieces, groups, modules = [], [], set()
    for key in codes_by_key:
        projection, factor = split_adapter_key(key)
        if factor != "lora_b":
            continue
        lora_a_key = f"{projection}.lora_a.weight"
        lora_b = sd_adapter.to_hf({f"{projection}.weight": codes_by_key[key]})
        assert lora_b, f"{projection}: to_hf exports nothing"
        lora_a = codes_by_key[lora_a_key]
        hf_modules = [name.removesuffix(".weight") for name in lora_b]
        modules.update(hf_modules)
        group = tuple(sorted(hf_modules))
        if len(group) > 1:
            # one titan lora_a serves every HF projection of a fused weight; it is stored once, under the group's
            # first member
            groups.append(group)
        named = [(f"{group[0]}.lora_A", lora_a)]
        named += [(f"{module}.lora_B", part) for module, part in zip(hf_modules, lora_b.values(), strict=True)]
        for hf_key, codes in named:
            if (mine := codes >= 0).any():
                pieces += [
                    HfPiece(hf_key, tuple(codes.shape), tuple(s.start for s in box), codes[box].clone())
                    for box in rectangles(hf_key, mine)
                ]
    return finish_local_pieces(pieces, total), groups, modules


class TorchtitanAdapterSlots(ProbedAdapterSlots):
    """Single-LoRA checkpoint members of the torchtitan backend; ``slot_parameters`` are the local shards."""

    bf16_adapter_key_style = "peft"

    def __init__(self, model_parts, iterator, sd_adapter, targets: HfLoraTargets):
        """Collective: probes the row mapping."""
        self._params = {
            key: param for part in model_parts for key, param in part.named_parameters() if is_adapter_key(key)
        }
        super().__init__(probe_torchtitan_row_mapping(self._params, sd_adapter), targets, iterator)

    def slot_parameters(self, slot: int) -> list[torch.Tensor]:
        assert slot == 0, "the torchtitan backend trains a single LoRA adapter"
        return [p.to_local() if isinstance(p, DTensor) else p for p in self._params.values()]
