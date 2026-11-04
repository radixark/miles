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

from miles.backends.torchtitan_utils.lora_names import is_adapter_key, iter_hf_adapter_tensors, split_adapter_key
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
    starts, total = {}, 0
    for key, param in params.items():
        starts[key] = total
        total += (param.to_local() if isinstance(param, DTensor) else param).numel()
    keys_by_projection: dict[str, list[str]] = {}
    for key in params:
        keys_by_projection.setdefault(split_adapter_key(key)[0], []).append(key)
    error, pieces, groups, modules = None, [], [], set()
    for keys in keys_by_projection.values():
        # placing is collective: a rank whose decoding failed keeps placing, so every rank enters every collective
        codes = {key: _local_codes(params[key], starts[key]) for key in keys}
        if error is None:
            try:
                found = _probe_local(codes, sd_adapter)
            except Exception as exc:
                error = exc
            else:
                pieces, groups, modules = pieces + found[0], groups + found[1], modules | found[2]
        # only one projection's global-shaped codes are alive at a time
        del codes
    if error is None:
        try:
            pieces = finish_local_pieces(pieces, total)
        except Exception as exc:
            error = exc
    return agreed_row_mapping(error, pieces, groups, modules)


def _probe_local(
    codes_by_key: Mapping[str, torch.Tensor], sd_adapter
) -> tuple[list[HfPiece], list[tuple[str, ...]], set[str]]:
    """Rank-local: the pieces of this rank's placed codes, named by the export itself (iter_hf_adapter_tensors)."""
    pieces, modules, a_by_source = [], set(), {}
    for name, codes in iter_hf_adapter_tensors(sd_adapter, codes_by_key):
        module, factor = name.removesuffix(".weight").rsplit(".", 1)
        modules.add(module)
        if factor == "lora_A":
            # the export yields one titan lora_a under every HF projection of a fused weight
            a_by_source.setdefault(id(codes), (codes, []))[1].append(module)
        else:
            pieces += _pieces_of(f"{module}.lora_B", codes)
    groups = []
    for codes, members in a_by_source.values():
        group = tuple(sorted(members))
        if len(group) > 1:
            groups.append(group)
        # a shared lora_A is stored once, under its sorted group's first member
        pieces += _pieces_of(f"{group[0]}.lora_A", codes)
    return pieces, groups, modules


def _pieces_of(hf_key: str, codes: torch.Tensor) -> list[HfPiece]:
    if not (mine := codes >= 0).any():
        return []
    return [
        HfPiece(hf_key, tuple(codes.shape), tuple(s.start for s in box), codes[box].clone())
        for box in rectangles(hf_key, mine)
    ]


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
