"""torchtitan LoRA's adapter checkpoint members and the HF layout of its adapter elements.

The layout comes from the model's own state-dict adapter: ``sd_adapter.to_hf`` runs on index-valued tensors of each
adapter's global shape, so fused projections (wqkv, w13) split exactly as their base weights do; each rank keeps the
HF positions its DTensor shard holds. Collective: ranks agree on a probe failure, then gather every pipeline stage's
targets and shared-A groups.
"""

from collections.abc import Mapping

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

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


def _local_box(param: torch.Tensor) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if isinstance(param, DTensor):
        shape, offset = compute_local_shape_and_global_offset(param.shape, param.device_mesh, param.placements)
        return tuple(shape), tuple(offset)
    return tuple(param.shape), (0,) * param.dim()


def _local_codes(param: torch.Tensor, start: int) -> torch.Tensor:
    """A global-shaped tensor: this rank's elements numbered from ``start`` in local row-major order, others -1."""
    shape, offset = _local_box(param)
    codes = torch.full(tuple(param.shape), -1, dtype=torch.long)
    box = tuple(slice(o, o + n) for o, n in zip(offset, shape, strict=True))
    codes[box] = torch.arange(start, start + int(torch.Size(shape).numel())).view(shape)
    return codes


def probe_torchtitan_row_mapping(params: Mapping[str, torch.Tensor], sd_adapter) -> RowMapping:
    """Collective. ``params``: the adapter parameters by titan key, in the order their local shards are
    concatenated."""
    error, pieces, groups, modules = None, [], [], set()
    try:
        pieces, groups, modules = _probe_local(params, sd_adapter)
    except Exception as exc:
        error = exc
    return agreed_row_mapping(error, pieces, groups, modules)


def _probe_local(
    params: Mapping[str, torch.Tensor], sd_adapter
) -> tuple[list[HfPiece], list[tuple[str, ...]], set[str]]:
    starts, start = {}, 0
    for key, param in params.items():
        shape = _local_box(param)[0]
        # pieces index the to_local() shards, so the computed box must be exactly that shard
        assert not isinstance(param, DTensor) or shape == tuple(param.to_local().shape), (key, shape)
        starts[key] = start
        start += int(torch.Size(shape).numel())
    pieces, groups, modules = [], [], set()
    for key in params:
        projection, factor = split_adapter_key(key)
        if factor != "lora_b":
            continue
        lora_a_key = f"{projection}.lora_a.weight"
        lora_b = sd_adapter.to_hf({f"{projection}.weight": _local_codes(params[key], starts[key])})
        assert lora_b, f"{projection}: to_hf exports nothing"
        lora_a = _local_codes(params[lora_a_key], starts[lora_a_key])
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
    return finish_local_pieces(pieces, start), groups, modules


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
