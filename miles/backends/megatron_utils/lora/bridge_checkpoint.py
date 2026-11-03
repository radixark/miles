"""Bridge LoRA's adapter checkpoint members: HF-keyed views of the adapter parameters.

The HF layout of every Megatron adapter element (TP / EP shards, the qkv group permutation, fc1 gate / up,
GDN in_proj, per-expert names) comes from Bridge's own export, run once on index-valued parameters.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import torch
import torch.distributed as dist

from miles.backends.megatron_utils.lora.utils import adapter_named_parameters
from miles.backends.training_utils.checkpoint.adapter.adapter_state import (
    LocalShardSpec,
    gloo_group,
    raise_if_any_rank_failed,
)
from miles.backends.training_utils.checkpoint.adapter.key_mapping import (
    fused_components,
    map_peft_keys_to_hf_keys,
    split_fused_lora_b,
)
from miles.backends.training_utils.checkpoint.adapter.parametrization import AdapterParametrization
from miles.utils.lora.hf_lora_targets import HfLoraTargets
from miles.utils.lora.utils import AdapterSpec

# a probe value is rank * _RANK_STRIDE + the element's offset in the rank's concatenated adapter parameters;
# float64 holds it exactly
_RANK_STRIDE = 2**40


@dataclass(frozen=True)
class _HfPiece:
    """This rank's rectangle of one HF-keyed tensor and where each of its elements lives locally."""

    key: str
    global_shape: tuple[int, ...]
    offsets: tuple[int, ...]
    # offsets into the rank's concatenated adapter elements, in HF order; its shape is the rectangle's
    local_elements: torch.Tensor


@dataclass(frozen=True)
class BridgeRowMapping:
    pieces: list[_HfPiece]
    # HF modules whose lora_A is one Megatron parameter, in export order
    shared_lora_a_groups: tuple[tuple[str, ...], ...]
    target_modules: tuple[str, ...]


def _rectangle(name: str, mine: torch.Tensor) -> tuple[slice, ...]:
    positions = mine.nonzero()
    low, high = positions.min(0).values.tolist(), (positions.max(0).values + 1).tolist()
    box = tuple(slice(lo, hi) for lo, hi in zip(low, high, strict=True))
    assert mine[box].all(), f"{name}: this rank's elements are not one rectangle"
    return box


def _piece(key: str, codes: torch.Tensor, rank: int) -> _HfPiece | None:
    mine = codes // _RANK_STRIDE == rank
    if not mine.any():
        return None
    box = _rectangle(key, mine)
    return _HfPiece(key, tuple(codes.shape), tuple(s.start for s in box), (codes[box] % _RANK_STRIDE).cpu())


def _stack_experts(key: str, pieces_by_expert: dict[int, _HfPiece], num_experts: int) -> _HfPiece:
    experts = sorted(pieces_by_expert)
    assert experts == list(range(experts[0], experts[-1] + 1)), f"{key}: this rank's experts are not contiguous"
    first = pieces_by_expert[experts[0]]
    assert all(
        p.offsets == first.offsets and p.local_elements.shape == first.local_elements.shape
        for p in pieces_by_expert.values()
    ), f"{key}: this rank holds different rectangles of different experts"
    return _HfPiece(
        key,
        (num_experts, *first.global_shape),
        (experts[0], *first.offsets),
        torch.stack([pieces_by_expert[e].local_elements for e in experts]),
    )


def probe_bridge_row_mapping(
    params: Sequence[torch.nn.Parameter],
    export_adapter: Callable[[], Iterable[tuple[str, torch.Tensor]]],
    targets: HfLoraTargets,
) -> BridgeRowMapping:
    """Collective: run ``export_adapter`` (Bridge's PEFT-named export) on index-valued ``params``."""
    rank = dist.get_rank()
    total = sum(p.numel() for p in params)
    assert total < _RANK_STRIDE and dist.get_world_size() < 2**53 // _RANK_STRIDE, "the probe encoding overflows"
    saved = [p.data for p in params]
    pieces, a_sources, modules = [], {}, set()
    per_expert: dict[str, dict[int, _HfPiece]] = {}
    num_experts: dict[str, int] = {}

    def decode(name: str, tensor: torch.Tensor) -> None:
        assert tensor.dtype == torch.float64, f"{name}: the export changed the probe's dtype to {tensor.dtype}"
        codes = tensor.long()
        hf_key = map_peft_keys_to_hf_keys(name)
        module, factor = hf_key.key.rsplit(".", 1)
        components = fused_components(module, targets)
        if factor == "lora_A":
            # one Megatron A exported under several names is one shared A
            source = (codes.min().item(), codes.max().item(), codes.sum().item(), tuple(codes.shape))
            a_sources.setdefault(source, []).append(module if components is None else components)
        if hf_key.expert_index is not None:
            num_experts[hf_key.key] = max(num_experts.get(hf_key.key, 0), hf_key.expert_index + 1)
            if (piece := _piece(hf_key.key, codes, rank)) is not None:
                per_expert.setdefault(hf_key.key, {})[hf_key.expert_index] = piece
            modules.add(module)
        elif components is None:
            pieces.append(_piece(hf_key.key, codes, rank))
            modules.add(module)
        elif factor == "lora_A":
            pieces.append(_piece(f"{components[0]}.lora_A", codes, rank))
            modules.update(components)
        else:
            for component, part in zip(components, split_fused_lora_b(module, codes, targets), strict=True):
                pieces.append(_piece(f"{component}.lora_B", part, rank))

    error = None
    try:
        start = 0
        for param in params:
            codes = torch.arange(start, start + param.numel(), dtype=torch.float64, device=param.device)
            param.data = (codes + rank * _RANK_STRIDE).view(param.shape)
            start += param.numel()
        # the export is collective: keep consuming it after a rank-local error so no rank is left inside it
        for name, tensor in export_adapter():
            if error is None:
                try:
                    decode(name, tensor)
                except Exception as exc:
                    error = exc
    finally:
        for param, data in zip(params, saved, strict=True):
            param.data = data

    groups = []
    if error is None:
        try:
            for members in a_sources.values():
                flat = [m for member in members for m in ((member,) if isinstance(member, str) else member)]
                if len(flat) > 1:
                    groups.append(tuple(sorted(flat)))
            pieces = [p for p in pieces if p is not None]
            pieces += [_stack_experts(key, by_expert, num_experts[key]) for key, by_expert in per_expert.items()]
            # a group's A pieces are copies of one parameter; the library stores it once, under the group's first
            # member, which a fused module's A need not have been decoded under
            for group in groups:
                a_keys = {f"{member}.lora_A" for member in group}
                held = [p.key for p in pieces if p.key in a_keys]
                assert held, f"{group}: this rank holds no piece of the shared A"
                pieces = [
                    replace(p, key=f"{group[0]}.lora_A") if p.key == held[0] else p
                    for p in pieces
                    if p.key not in a_keys or p.key == held[0]
                ]
            covered = torch.zeros(total, dtype=torch.bool)
            for piece in pieces:
                covered[piece.local_elements.flatten()] = True
            assert covered.all(), f"Bridge's export does not reach {int((~covered).sum())} of this rank's elements"
        except Exception as exc:
            error = exc
    raise_if_any_rank_failed(error)

    # pipeline stages export different layers; every rank needs the whole parametrization
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, (sorted(groups), sorted(modules)), group=gloo_group())
    return BridgeRowMapping(
        pieces,
        tuple(sorted({g for rank_groups, _ in gathered for g in rank_groups})),
        tuple(sorted({m for _, rank_modules in gathered for m in rank_modules})),
    )


def _rank_axis(key: str, ndim: int) -> int:
    return ndim - 2 if key.endswith(".lora_A") else ndim - 1


class BridgeAdapterSlots:
    """Checkpoint members of Bridge LoRA; single LoRA is slot 0 of one, multi-LoRA slots are its MultiLoRALinears'.

    Multi-LoRA slots are padded to ``max_rank``; a slot's HF tensors have its real rank.
    """

    bf16_adapter_key_style: Literal["peft", "tinker"] = "peft"

    def __init__(self, model, iterator, targets: HfLoraTargets, *, multi_lora: bool):
        """Collective: probes the row mapping now, while nothing else touches the parameters."""
        self._model, self._iterator, self.targets, self._multi_lora = model, iterator, targets, multi_lora
        self._specs: dict[int, list[LocalShardSpec]] = {}
        # every slot has the same layout
        self.row_mapping = probe_bridge_row_mapping(self.slot_parameters(0), self._export_probe, targets)

    def slot_parameters(self, slot: int) -> list[torch.nn.Parameter]:
        if self._multi_lora:
            from miles.backends.megatron_utils.lora.optimizer import adapter_slot_parameters

            return adapter_slot_parameters(self._model, slot)
        assert slot == 0, "single LoRA has only slot 0"
        return [param for params in adapter_named_parameters(self._model) for param in params.values()]

    def _slot_spec(self, slot: int) -> AdapterSpec | None:
        if not self._multi_lora:
            return None
        from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear

        module = next(m for chunk in self._model for m in chunk.modules() if isinstance(m, MultiLoRALinear))
        return AdapterSpec(slot=slot, rank=int(module.rank_values[slot]), alpha=float(module.alpha_values[slot]))

    def _export_probe(self):
        if not self._multi_lora:
            return self._iterator.iter_current_adapter()
        from megatron.bridge.peft.multi_lora_layers import expose_adapter_slot

        def exported():
            with expose_adapter_slot(self._model, 0):
                yield from self._iterator.iter_current_adapter()

        return exported()

    def lora_rank(self, slot: int) -> int | None:
        """None for single LoRA, whose tensors are never padded."""
        spec = self._slot_spec(slot)
        return None if spec is None else spec.rank

    def _pieces(self, slot: int) -> list[tuple[_HfPiece, tuple[slice, ...]]]:
        """Each piece with the part of it inside the slot's real rank."""
        rank = self.lora_rank(slot)
        result = []
        for piece in self.row_mapping.pieces:
            within = [slice(None)] * len(piece.global_shape)
            if rank is not None:
                axis = _rank_axis(piece.key, len(piece.global_shape))
                within[axis] = slice(0, max(0, min(piece.local_elements.shape[axis], rank - piece.offsets[axis])))
            if all(s.stop != 0 for s in within):
                result.append((piece, tuple(within)))
        return result

    def get_local_shard_specs(self, slot: int) -> list[LocalShardSpec]:
        rank = self.lora_rank(slot)
        if rank not in self._specs:
            boxes = []
            for piece, within in self._pieces(slot):
                global_shape = list(piece.global_shape)
                if rank is not None:
                    global_shape[_rank_axis(piece.key, len(global_shape))] = rank
                boxes.append(
                    (piece.key, tuple(global_shape), piece.offsets, tuple(piece.local_elements[within].shape))
                )
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, boxes, group=gloo_group())
            first_holder = {}
            for holder, rank_boxes in enumerate(gathered):
                for box in rank_boxes:
                    first_holder.setdefault(box, holder)
            self._specs[rank] = [
                LocalShardSpec(*box, is_primary=first_holder[box] == dist.get_rank()) for box in boxes
            ]
        return self._specs[rank]

    def local_pieces(self, slot: int, tensors: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """``tensors`` aligned with ``slot_parameters(slot)`` (the parameters, their masters or moments) cut into
        the pieces ``get_local_shard_specs(slot)`` describes."""
        pieces = self._pieces(slot)
        if not pieces:
            return []
        flat = torch.cat([t.detach().reshape(-1) for t in tensors])
        return [flat[piece.local_elements[within].to(flat.device)] for piece, within in pieces]

    def export_bf16_adapter(self, slot: int) -> Iterable[tuple[str, torch.Tensor]]:
        is_writer = dist.get_rank() == 0
        return self._iterator.materialize_adapter(self._slot_spec(slot), materialize=is_writer).items()

    def parametrization(
        self,
        *,
        base_model: str,
        lora_rank: int,
        lora_alpha: float,
        lora_type: str,
        experts_shared_outer_loras: bool,
    ) -> AdapterParametrization:
        """Targets and shared-A groups are the model's (same for every slot); the rest is the caller's."""
        mapping = self.row_mapping
        return AdapterParametrization(
            base_model=base_model,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_type=lora_type,
            target_modules=mapping.target_modules,
            experts_shared_outer_loras=experts_shared_outer_loras,
            shared_lora_a_groups=mapping.shared_lora_a_groups,
        )
