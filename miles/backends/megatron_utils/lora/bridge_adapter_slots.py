"""Bridge LoRA's adapter checkpoint members: HF-keyed views of the adapter parameters.

The HF layout of every Megatron adapter element (TP / EP shards, the qkv group permutation, fc1 gate / up,
GDN in_proj, per-expert names) comes from Bridge's own export, run once on index-valued parameters.
"""

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from typing import Literal

import torch
import torch.distributed as dist

from miles.backends.megatron_utils.lora.utils import adapter_named_parameters
from miles.backends.training_utils.checkpoint.adapter.key_mapping import (
    HfKey,
    fused_components,
    map_peft_keys_to_hf_keys,
    split_fused_lora_b,
)
from miles.backends.training_utils.checkpoint.adapter.probed_slots import (
    HfPiece,
    ProbedAdapterSlots,
    RowMapping,
    agreed_row_mapping,
    finish_local_pieces,
    rectangles,
)
from miles.utils.lora.hf_lora_targets import HfLoraTargets
from miles.utils.lora.utils import AdapterSpec

# a probe value is rank * _RANK_STRIDE + the element's offset in the rank's concatenated adapter parameters;
# float64 holds it exactly
_RANK_STRIDE = 2**40


def _pieces_of(key: str, codes: torch.Tensor, rank: int) -> list[HfPiece]:
    mine = codes // _RANK_STRIDE == rank
    if not mine.any():
        return []
    return [
        HfPiece(key, tuple(codes.shape), tuple(s.start for s in box), (codes[box] % _RANK_STRIDE).cpu())
        for box in rectangles(key, mine)
    ]


def _stack_experts(key: str, pieces_by_expert: dict[int, list[HfPiece]], num_experts: int) -> list[HfPiece]:
    experts = sorted(pieces_by_expert)
    assert experts == list(range(experts[0], experts[-1] + 1)), f"{key}: this rank's experts are not contiguous"
    first = pieces_by_expert[experts[0]]
    assert all(
        len(pieces) == len(first)
        and all(
            p.offsets == f.offsets and p.local_elements.shape == f.local_elements.shape
            for p, f in zip(pieces, first, strict=True)
        )
        for pieces in pieces_by_expert.values()
    ), f"{key}: this rank holds different rectangles of different experts"
    return [
        HfPiece(
            key,
            (num_experts, *piece.global_shape),
            (experts[0], *piece.offsets),
            torch.stack([pieces_by_expert[e][j].local_elements for e in experts]),
        )
        for j, piece in enumerate(first)
    ]


def probe_bridge_row_mapping(
    model: Sequence[torch.nn.Module],
    params: Sequence[torch.nn.Parameter],
    export_adapter: Callable[[], Iterable[tuple[str, torch.Tensor]]],
    targets: HfLoraTargets,
    map_name: Callable[[str], HfKey] = map_peft_keys_to_hf_keys,
    unexported: torch.Tensor | None = None,
) -> RowMapping:
    """Collective: run ``export_adapter`` (a backend's HF-named export, names mapped by ``map_name``) with
    index-valued stand-ins registered in ``model`` in place of ``params``.

    ``unexported``: offsets of the elements the export leaves out by construction, which the caller stores itself.
    """
    rank = dist.get_rank()
    total = sum(p.numel() for p in params)
    assert total < _RANK_STRIDE and dist.get_world_size() < 2**53 // _RANK_STRIDE, "the probe encoding overflows"
    pieces, a_sources, modules = [], {}, set()
    per_expert: dict[str, dict[int, list[HfPiece]]] = {}
    num_experts: dict[str, int] = {}

    def decode(name: str, tensor: torch.Tensor) -> None:
        assert tensor.dtype == torch.float64, f"{name}: the export changed the probe's dtype to {tensor.dtype}"
        codes = tensor.long()
        hf_key = map_name(name)
        module, factor = hf_key.key.rsplit(".", 1)
        components = fused_components(module, targets)
        if factor == "lora_A":
            # one Megatron A exported under several names is one shared A
            source = (codes.min().item(), codes.max().item(), codes.sum().item(), tuple(codes.shape))
            a_sources.setdefault(source, []).append(module if components is None else components)
        if hf_key.expert_index is not None:
            num_experts[hf_key.key] = max(num_experts.get(hf_key.key, 0), hf_key.expert_index + 1)
            if expert_pieces := _pieces_of(hf_key.key, codes, rank):
                per_expert.setdefault(hf_key.key, {})[hf_key.expert_index] = expert_pieces
            modules.add(module)
        elif components is None:
            pieces.extend(_pieces_of(hf_key.key, codes, rank))
            modules.add(module)
        elif factor == "lora_A":
            pieces.extend(_pieces_of(f"{components[0]}.lora_A", codes, rank))
            modules.update(components)
        else:
            for component, part in zip(components, split_fused_lora_b(module, codes, targets), strict=True):
                pieces.extend(_pieces_of(f"{component}.lora_B", part, rank))

    stand_ins, start = [], 0
    for param in params:
        codes = torch.arange(start, start + param.numel(), dtype=torch.float64, device=param.device)
        stand_ins.append(torch.nn.Parameter((codes + rank * _RANK_STRIDE).view(param.shape), requires_grad=False))
        start += param.numel()
    error = None
    try:
        with _registered_in_place_of(model, params, stand_ins):
            # the export is collective: keep consuming it after a rank-local error so no rank is left inside it
            for name, tensor in export_adapter():
                if error is None:
                    try:
                        decode(name, tensor)
                    except Exception as exc:
                        error = exc
    except Exception as exc:
        # an export failure outside Bridge's collectives goes through the rank agreement, so the other ranks fail
        # instead of waiting; a rank failing mid-collective still leaves the others hanging in Bridge's NCCL
        error = exc

    # every expert's shared A yields the same group
    groups = set()
    if error is None:
        try:
            for members in a_sources.values():
                flat = [m for member in members for m in ((member,) if isinstance(member, str) else member)]
                if len(flat) > 1:
                    groups.add(tuple(sorted(flat)))
            pieces += [
                piece
                for key, by_expert in per_expert.items()
                for piece in _stack_experts(key, by_expert, num_experts[key])
            ]
            # a group's A pieces are copies of one parameter; the library stores it once, under the group's first
            # member, which a fused module's A need not have been decoded under
            for group in groups:
                a_keys = {f"{member}.lora_A" for member in group}
                held = [p.key for p in pieces if p.key in a_keys]
                # Bridge broadcasts every pipeline stage's adapters to all stages; another stage's group has no
                # piece here, and the coverage check below still catches a group of this rank's that has none
                if not held:
                    continue
                pieces = [
                    replace(p, key=f"{group[0]}.lora_A") if p.key == held[0] else p
                    for p in pieces
                    if p.key not in a_keys or p.key == held[0]
                ]
            pieces = finish_local_pieces(pieces, total, unexported)
        except Exception as exc:
            error = exc
    return agreed_row_mapping(error, pieces, groups, modules)


@contextmanager
def _registered_in_place_of(
    model: Sequence[torch.nn.Module], params: Sequence[torch.nn.Parameter], stand_ins: Sequence[torch.nn.Parameter]
) -> Iterator[None]:
    """Setting a parameter's ``.data`` to another dtype replaces its grad accumulator, and with it the DDP hook
    that fills ``main_grad``; the probe registers stand-ins instead and leaves the parameters alone."""
    stand_in_by_param_id = {id(param): stand_in for param, stand_in in zip(params, stand_ins, strict=True)}
    registrations = [
        (module, name, param)
        for chunk in model
        for module in chunk.modules()
        for name, param in module._parameters.items()
        if param is not None and id(param) in stand_in_by_param_id
    ]
    assert {
        id(param) for _, _, param in registrations
    } == stand_in_by_param_id.keys(), "a probed parameter is unregistered"
    for module, name, param in registrations:
        module._parameters[name] = stand_in_by_param_id[id(param)]
    try:
        yield
    finally:
        for module, name, param in registrations:
            module._parameters[name] = param


class BridgeAdapterSlots(ProbedAdapterSlots):
    """Checkpoint members of Bridge LoRA; single LoRA is slot 0 of one, multi-LoRA slots are its MultiLoRALinears'.

    Multi-LoRA slots are padded to ``max_rank``; a slot's HF tensors have its real rank.
    """

    bf16_adapter_key_style: Literal["peft", "sampler"] = "peft"

    def __init__(self, model, iterator, targets: HfLoraTargets, *, multi_lora: bool):
        """Collective: probes the row mapping now, while nothing else touches the parameters."""
        # _export_probe and slot_parameters read these during the probe, before super().__init__
        self._model, self._iterator, self.targets, self._multi_lora = model, iterator, targets, multi_lora
        # every slot has the same layout
        row_mapping = probe_bridge_row_mapping(model, self.slot_parameters(0), self._export_probe, targets)
        super().__init__(row_mapping, targets, iterator)

    def slot_parameters(self, slot: int) -> list[torch.nn.Parameter]:
        if self._multi_lora:
            from miles.backends.megatron_utils.lora.optimizer import adapter_slot_parameters

            return adapter_slot_parameters(self._model, slot)
        assert slot == 0, "single LoRA has only slot 0"
        return [param for params in adapter_named_parameters(self._model) for param in params.values()]

    def slot_spec(self, slot: int) -> AdapterSpec | None:
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
