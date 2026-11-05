"""Native LoRA's adapter slots: the same probe-built HF layout as Bridge, over native's own export."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

import torch

from miles.backends.megatron_utils.lora.bridge_checkpoint import SlotComputeMembers, probe_bridge_row_mapping
from miles.backends.megatron_utils.lora.native.hf_adapter import export_lora_hf_named
from miles.backends.megatron_utils.lora.native.modules.linear import (
    adapters_hidden_context,
    iter_adapters,
    iter_multi_lora_adapters,
    iter_slot_adapters,
)
from miles.backends.megatron_utils.lora.native.modules.lm_head import LoRAOutputHead
from miles.backends.megatron_utils.lora.utils import adapter_named_parameters
from miles.backends.training_utils.checkpoint.adapter.adapter_state import VOCAB_PADDING_SUFFIX
from miles.backends.training_utils.checkpoint.adapter.key_mapping import HfKey, map_sampler_keys_to_hf_keys
from miles.backends.training_utils.checkpoint.adapter.probed_slots import HfPiece, ProbedAdapterSlots
from miles.utils.lora.utils import AdapterSpec


class NativeAdapterSlots(SlotComputeMembers, ProbedAdapterSlots):
    """Native exports grouped experts stacked under sglang's ``experts.w1/w3/w2`` (per-expert or ``[1, ...]``).

    Multi-LoRA slots are ``NativeMultiLoRAAdapter``s; they cover dense projections and lm_head. Every slot has
    slot 0's layout.
    """

    bf16_adapter_key_style = "sampler"

    def __init__(self, model, iterator, targets, *, multi_lora: bool):
        """Collective: probes the row mapping now, while nothing else touches the parameters."""
        # _vocab_padding_pieces, slot_parameters, _map_name and _export_probe read these before super().__init__
        self._model, self._iterator, self.targets, self._multi_lora = model, iterator, targets, multi_lora
        padding = self._vocab_padding_pieces()
        row_mapping = probe_bridge_row_mapping(
            model,
            self.slot_parameters(0),
            self._export_probe,
            targets,
            self._map_name,
            torch.cat([piece.local_elements.flatten() for piece in padding]) if padding else None,
        )
        # the export trims the vocab padding; adapter_state keeps it, under its own key
        super().__init__(replace(row_mapping, pieces=[*row_mapping.pieces, *padding]), targets, iterator)
        if multi_lora:
            self._count_slots()

    def _slot_adapters(self, slot: int):
        if self._multi_lora:
            return iter_slot_adapters(self._model, slot)
        assert slot == 0, "single LoRA has only slot 0"
        return iter_adapters(self._model)

    def slot_parameters(self, slot: int) -> list[torch.nn.Parameter]:
        if not self._multi_lora:
            assert slot == 0, "single LoRA has only slot 0"
            return [param for params in adapter_named_parameters(self._model) for param in params.values()]
        return [param for adapter in self._slot_adapters(slot) for param in adapter.parameters()]

    def slot_spec(self, slot: int) -> AdapterSpec | None:
        if not self._multi_lora:
            return None
        self._assert_slot(slot)
        multi_adapter = next(iter_multi_lora_adapters(self._model))
        return AdapterSpec(
            slot=slot, rank=multi_adapter.lora_ranks_by_slot[slot], alpha=multi_adapter.lora_alphas_by_slot[slot]
        )

    def _local_num_slots(self) -> int:
        return next((len(multi_adapter.adapters) for multi_adapter in iter_multi_lora_adapters(self._model)), 0)

    def _init_slot(self, slot: int, lora_rank: int, lora_alpha: float, seed: int | None) -> None:
        if seed is not None:
            with _reseeded_data_parallel_stream(seed):
                for multi_adapter in iter_multi_lora_adapters(self._model):
                    multi_adapter.redraw_slot(slot)
        for multi_adapter in iter_multi_lora_adapters(self._model):
            multi_adapter.init_slot(slot, lora_rank, lora_alpha)

    def _clear_slot(self, slot: int) -> None:
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, get_data_parallel_rng_tracker_name

        with get_cuda_rng_tracker().fork(get_data_parallel_rng_tracker_name()):
            for multi_adapter in iter_multi_lora_adapters(self._model):
                multi_adapter.clear_slot(slot)

    def _install_routing(self, counts: tuple[int, ...]) -> None:
        for multi_adapter in iter_multi_lora_adapters(self._model):
            multi_adapter.token_counts_by_slot = counts

    def adapters_hidden_context(self):
        return adapters_hidden_context(self._model)

    def _map_name(self, name: str) -> HfKey:
        return map_sampler_keys_to_hf_keys(name, self.targets)

    def _head_key(self) -> str | None:
        # native names the head by the model's own module path (e.g. language_model.lm_head), not the family's
        heads = [adapter for adapter in self._slot_adapters(0) if isinstance(adapter, LoRAOutputHead)]
        assert len(heads) <= 1, "a model has one output head"
        return self._map_name(f"{heads[0].hf_prefix}lora_B.weight").key if heads else None

    def _vocab_padding_pieces(self) -> list[HfPiece]:
        """head_B's rows past the served vocab, which the export trims: adapter_state keeps them, so that the rows
        training moves resume exactly."""
        heads = {id(a.head_B): a for a in self._slot_adapters(0) if isinstance(a, LoRAOutputHead)}
        pieces, start = [], 0
        for param in self.slot_parameters(0):
            head = heads.get(id(param))
            rows = head.padded_local_rows() if head is not None else None
            if rows is not None and rows.start < rows.stop:
                total, first = head.padding_rows()
                row_numel = param.shape[1]
                elements = torch.arange(start + rows.start * row_numel, start + rows.stop * row_numel)
                pieces.append(
                    HfPiece(
                        self._head_key() + VOCAB_PADDING_SUFFIX,
                        (total, row_numel),
                        (first, 0),
                        elements.view(rows.stop - rows.start, row_numel),
                    )
                )
            start += param.numel()
        return pieces

    def _export_probe(self):
        # one adapter at a time: a float64 export gathers every expert factor over EP onto every rank
        for adapter in self._slot_adapters(0):
            yield from export_lora_hf_named([adapter], dtype=torch.float64)


@contextmanager
def _reseeded_data_parallel_stream(seed: int) -> Iterator[None]:
    """The RNG tracker's data-parallel stream reseeded from ``seed``, every stream restored after.

    That stream is equal on every rank, so DP replicas and the TP ranks of an A that TP replicates draw alike,
    as the model's own construction does.
    """
    from megatron.core.tensor_parallel.random import (
        get_cuda_rng_tracker,
        get_data_parallel_rng_tracker_name,
        model_parallel_cuda_manual_seed,
    )

    tracker = get_cuda_rng_tracker()
    saved_tracker_states, saved_cuda_rng_state = tracker.get_states(), torch.cuda.get_rng_state()
    # the tracker snapshot / restore of Bridge's init_adapter_slot
    try:
        model_parallel_cuda_manual_seed(seed)
        with tracker.fork(get_data_parallel_rng_tracker_name()):
            yield
    finally:
        tracker.set_states(saved_tracker_states)
        torch.cuda.set_rng_state(saved_cuda_rng_state)
