"""Native LoRA's adapter slots: the same probe-built HF layout as Bridge, over native's own export."""

from collections.abc import Iterator
from contextlib import contextmanager

import torch

from miles.backends.megatron_utils.lora.bridge_checkpoint import BridgeAdapterSlots
from miles.backends.megatron_utils.lora.native.hf_adapter import export_lora_hf_named
from miles.backends.megatron_utils.lora.native.modules.linear import (
    adapters_hidden_context,
    iter_adapters,
    iter_multi_lora_adapters,
    iter_slot_adapters,
)
from miles.backends.megatron_utils.lora.native.modules.lm_head import LoRAOutputHead
from miles.backends.training_utils.checkpoint.adapter.key_mapping import HfKey, map_sampler_keys_to_hf_keys
from miles.utils.lora.utils import AdapterSpec


class NativeAdapterSlots(BridgeAdapterSlots):
    """Native exports grouped experts stacked under sglang's ``experts.w1/w3/w2`` (per-expert or ``[1, ...]``).

    Multi-LoRA slots are ``NativeMultiLoRAAdapter``s; they cover dense projections and lm_head.
    """

    bf16_adapter_key_style = "sampler"

    def _map_name(self, name: str) -> HfKey:
        return map_sampler_keys_to_hf_keys(name, self.targets)

    def _slot_adapters(self, slot: int):
        if self._multi_lora:
            return iter_slot_adapters(self._model, slot)
        assert slot == 0, "single LoRA has only slot 0"
        return iter_adapters(self._model)

    def slot_parameters(self, slot: int) -> list[torch.nn.Parameter]:
        if not self._multi_lora:
            return super().slot_parameters(slot)
        return [param for adapter in self._slot_adapters(slot) for param in adapter.parameters()]

    def slot_spec(self, slot: int) -> AdapterSpec | None:
        assert not self._multi_lora, "native multi-LoRA slots have no checkpoint members yet"
        return None

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

    def _local_num_slots(self) -> int:
        return next((len(multi_adapter.adapters) for multi_adapter in iter_multi_lora_adapters(self._model)), 0)

    def adapters_hidden_context(self):
        return adapters_hidden_context(self._model)

    def _unexported_elements(self) -> torch.Tensor:
        """head_B's padded vocab rows: the export trims them, so no checkpoint holds them and a load leaves them."""
        padded_rows = {
            id(adapter.head_B): adapter.padded_local_rows()
            for adapter in self._slot_adapters(0)
            if isinstance(adapter, LoRAOutputHead)
        }
        offsets, start = [torch.empty(0, dtype=torch.long)], 0
        for param in self.slot_parameters(0):
            if id(param) in padded_rows:
                rows, row_numel = padded_rows[id(param)], param.shape[1]
                offsets.append(torch.arange(start + rows.start * row_numel, start + rows.stop * row_numel))
            start += param.numel()
        return torch.cat(offsets)

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
