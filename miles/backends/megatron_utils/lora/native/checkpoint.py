"""Native LoRA's adapter slots: the same probe-built HF layout as Bridge, over native's own export."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import torch

from miles.backends.megatron_utils.lora.bridge_checkpoint import BridgeAdapterSlots
from miles.backends.megatron_utils.lora.native.hf_adapter import export_lora_hf_named
from miles.backends.megatron_utils.lora.native.modules.linear import (
    adapters_hidden,
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

    def init(self, slot: int, lora_rank: int, lora_alpha: float, seed: int) -> None:
        """Claim ``slot``: A drawn within ``lora_rank`` from ``seed`` through the RNG tracker, B and the rest zero.

        The draw comes from the data-parallel stream, equal on every rank, so DP replicas and the TP ranks of an
        A that TP replicates agree, as the model's own construction does.
        """
        assert self._multi_lora, "single LoRA's adapter is built with the model"
        from megatron.core import parallel_state
        from megatron.core.tensor_parallel.random import (
            get_cuda_rng_tracker,
            get_data_parallel_rng_tracker_name,
            model_parallel_cuda_manual_seed,
        )

        tracker = get_cuda_rng_tracker()
        saved_tracker_states, saved_cuda_rng_state = tracker.get_states(), torch.cuda.get_rng_state()
        try:
            # pipeline stages hold different layers; offset them as Megatron's own seeding does
            model_parallel_cuda_manual_seed(seed + 100 * parallel_state.get_pipeline_model_parallel_rank())
            with tracker.fork(get_data_parallel_rng_tracker_name()):
                for slot_adapters in iter_multi_lora_adapters(self._model):
                    slot_adapters.init_slot(slot, lora_rank, lora_alpha)
        finally:
            tracker.set_states(saved_tracker_states)
            torch.cuda.set_rng_state(saved_cuda_rng_state)

    def clear(self, slot: int) -> None:
        assert self._multi_lora, "single LoRA's adapter is built with the model"
        for slot_adapters in iter_multi_lora_adapters(self._model):
            slot_adapters.clear_slot(slot)

    @contextmanager
    def routed_tokens_context(self, token_counts_by_slot: Sequence[int] | torch.Tensor) -> Iterator[None]:
        """Route the forward's tokens, sorted by slot, to their slots; the routing ends with the context.

        Recomputation reruns the forward in backward, so the context spans the backward too.
        """
        assert self._multi_lora, "single LoRA routes no tokens"
        # one host sync per forward
        counts = tuple(int(count) for count in torch.as_tensor(token_counts_by_slot).tolist())
        all_slot_adapters = list(iter_multi_lora_adapters(self._model))
        num_slots = len(all_slot_adapters[0].slots)
        assert len(counts) == num_slots, f"{len(counts)} token counts for {num_slots} slots"
        assert min(counts) >= 0, f"negative token counts {counts}"
        for slot_adapters in all_slot_adapters:
            slot_adapters.token_counts_by_slot = counts
        try:
            yield
        finally:
            for slot_adapters in all_slot_adapters:
                slot_adapters.token_counts_by_slot = None

    def adapters_hidden_context(self):
        return adapters_hidden(self._model)

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
