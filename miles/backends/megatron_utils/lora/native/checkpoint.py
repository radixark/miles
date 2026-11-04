"""Native LoRA (N = 1) checkpoint members: the same probe-built HF layout as Bridge, over native's own export."""

import torch

from miles.backends.megatron_utils.lora.bridge_checkpoint import (
    agreed_load_overwrites_every_element,
    probe_bridge_row_mapping,
)
from miles.backends.megatron_utils.lora.native.hf_adapter import export_lora_hf_named
from miles.backends.megatron_utils.lora.native.modules.linear import iter_adapters
from miles.backends.megatron_utils.lora.native.modules.lm_head import LoRAOutputHead
from miles.backends.megatron_utils.lora.utils import adapter_named_parameters
from miles.backends.training_utils.checkpoint.adapter.key_mapping import HfKey, map_sampler_keys_to_hf_keys
from miles.backends.training_utils.checkpoint.adapter.probed_slots import ProbedAdapterSlots


class NativeAdapterSlots(ProbedAdapterSlots):
    """Native exports grouped experts stacked under sglang's ``experts.w1/w3/w2`` (per-expert or ``[1, ...]``)."""

    bf16_adapter_key_style = "sampler"

    def __init__(self, model, iterator, targets):
        """Collective: probes the row mapping now, while nothing else touches the parameters."""
        # the probe's name mapping and export read these before the base is set up
        self._model, self._iterator, self.targets = model, iterator, targets
        unexported = self._unexported_elements()
        row_mapping = probe_bridge_row_mapping(
            model, self.slot_parameters(0), self._export_probe, targets, self._map_name, unexported
        )
        super().__init__(row_mapping, targets, iterator)
        self.load_overwrites_every_element = agreed_load_overwrites_every_element(unexported)

    def slot_parameters(self, slot: int) -> list[torch.nn.Parameter]:
        assert slot == 0, "native LoRA has only slot 0"
        return [param for params in adapter_named_parameters(self._model) for param in params.values()]

    def _map_name(self, name: str) -> HfKey:
        return map_sampler_keys_to_hf_keys(name, self.targets)

    def _head_key(self) -> str | None:
        # native names the head by the model's own module path (e.g. language_model.lm_head), not the family's
        heads = [adapter for adapter in iter_adapters(self._model) if isinstance(adapter, LoRAOutputHead)]
        assert len(heads) <= 1, "a model has one output head"
        return self._map_name(f"{heads[0].hf_prefix}lora_B.weight").key if heads else None

    def _unexported_elements(self) -> torch.Tensor:
        """head_B's padded vocab rows: the export trims them, so no checkpoint holds them and a load leaves them."""
        padded_rows = {
            id(adapter.head_B): adapter.padded_local_rows()
            for adapter in iter_adapters(self._model)
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
        for adapter in iter_adapters(self._model):
            yield from export_lora_hf_named([adapter], dtype=torch.float64)
