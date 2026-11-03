"""Native LoRA (N = 1) checkpoint members: the same probe-built HF layout as Bridge, over native's own export."""

import torch

from miles.backends.megatron_utils.lora.bridge_checkpoint import BridgeAdapterSlots
from miles.backends.megatron_utils.lora.native.hf_adapter import export_lora_hf_named
from miles.backends.megatron_utils.lora.native.modules.linear import iter_adapters
from miles.backends.megatron_utils.lora.native.modules.lm_head import LoRAOutputHead
from miles.backends.training_utils.checkpoint.adapter.key_mapping import HfKey, map_sampler_keys_to_hf_keys


class NativeAdapterSlots(BridgeAdapterSlots):
    """Native exports grouped experts stacked under sglang's ``experts.w1/w3/w2`` (per-expert or ``[1, ...]``)."""

    bf16_adapter_key_style = "sampler"

    def __init__(self, model, iterator, targets):
        super().__init__(model, iterator, targets, multi_lora=False)

    def _map_name(self, name: str) -> HfKey:
        return map_sampler_keys_to_hf_keys(name, self.targets)

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
