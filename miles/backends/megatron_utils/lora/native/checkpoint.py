"""Native LoRA (N = 1) checkpoint members: the same probe-built HF layout as Bridge, over native's own export."""

from dataclasses import replace

import torch

from miles.backends.megatron_utils.lora.bridge_adapter_slots import probe_bridge_row_mapping
from miles.backends.megatron_utils.lora.native.hf_adapter import export_lora_hf_named
from miles.backends.megatron_utils.lora.native.modules.linear import iter_adapters
from miles.backends.megatron_utils.lora.native.modules.lm_head import LoRAOutputHead
from miles.backends.megatron_utils.lora.utils import adapter_named_parameters
from miles.backends.training_utils.checkpoint.adapter.adapter_state import VOCAB_PADDING_SUFFIX
from miles.backends.training_utils.checkpoint.adapter.key_mapping import HfKey, map_sampler_keys_to_hf_keys
from miles.backends.training_utils.checkpoint.adapter.probed_slots import HfPiece, ProbedAdapterSlots


class NativeAdapterSlots(ProbedAdapterSlots):
    """Native exports grouped experts stacked under sglang's ``experts.w1/w3/w2`` (per-expert or ``[1, ...]``)."""

    bf16_adapter_key_style = "sampler"

    def __init__(self, model, iterator, targets):
        """Collective: probes the row mapping now, while nothing else touches the parameters."""
        # _vocab_padding_pieces, slot_parameters, _map_name and _export_probe read these before super().__init__
        self._model, self._iterator, self.targets = model, iterator, targets
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

    def _vocab_padding_pieces(self) -> list[HfPiece]:
        """head_B's rows past the served vocab, which the export trims: adapter_state keeps them, so that the rows
        training moves resume exactly."""
        heads = {id(a.head_B): a for a in iter_adapters(self._model) if isinstance(a, LoRAOutputHead)}
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
        for adapter in iter_adapters(self._model):
            yield from export_lora_hf_named([adapter], dtype=torch.float64)
