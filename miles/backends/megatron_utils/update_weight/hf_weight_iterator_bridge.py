from typing import Any, Callable, Iterable, List, Tuple

import torch

from .hf_weight_iterator_base import HfWeightIteratorBase


class HfWeightIteratorBridge(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from megatron.bridge import AutoBridge

        self._bridge = AutoBridge.from_hf_pretrained(TODO_what)
        self._conversion_tasks = self._bridge.build_conversion_tasks(TODO_hf_config, self.model)

    def get_hf_weight_chunks(self, megatron_local_weights):
        # TODO support quantization (e.g. modify megatron-bridge to provide megatron param name)
        vanilla_named_weights = self._bridge.export_hf_weights(self.model, cpu=False, conversion_tasks=self._conversion_tasks)
        yield from chunk_named_params_by_size(vanilla_named_weights, chunk_size=self.args.update_weight_buffer_size)

