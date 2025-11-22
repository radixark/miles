from typing import Any, Callable, Iterable, List, Tuple

import torch

from .hf_weight_iterator_base import HfWeightIteratorBase


class HfWeightIteratorBridge(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from megatron.bridge import AutoBridge

        self._bridge = AutoBridge.from_hf_pretrained(TODO_what)
        self._conversion_tasks = self._bridge.build_conversion_tasks(hf_config, megatron_model)

    def get_hf_weight_chunks(self, megatron_local_weights):
        # TODO support quantization (e.g. modify megatron-bridge to provide megatron param name)
        vanilla_named_weights = self._bridge.export_hf_weights(self.model, cpu=False, conversion_tasks=self._conversion_tasks)
        yield from chunk_named_params_by_size(vanilla_named_weights, chunk_size=self.args.update_weight_buffer_size)


# TODO fsdp can also use this
def chunk_named_params_by_size(named_params: Iterable[Tuple[str, torch.Tensor]], chunk_size: int):
    return _chunk_by_size(
        named_params,
        compute_size=lambda named_weight: named_weight[1].nbytes,
        chunk_size=chunk_size,
    )


def _chunk_by_size(objects: Iterable[Any], compute_size: Callable[[Any], int], chunk_size: int):
    bucket: List[Any] = []
    bucket_size = 0

    for obj in objects:
        obj_size = compute_size(obj)

        if bucket and (bucket_size + obj_size) >= chunk_size:
            yield bucket
            bucket = []
            bucket_size = 0

        bucket.append(obj)
        bucket_size += obj_size

    if bucket:
        yield bucket
