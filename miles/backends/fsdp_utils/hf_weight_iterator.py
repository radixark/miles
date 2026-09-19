from collections import deque
from collections.abc import Iterator

import torch
from torch.distributed._functional_collectives import AsyncCollectiveTensor

from miles.backends.fsdp_utils.adaptations.weight_bridge import get_param_transform
from miles.backends.fsdp_utils.dtensor import gather_full_param
from miles.backends.training_utils.weight_update.hf_weight_iterator import HfWeightIteratorBase
from miles.backends.training_utils.weight_update.hf_weight_iterator.atomic_groups import get_hf_atomic_update_groups


class FSDPHfWeightIterator(HfWeightIteratorBase):
    forced_placement = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sync_dtypes: dict[str, torch.dtype] = getattr(self.model, "_fsdp_sync_dtypes", None) or {}

    def _iter_hf_param_units(self, weights, *, materialize):
        pending: deque[tuple[str, torch.Tensor, torch.Tensor]] = deque()
        pending_bytes = 0
        for name, param in self.model.state_dict().items():
            pending.append((name, param, gather_full_param(param, async_op=True)))
            pending_bytes += param.numel() * param.element_size()
            if pending_bytes >= self.args.update_weight_buffer_size:
                yield from self._drain(pending, materialize=materialize)
                pending_bytes = 0
        yield from self._drain(pending, materialize=materialize)

    def _drain(self, pending: deque, *, materialize: bool) -> Iterator[list[tuple[str, torch.Tensor]]]:
        model_type = self.model.config.model_type
        while pending:
            name, param, full = pending.popleft()
            if isinstance(full, AsyncCollectiveTensor):
                full = full.wait()
            if not materialize:
                continue
            full = self._to_sync_dtype(name, full)
            expand = get_param_transform(name, param, model_type)
            if expand is None:
                yield [(name, full)]
            else:
                yield [
                    (hf_name, self._to_sync_dtype(hf_name, tensor))
                    for hf_name, tensor in expand(name, full, self.model)
                ]

    def _to_sync_dtype(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        target = self._sync_dtypes.get(name)
        return tensor if target is None or tensor.dtype == target else tensor.to(target)

    def _hf_atomic_update_groups(self):
        return get_hf_atomic_update_groups(
            self.model_name, q_lora_rank=getattr(self.model.config, "q_lora_rank", None) or None
        )

    def _iter_hf_adapter_units(self, lora_name, adapter, *, materialize) -> Iterator:
        raise NotImplementedError("the FSDP backend has no LoRA weight sync")
