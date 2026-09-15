from collections.abc import Iterator

import torch

from miles.backends.fsdp_utils.adaptations.weight_bridge import get_param_transform
from miles.backends.fsdp_utils.dtensor import gather_full_param
from miles.backends.training_utils.weight_update.hf_weight_iterator import HfWeightIteratorBase
from miles.backends.training_utils.weight_update.hf_weight_iterator.atomic_groups import get_hf_atomic_update_groups


class FSDPHfWeightIterator(HfWeightIteratorBase):
    forced_placement = None

    def _iter_hf_param_units(self, weights, *, materialize):
        model_type = getattr(getattr(self.model, "config", None), "model_type", "")
        for name, param in self.model.state_dict().items():
            full = self._to_sync_dtype(name, gather_full_param(param))
            if not materialize:
                continue
            expand = get_param_transform(name, param, model_type)
            if expand is None:
                yield [(name, full)]
            else:
                yield [
                    (hf_name, self._to_sync_dtype(hf_name, tensor))
                    for hf_name, tensor in expand(name, full, self.model)
                ]

    def _to_sync_dtype(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        target = (getattr(self.model, "_fsdp_sync_dtypes", None) or {}).get(name)
        return tensor if target is None or tensor.dtype == target else tensor.to(target)

    def _hf_atomic_update_groups(self):
        q_lora_rank = getattr(getattr(self.model, "config", None), "q_lora_rank", None) or None
        return get_hf_atomic_update_groups(self.model_name, q_lora_rank=q_lora_rank)

    def _iter_hf_adapter_units(self, lora_name, adapter, *, materialize) -> Iterator:
        raise NotImplementedError("the FSDP backend has no LoRA weight sync")
