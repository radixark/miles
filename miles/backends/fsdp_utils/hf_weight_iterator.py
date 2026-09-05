"""FSDP's hook into the shared weight-update machinery.

The FSDP2 module already carries HF names, so the iterator only gathers each
shard to a full tensor, applies the registered per-architecture transform
(batched experts unfused into the per-expert names the engine expects), and
casts fp32 master weights to the dtype the rollout contract holds for them.
"""

from argparse import Namespace

import torch

from miles.backends.fsdp_utils.adaptations.weight_bridge import get_param_transform
from miles.backends.fsdp_utils.dtensor import gather_full_param
from miles.backends.training_utils.weight_update.hf_weight_iterator import (
    HfWeightIteratorBase,
    WeightUpdatePlacement,
    resolve_placement,
)
from miles.backends.training_utils.weight_update.hf_weight_iterator.atomic_groups import get_hf_atomic_update_groups


def _iter_sync_named_params(name, param, model_type, model, sync_dtypes=None):
    """Yield (name, tensor) pairs for the rollout engine, applying the registered WeightBridge transform
    for this model_type (e.g. unfusing batched MoE experts); params with no transform stream unchanged.
    ``model`` is the HF module the params came from (transforms resolve its checkpoint-conversion mapping);
    ``sync_dtypes`` casts an fp32 master tensor to the rollout contract's target dtype."""
    expand = get_param_transform(name, param, model_type)
    if expand is None:
        yield name, param
        return

    # Materialize the full (unsharded) tensor before the transform slices it.
    full = gather_full_param(param)
    if sync_dtypes is not None:
        target = sync_dtypes.get(name)
        if target is not None and full.dtype != target:
            full = full.to(target)
    yield from expand(name, full, model)


class FSDPHfWeightIterator(HfWeightIteratorBase):
    """Streams an FSDP2 module's weights as HF-named full tensors; ``model`` is the module."""

    forced_placement = None

    def _iter_hf_param_units(self, weights, *, materialize):
        model = self.model
        model_type = getattr(getattr(model, "config", None), "model_type", "")
        sync_dtypes = getattr(model, "_fsdp_sync_dtypes", None)
        for raw_name, raw_param in model.state_dict().items():
            for name, param in _iter_sync_named_params(raw_name, raw_param, model_type, model, sync_dtypes):
                full = gather_full_param(param)
                if not materialize:
                    continue
                target = sync_dtypes.get(name) if sync_dtypes is not None else None
                if target is not None and full.dtype != target:
                    full = full.to(target)
                yield [(name, full)]

    def _hf_atomic_update_groups(self):
        q_lora_rank = getattr(getattr(self.model, "config", None), "q_lora_rank", None) or None
        return get_hf_atomic_update_groups(self.model_name, q_lora_rank=q_lora_rank)

    def _iter_hf_adapter_units(self, lora_name, adapter, *, materialize):
        raise NotImplementedError("the FSDP backend has no LoRA weight sync")


def get_hf_weight_iterator(
    args: Namespace,
    model: torch.nn.Module,
    *,
    required_placement: WeightUpdatePlacement,
    model_name: str,
    quantization_config: dict | None,
) -> HfWeightIteratorBase:
    return FSDPHfWeightIterator(
        args,
        model,
        placement=resolve_placement(required_placement, FSDPHfWeightIterator.forced_placement),
        model_name=model_name,
        quantization_config=quantization_config,
    )
