"""torchtitan's hook into the shared weight-update machinery.

Transport, engine session, bucketing and atomic groups all live in
``training_utils/weight_update``; the one thing a backend supplies is an HF
weight iterator. torchtitan's turns the trainer's DTensor shards into HF-named
full tensors through the model's own ``state_dict_adapter.to_hf`` -- the same
mapping its checkpointer used to load the weights, run in reverse.
"""

from argparse import Namespace


from miles.backends.training_utils.weight_update.hf_weight_iterator import (
    HfWeightIteratorBase,
    WeightUpdatePlacement,
    resolve_placement,
)
from miles.backends.training_utils.weight_update.hf_weight_iterator.atomic_groups import get_hf_atomic_update_groups


class TitanHfWeightIterator(HfWeightIteratorBase):
    """Streams a TitanTrainer's weights as HF-named tensors.

    ``model`` is the trainer rather than a module: the trainer owns the model
    parts and the adapter, and its ``hf_weights`` already completes the stream
    on every rank -- dp/tp shards reassembled, pp- and ep-partial tensors
    broadcast from their owners -- which is why every dim is forced gathered
    regardless of what the protocol asks for. Only the tensors handed out are
    materialized, one at a time, so a 30B never exists unsharded on a rank.
    """

    forced_placement = WeightUpdatePlacement(gather_pp=True)

    def _iter_hf_param_units(self, weights, *, materialize):
        # ``weights`` is None by contract here: the trainer reads its live parts.
        # Non-materializing ranks still walk the stream, since producing each
        # tensor is a collective every rank has to join.
        for name, tensor in self.model.hf_weights():
            if materialize:
                yield [(name, tensor)]

    def _hf_atomic_update_groups(self):
        # DeepSeek's MLA down-projections are fused by sglang from two HF
        # tensors; whether the model has them is a fact of its architecture.
        q_lora_rank = getattr(self.model.model_config, "q_lora_rank", None) or None
        return get_hf_atomic_update_groups(self.model_name, q_lora_rank=q_lora_rank)

    def _iter_hf_adapter_units(self, lora_name, adapter, *, materialize):
        raise NotImplementedError("the torchtitan backend has no LoRA")


def get_hf_weight_iterator(
    args: Namespace,
    trainer,
    *,
    required_placement: WeightUpdatePlacement,
    model_name: str,
    quantization_config: dict | None,
) -> HfWeightIteratorBase:
    return TitanHfWeightIterator(
        args,
        trainer,
        placement=resolve_placement(required_placement, TitanHfWeightIterator.forced_placement),
        model_name=model_name,
        quantization_config=quantization_config,
    )
