from __future__ import annotations

import logging

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.utils import unwrap_model

from miles.backends.megatron_utils.lora.slots import create_multi_lora_instance, slice_lora_to_rank

logger = logging.getLogger(__name__)

_TARGET_MODULES = (
    "decoder.layers.*.mlp.linear_fc1",
    "decoder.layers.*.mlp.linear_fc2",
    "decoder.layers.*.mlp.shared_experts.linear_fc1",
    "decoder.layers.*.mlp.shared_experts.linear_fc2",
)


def wrap_model_provider_with_glm5_next_multi_lora(provider_func, args):
    """Apply the existing multi-slot transform to the native GLM-5.3 model.

    This is deliberately limited to dense and shared-expert MLPs.  The native
    raw exporter below does not yet implement the EP-packed routed experts.
    """

    unsupported = [target for target in args.hf_lora_targets if ".mlp.experts." in target]
    if unsupported:
        raise NotImplementedError("native GLM-5.3 Multi-LoRA smoke does not yet export routed experts; exclude model.language_model.layers.*.mlp.experts.*")

    def wrapped(*provider_args, **provider_kwargs):
        from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear

        model = provider_func(*provider_args, **provider_kwargs)
        for parameter in model.parameters():
            parameter.requires_grad = False
        transform = create_multi_lora_instance(args, target_modules=list(_TARGET_MODULES))
        transformed = transform([model], training=True)

        # The reference checkpoint contains only the frozen base model.  The
        # stock MultiLoRA sharded-state hook asks the loader to restore fresh
        # adapter slots too, which is both unnecessary and incompatible with
        # this PP-resharded base checkpoint.  Slot weights/state use Miles'
        # dedicated per-slot checkpoint path after the service starts.
        def base_only_sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
            return self.to_wrap.sharded_state_dict(prefix, sharded_offsets, metadata)

        MultiLoRALinear.sharded_state_dict = base_only_sharded_state_dict
        logger.info("Applied native GLM-5.3 Multi-LoRA smoke targets: %s", _TARGET_MODULES)
        return transformed[0]

    return wrapped


def _gather_tp(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    if tp_size == 1:
        return tensor
    parts = [torch.empty_like(tensor) for _ in range(tp_size)]
    dist.all_gather(parts, tensor.contiguous(), group=parallel_state.get_tensor_model_parallel_group())
    return torch.cat(parts, dim=dim)


def _export_linear(wrapper, slot: int, rank: int, hf_prefix: str, projection: str):
    adapter = wrapper.adapters[slot]
    a_dim = 1 if adapter.input_is_parallel else 0
    lora_a = _gather_tp(adapter.linear_in.weight, a_dim)
    if projection == "gate_up":
        # Megatron keeps every TP-local fused FC1 shard as
        # ``[gate_local | up_local]`` so SwiGLU can split its local output.
        # Gathering the fused shards before splitting would interleave those
        # local halves and corrupt both HF projections.
        local_gate_b, local_up_b = adapter.linear_out.weight.chunk(2, dim=0)
        gate_b = _gather_tp(local_gate_b, 0)
        up_b = _gather_tp(local_up_b, 0)
        return [
            (f"{hf_prefix}gate_proj.lora_A.weight", slice_lora_to_rank("lora_A", lora_a, rank)),
            (f"{hf_prefix}gate_proj.lora_B.weight", slice_lora_to_rank("lora_B", gate_b, rank)),
            (f"{hf_prefix}up_proj.lora_A.weight", slice_lora_to_rank("lora_A", lora_a, rank)),
            (f"{hf_prefix}up_proj.lora_B.weight", slice_lora_to_rank("lora_B", up_b, rank)),
        ]
    assert projection == "down"
    lora_b = _gather_tp(adapter.linear_out.weight, 0)
    return [
        (f"{hf_prefix}down_proj.lora_A.weight", slice_lora_to_rank("lora_A", lora_a, rank)),
        (f"{hf_prefix}down_proj.lora_B.weight", slice_lora_to_rank("lora_B", lora_b, rank)),
    ]


def export_glm5_next_multi_lora_hf(model_chunks, adapter):
    """Export one slot with HF names understood by the GLM-5.3 SGLang fork."""
    from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear

    named_tensors = []
    for model in unwrap_model(model_chunks):
        for layer in model.decoder.layers:
            layer_idx = layer.layer_number - 1
            mlp = layer.mlp
            if isinstance(getattr(mlp, "linear_fc1", None), MultiLoRALinear):
                prefix = f"model.language_model.layers.{layer_idx}.mlp."
                named_tensors.extend(_export_linear(mlp.linear_fc1, adapter.slot, adapter.rank, prefix, "gate_up"))
                named_tensors.extend(_export_linear(mlp.linear_fc2, adapter.slot, adapter.rank, prefix, "down"))
            shared = getattr(mlp, "shared_experts", None)
            if shared is not None and isinstance(getattr(shared, "linear_fc1", None), MultiLoRALinear):
                prefix = f"model.language_model.layers.{layer_idx}.mlp.shared_experts."
                named_tensors.extend(_export_linear(shared.linear_fc1, adapter.slot, adapter.rank, prefix, "gate_up"))
                named_tensors.extend(_export_linear(shared.linear_fc2, adapter.slot, adapter.rank, prefix, "down"))
    if not named_tensors:
        raise RuntimeError("native GLM-5.3 Multi-LoRA export found no adapters")
    return named_tensors
