"""Bridge / LoRA model setup helpers.

Extracted from ``model.py`` to keep the main training module focused on
forward / backward / optimizer logic.
"""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass

from megatron.core.utils import get_attr_wrapped_model

from .lora_utils import create_lora_instance


@dataclass
class _BridgeWrapperConfig:
    """Configuration for Megatron-Bridge module wrapping."""

    is_value_model: bool = False
    wrap_with_ddp: bool = True
    use_distributed_optimizer: bool = True


def _ensure_model_list(model):
    return model if isinstance(model, list) else [model]


def _make_value_model_hook(hidden_size: int, sequence_parallel: bool):
    """Create a pre-wrap hook that replaces the output layer with a value head."""
    from megatron.core import parallel_state

    from .model_provider import LinearForLastLayer

    def hook(model):
        model_post_process = []
        if (
            parallel_state.get_pipeline_model_parallel_world_size() > 1
            and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None
        ):
            for i in range(parallel_state.get_virtual_pipeline_model_parallel_world_size()):
                model_post_process.append(parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=i))
        else:
            model_post_process.append(parallel_state.is_pipeline_last_stage())

        model_list = _ensure_model_list(model)
        assert len(model_post_process) == len(model_list), "Model list length and post process list length must match."

        for index, model_chunk in enumerate(model_list):
            if not model_post_process[index]:
                continue
            model_chunk.output_layer = LinearForLastLayer(
                input_size=hidden_size,
                output_size=1,
                sequence_parallel=sequence_parallel,
            )

    return hook


def _get_model_config_from_wrapped(model):
    return get_attr_wrapped_model(model, "config", allow_none=False)


def _setup_lora_model_via_bridge(args: Namespace) -> list:
    """Build Megatron model with LoRA using Megatron-Bridge.

    This handles:
    1. Creating the Bridge and Provider
    2. Creating and registering the LoRA pre-wrap hook
    3. Registering value-model hooks if needed
    4. Building the DDP-wrapped model

    Args:
        args: Training arguments.

    Returns:
        List of DDP-wrapped model chunks with LoRA applied.
    """
    from megatron.bridge import AutoBridge
    from megatron.bridge.training.config import DistributedDataParallelConfig
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)

    provider.tensor_model_parallel_size = args.tensor_model_parallel_size
    provider.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    provider.expert_model_parallel_size = args.expert_model_parallel_size
    provider.expert_tensor_parallel_size = args.expert_tensor_parallel_size
    provider.sequence_parallel = args.sequence_parallel
    provider.virtual_pipeline_model_parallel_size = args.virtual_pipeline_model_parallel_size
    provider.context_parallel_size = args.context_parallel_size
    provider.variable_seq_lengths = True
    provider.moe_token_dispatcher_type = "alltoall"
    provider.moe_router_load_balancing_type = "none"
    provider.finalize()

    lora = create_lora_instance(args)

    def apply_lora_hook(model_chunks):
        transformed = lora(model_chunks, training=True)
        lora.set_params_to_save(transformed)
        return transformed

    provider.register_pre_wrap_hook(apply_lora_hook)

    is_value_model = (
        "ForTokenClassification" in hf_config.architectures[0]
        or "ForSequenceClassification" in hf_config.architectures[0]
    )
    if is_value_model:
        hidden_size = hf_config.text_config.hidden_size if hasattr(hf_config, "text_config") else hf_config.hidden_size
        provider.register_pre_wrap_hook(_make_value_model_hook(hidden_size, provider.sequence_parallel))

    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
    ddp_config.finalize()

    model = provider.provide_distributed_model(wrap_with_ddp=True, ddp_config=ddp_config)
    return model
