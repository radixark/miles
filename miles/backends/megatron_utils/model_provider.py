# Adapt from https://github.com/NVIDIA/Megatron-LM/blob/b1efb3c7126ef7615e8c333432d76e08038e17ff/pretrain_gpt.py
import argparse
import inspect
import logging
from typing import Literal

import torch
from megatron.core import tensor_parallel
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import core_transformer_config_from_args

from miles.utils.misc import load_function
from miles.utils.replay_base import routing_replay_manager

logger = logging.getLogger(__name__)


_PARAM_ATTRS_TO_PRESERVE = (
    "tensor_model_parallel",
    "partition_dim",
    "partition_stride",
    "sequence_parallel",
    "allreduce",
    "parallel_mode",
    "shared",
    "is_embedding_or_output_parameter",
    "skip_backward_post_hook",
    "overwrite_main_grad",
    "grad_added_to_main_grad",
    "main_grad",
)

try:
    from megatron.core.fp8_utils import dequantize_fp8_tensor, is_float8tensor
except ImportError:
    dequantize_fp8_tensor = None

    def is_float8tensor(_tensor: torch.Tensor) -> bool:
        return False


def _get_active_te_recipe(module: torch.nn.Module):
    """Get the active TE quantization recipe for current module mode."""
    te_quant_params = getattr(module, "te_quant_params", None)
    if te_quant_params is None:
        return None
    if module.training or te_quant_params.evaluation_recipe is None:
        return te_quant_params.training_recipe
    return te_quant_params.evaluation_recipe


def _is_high_precision_te_recipe(recipe) -> bool:
    """Whether TE recipe explicitly forces high precision (bf16/fp16/fp32)."""
    return (
        recipe is not None
        and getattr(recipe, "fp8_quantization_recipe", None) is None
        and getattr(recipe, "fp4_quantization_recipe", None) is None
    )


def _copy_param_attrs(src: torch.nn.Parameter, dst: torch.nn.Parameter) -> None:
    """Preserve Megatron/TE parallel metadata after parameter replacement."""
    for attr_name in _PARAM_ATTRS_TO_PRESERVE:
        if hasattr(src, attr_name):
            setattr(dst, attr_name, getattr(src, attr_name))


def _dequantize_primary_fp8_params_for_hp_module(module: torch.nn.Module, target_dtype: torch.dtype) -> int:
    """
    Convert FP8 primary params to high precision for TE modules that run HP compute.

    This avoids mixed mode "quantized weights + non-quantized compute", which can
    break fused wgrad accumulation paths in TE grouped linear backward.
    """
    if dequantize_fp8_tensor is None:
        raise RuntimeError("--fp8-param-gather reconciliation requires megatron.core.fp8_utils.dequantize_fp8_tensor.")

    def _resolve_fp8_source(param: torch.nn.Parameter) -> torch.Tensor | None:
        if is_float8tensor(param):
            return param
        tensor_data = getattr(param, "data", None)
        if tensor_data is not None and is_float8tensor(tensor_data):
            return tensor_data
        return None

    converted = 0
    for param_name, param in list(module.named_parameters(recurse=False)):
        if param is None:
            continue
        fp8_src = _resolve_fp8_source(param)
        if fp8_src is None:
            continue

        dequantized = dequantize_fp8_tensor(fp8_src).to(dtype=target_dtype).detach()
        new_param = torch.nn.Parameter(dequantized, requires_grad=param.requires_grad)
        _copy_param_attrs(param, new_param)
        module.register_parameter(param_name, new_param)
        converted += 1

    has_remaining_fp8_primary = any(
        is_float8tensor(p) or is_float8tensor(getattr(p, "data", p))
        for p in module.parameters(recurse=False)
        if p is not None
    )
    if not has_remaining_fp8_primary:
        module.primary_weights_in_fp8 = False
        if hasattr(module, "_fp8_workspaces"):
            module._fp8_workspaces.clear()

    return converted


def _reconcile_te_precision_overrides_for_fp8_param_gather(model: GPTModel, config: TransformerConfig) -> None:
    """
    Align TE parameter storage with per-module precision overrides.

    For modules explicitly configured to high-precision compute by
    `--te-precision-config-file`, FP8 primary weights are dequantized to avoid
    invalid mixed-mode execution in backward.
    """
    converted_modules: list[str] = []
    for module_name, module in model.named_modules():
        if not getattr(module, "primary_weights_in_fp8", False):
            continue
        if not _is_high_precision_te_recipe(_get_active_te_recipe(module)):
            continue

        converted = _dequantize_primary_fp8_params_for_hp_module(module, config.params_dtype)
        if converted == 0:
            raise RuntimeError(
                "Failed to dequantize TE module forced to high precision: "
                f"{module_name}. This module still has FP8 primary weights."
            )
        converted_modules.append(module_name)

    incompatible_modules: list[str] = []
    for module_name, module in model.named_modules():
        if not _is_high_precision_te_recipe(_get_active_te_recipe(module)):
            continue

        has_fp8_primary = any(
            is_float8tensor(p) or is_float8tensor(getattr(p, "data", p))
            for p in module.parameters(recurse=False)
            if p is not None
        )
        if has_fp8_primary:
            incompatible_modules.append(module_name)
    if incompatible_modules:
        sample = ", ".join(incompatible_modules[:8])
        raise RuntimeError(
            "Detected TE modules with high-precision compute but FP8 primary weights after "
            f"reconciliation. Sample modules: {sample}"
        )

    if converted_modules:
        logger.warning(
            "Dequantized %d TE module(s) to match high-precision overrides under --fp8-param-gather.",
            len(converted_modules),
        )
        logger.info(
            "Converted TE modules (first 12): %s",
            ", ".join(converted_modules[:12]),
        )


# Adapt from https://github.com/volcengine/verl/blob/c3b20575d2bc815fcccd84bddb4c0401fc4b632b/verl/models/llama/megatron/layers/parallel_linear.py#L82
class LinearForLastLayer(torch.nn.Linear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features=input_size, out_features=output_size, bias=bias)
        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel:
            self.weight.sequence_parallel = True

        self.weight.data.normal_(mean=0.0, std=0.02)
        if bias:
            self.bias.data.zero_()

    def forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ) -> tuple[torch.Tensor, None]:
        logits = super().forward(input_)
        logits = logits.float()
        if self.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(logits, tensor_parallel_output_grad=False)
        return logits, None


def get_model_provider_func(
    args: argparse.Namespace,
    role: Literal["actor", "critic"] = "actor",
):
    # Support custom model provider path (similar to --custom-rm-path for reward models)
    if getattr(args, "custom_model_provider_path", None):

        def wrapped_model_provider(
            pre_process: bool = True,
            post_process: bool = True,
            vp_stage: int | None = None,
            config: TransformerConfig | None = None,
            pg_collection=None,
        ) -> GPTModel:
            assert config is None, "miles builds the config from args, so it expects config to be None"
            custom_model_provider = load_function(args.custom_model_provider_path)
            # Check if the custom provider supports vp_stage parameter
            has_vp_stage = "vp_stage" in inspect.signature(custom_model_provider).parameters
            if has_vp_stage:
                model = custom_model_provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
            else:
                model = custom_model_provider(pre_process=pre_process, post_process=post_process)

            if args.fp8_param_gather and hasattr(model, "config"):
                _reconcile_te_precision_overrides_for_fp8_param_gather(model, model.config)

            # Apply critic output layer if needed
            if post_process and role == "critic":
                model.output_layer = LinearForLastLayer(
                    input_size=model.config.hidden_size, output_size=1, config=model.config
                )
            return model

        return wrapped_model_provider

    if args.megatron_to_hf_mode == "bridge":
        from megatron.bridge import AutoBridge

        bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
        provider = bridge.to_megatron_provider(load_weights=False)
        # TODO: we should not manually set this...
        provider.tensor_model_parallel_size = args.tensor_model_parallel_size
        provider.pipeline_model_parallel_size = args.pipeline_model_parallel_size
        provider.expert_model_parallel_size = args.expert_model_parallel_size
        provider.expert_tensor_parallel_size = args.expert_tensor_parallel_size
        provider.sequence_parallel = args.sequence_parallel
        provider.context_parallel_size = args.context_parallel_size
        provider.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        provider.variable_seq_lengths = args.variable_seq_lengths
        if hasattr(args, "moe_token_dispatcher_type"):
            provider.moe_token_dispatcher_type = args.moe_token_dispatcher_type
        if getattr(args, "decoder_first_pipeline_num_layers", None) is not None:
            provider.num_layers_in_first_pipeline_stage = args.decoder_first_pipeline_num_layers
        if getattr(args, "decoder_last_pipeline_num_layers", None) is not None:
            provider.num_layers_in_last_pipeline_stage = args.decoder_last_pipeline_num_layers
        if getattr(args, "moe_router_bias_update_rate", None) is not None:
            provider.moe_router_bias_update_rate = args.moe_router_bias_update_rate
        if getattr(args, "moe_aux_loss_coeff", None) is not None:
            provider.moe_aux_loss_coeff = args.moe_aux_loss_coeff
        provider.finalize()

        def wrapped_bridge_provider(
            pre_process: bool = True,
            post_process: bool = True,
            vp_stage: int | None = None,
            config: TransformerConfig | None = None,
            pg_collection=None,
        ) -> GPTModel:
            assert config is None, "miles builds the config from args, so it expects config to be None"
            model = provider.provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
            if args.fp8_param_gather:
                _reconcile_te_precision_overrides_for_fp8_param_gather(model, model.config)
            return model

        return wrapped_bridge_provider

    def model_provider(
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: int | None = None,
        config: TransformerConfig | None = None,
        pg_collection=None,
    ) -> GPTModel:
        """Builds the model.

        If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

        Args:
            pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
            post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


        Returns:
            Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
        """
        use_te = args.transformer_impl == "transformer_engine"

        # Experimental loading arguments from yaml
        assert config is None, "miles builds the config from args, so it expects config to be None"
        config = core_transformer_config_from_args(args)

        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
            # Allow the spec to be a function so that user can use customized Megatron easier.
            if callable(transformer_layer_spec):
                transformer_layer_spec = transformer_layer_spec(args, config, vp_stage)
        else:
            if args.num_experts:
                # Define the decoder block spec
                kwargs = {
                    "use_transformer_engine": use_te,
                }
                if vp_stage is not None:
                    kwargs["vp_stage"] = vp_stage
                transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)
            else:
                # Define the decoder layer spec
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        num_experts=args.num_experts,
                        moe_grouped_gemm=args.moe_grouped_gemm,
                        qk_layernorm=args.qk_layernorm,
                        multi_latent_attention=args.multi_latent_attention,
                        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
                    )
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        num_experts=args.num_experts,
                        moe_grouped_gemm=args.moe_grouped_gemm,
                        qk_layernorm=args.qk_layernorm,
                        multi_latent_attention=args.multi_latent_attention,
                        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
                    )

        kwargs = {
            "config": config,
            "transformer_layer_spec": transformer_layer_spec,
            "vocab_size": args.padded_vocab_size,
            "max_sequence_length": args.max_position_embeddings,
            "pre_process": pre_process,
            "post_process": post_process,
            "fp16_lm_cross_entropy": args.fp16_lm_cross_entropy,
            "parallel_output": True,
            "share_embeddings_and_output_weights": not args.untie_embeddings_and_output_weights,
            "position_embedding_type": args.position_embedding_type,
            "rotary_percent": args.rotary_percent,
            "rotary_base": args.rotary_base,
            "rope_scaling": args.use_rope_scaling,
        }

        if vp_stage is not None:
            kwargs["vp_stage"] = vp_stage

        if args.mtp_num_layers:
            from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec

            mtp_kwargs = {
                "use_transformer_engine": use_te,
            }
            if vp_stage is not None:
                mtp_kwargs["vp_stage"] = vp_stage

            # hard code here to skip r3 registration for mtp layers
            # getattr is required to avoid ckpt conversion errors
            if getattr(args, "use_rollout_routing_replay", False):
                routing_replay_manager.enabled = False
                logger.warning(
                    "Rollout routing replay is not applicable for MTP modules, so skipped replay registration"
                )
            mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, **mtp_kwargs)
            kwargs["mtp_block_spec"] = mtp_block_spec
            if getattr(args, "use_rollout_routing_replay", False):
                routing_replay_manager.enabled = True

        model = GPTModel(**kwargs)

        if args.fp8_param_gather:
            _reconcile_te_precision_overrides_for_fp8_param_gather(model, config)

        if post_process and role == "critic":
            model.output_layer = LinearForLastLayer(input_size=config.hidden_size, output_size=1, config=config)

        return model

    return model_provider
