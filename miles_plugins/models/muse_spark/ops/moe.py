import torch
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.transformer.module import mark_keep_in_fp32
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.spec_utils import ModuleSpec
from torch import nn

from miles_plugins.models.muse_spark.ops.norms import (
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
    calculate_gain_center,
)


class MuseSpark1xTopKRouter(TopKRouter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # by dtype rather than by name, so router buffers MCore adds later stay FP32 too
        for tensor in [*self.parameters(), *self.buffers()]:
            if tensor.dtype == torch.float32:
                mark_keep_in_fp32(tensor)

    def reset_parameters(self):
        # unlike TopKRouter, no params_dtype cast: the Spark router stays FP32
        for param in (self.weight, self.bias):
            if param is None:
                continue
            if self.config.perform_initialization:
                self.config.init_method(param)
            param.sequence_parallel = self.config.sequence_parallel

    def gating(self, input):
        if input.numel() == 0:
            # MCore's inferred output dimension is ambiguous for zero tokens.
            logits = torch.nn.functional.linear(input.float(), self.weight, self.bias)
        else:
            logits = super().gating(input)
        multiplier = self.config.spark.metap_router_output_multiplier
        return logits * multiplier if multiplier is not None else logits


class MuseSpark1xMoELayer(MoELayer):
    def __init__(
        self,
        config,
        submodules=None,
        layer_number=None,
        pg_collection=None,
        is_mtp_layer=False,
        name=None,
    ):
        super().__init__(config, submodules, layer_number, pg_collection, is_mtp_layer, name)
        spark = config.spark
        dim = spark.expert_hidden_size
        self.pre_expert_norm = (
            MuseSpark1xRMSNorm(
                dim,
                eps=spark.rms_norm_eps,
                has_weight=spark.use_ffn_learnable_scales,
                weight_offset=1.0,
                config=config,
            )
            if spark.moe_use_pre_expert_norm
            else nn.Identity()
        )
        post_expert_norm = MuseSpark1xRMSNormWithInputScale if spark.use_ffn_learnable_scales else MuseSpark1xRMSNorm
        self.post_expert_norm = (
            post_expert_norm(
                dim,
                eps=spark.rms_post_norm_eps,
                has_weight=spark.use_ffn_learnable_scales or spark.post_norm_affine,
                weight_offset=(
                    calculate_gain_center(spark.post_norm_gain_center_type, 2 * layer_number)
                    if spark.norm_zero_centered_gamma
                    else 0.0
                ),
                config=config,
            )
            if spark.moe_use_post_expert_norm
            else nn.Identity()
        )

    def preprocess(self, hidden_states, probs, routing_map):
        hidden_states = hidden_states.to(self.config.params_dtype)
        if self.config.moe_latent_size:
            hidden_states, _ = self.fc1_latent_proj(hidden_states)
        hidden_states = self.pre_expert_norm(hidden_states)
        return self.token_dispatcher.dispatch_preprocess(hidden_states, routing_map, probs)

    def postprocess(self, output, shared_expert_output):
        # Reverse AlltoAll still returns individual, unweighted token-expert rows here.
        output = self.post_expert_norm(output)
        return super().postprocess(output, shared_expert_output)


def get_muse_spark_1x_moe_spec(config):
    return ModuleSpec(
        module=MuseSpark1xMoELayer,
        submodules=MoESubmodules(
            router=MuseSpark1xTopKRouter,
            experts=TESpecProvider().grouped_mlp_modules(True),
        ),
    )
