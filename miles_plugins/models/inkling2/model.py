"""Inkling2 model assembly: TransformerConfig, layer/block specs, and the GPTModel subclass
plus provider that miles' custom-model-provider path loads."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.mappings import copy_to_tensor_model_parallel_region
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.linear_cross_entropy import LinearCrossEntropyModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig

from miles.backends.megatron_utils.megatron_to_hf.inkling2 import (
    inkling2_quantization_config,
    load_inkling2_text_config,
)
from miles_plugins.models.inkling2.layers import (
    Inkling2Attention,
    Inkling2ColumnParallelGroupedLinear,
    Inkling2GroupedMLP,
    Inkling2RMSNorm,
    Inkling2Router,
    Inkling2RowParallelGroupedLinear,
    Inkling2TransformerLayer,
)
from miles_plugins.models.inkling2.ops import softcap_logits

if TYPE_CHECKING:
    from miles.utils.args.runtime import TrainerConfig


@dataclass(frozen=True)
class Inkling2Config:
    """The HF ``text_config`` fields the Megatron implementation reads."""

    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    swa_num_attention_heads: int
    swa_num_key_value_heads: int
    swa_head_dim: int
    sliding_window_size: int
    d_rel: int
    rel_extent: int
    local_layer_ids: frozenset[int]
    sconv_kernel_size: int
    rms_norm_eps: float
    qk_l2_clip_threshold: float
    rel_l2_clip_threshold: float
    use_value_norm_no_gamma: bool
    log_scaling_n_floor: float
    log_scaling_alpha: float
    n_layers_per_attnres_block: int
    attn_res_normalize_values: bool
    route_scale: float
    mlp_gate_softcap: float
    mlp_up_softcap: float
    final_logit_softcapping: float
    unpadded_vocab_size: int
    # Per layer, the format the trainer fake-quantizes trained expert weights to; empty keeps bf16.
    expert_fake_quant: tuple[str, ...] = ()

    @classmethod
    def from_text_config(cls, text_cfg: dict) -> Inkling2Config:
        assert text_cfg["n_shared_experts"] == 0, "Inkling2 shared experts are not implemented"
        assert not text_cfg["pre_norm_pre_affine"] and not text_cfg["final_norm_pre_affine"]
        assert text_cfg["use_gate_bias"]
        names = cls.__dataclass_fields__.keys() - {"local_layer_ids", "expert_fake_quant"}
        return cls(local_layer_ids=frozenset(text_cfg["local_layer_ids"]), **{n: text_cfg[n] for n in names})


def _expert_formats(hf_checkpoint: str, text_cfg: dict) -> tuple[str, ...]:
    formats = inkling2_quantization_config(hf_checkpoint)["expert_formats"]
    per_layer = []
    for layer in range(text_cfg["num_hidden_layers"]):
        w13, w2 = (formats[f"model.llm.layers.{layer}.mlp.experts.{leaf}"] for leaf in ("w13_weight", "w2_weight"))
        assert w13 == w2, f"layer {layer}: w13 {w13} and w2 {w2} differ"
        per_layer.append(w13)
    return tuple(per_layer)


def build_inkling2_config(text_cfg: dict, args) -> TransformerConfig:
    config = TransformerConfig(
        num_layers=text_cfg["num_hidden_layers"],
        hidden_size=text_cfg["hidden_size"],
        num_attention_heads=text_cfg["num_attention_heads"],
        num_query_groups=text_cfg["num_key_value_heads"],
        kv_channels=text_cfg["head_dim"],
        ffn_hidden_size=text_cfg["intermediate_size"],
        hidden_dropout=0.0,
        attention_dropout=0.0,
        add_bias_linear=False,
        gated_linear_unit=True,
        normalization="RMSNorm",
        layernorm_epsilon=text_cfg["rms_norm_eps"],
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        expert_model_parallel_size=args.expert_model_parallel_size,
        expert_tensor_parallel_size=args.expert_tensor_parallel_size or 1,
        context_parallel_size=args.context_parallel_size,
        sequence_parallel=args.sequence_parallel,
        variable_seq_lengths=True,
        num_moe_experts=text_cfg["n_routed_experts"],
        moe_router_topk=text_cfg["num_experts_per_tok"],
        moe_ffn_hidden_size=text_cfg["intermediate_size"],
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_bias_update_rate=0.0,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        recompute_granularity=args.recompute_granularity,
        recompute_method=args.recompute_method,
        recompute_num_layers=args.recompute_num_layers,
        bf16=args.bf16,
        params_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        pipeline_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        # Local and full-attention layers own different parameters, so layers are not stacked.
        hetereogenous_dist_checkpoint=True,
    )
    assert config.context_parallel_size == 1, "Inkling2 supports CP=1"
    assert not config.sequence_parallel, "Inkling2 does not support sequence parallelism"
    config.inkling2 = Inkling2Config.from_text_config(text_cfg)
    return config


def get_inkling2_block_spec(config: TransformerConfig) -> TransformerBlockSubmodules:
    spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=config.num_moe_experts, moe_grouped_gemm=True, qk_layernorm=False
    )
    moe = spec.submodules.mlp
    moe_submodules = moe.keywords["submodules"]
    moe_submodules.router = Inkling2Router
    expert_submodules = moe_submodules.experts.keywords["submodules"]
    expert_submodules.linear_fc1 = Inkling2ColumnParallelGroupedLinear
    expert_submodules.linear_fc2 = Inkling2RowParallelGroupedLinear
    moe_submodules.experts = partial(Inkling2GroupedMLP, submodules=expert_submodules)
    moe_submodules.shared_experts = None
    spec.module = Inkling2TransformerLayer
    spec.submodules.input_layernorm = IdentityOp
    spec.submodules.self_attention = ModuleSpec(module=Inkling2Attention)
    spec.submodules.pre_mlp_layernorm = Inkling2RMSNorm
    return TransformerBlockSubmodules(layer_specs=[spec] * config.num_layers, layer_norm=Inkling2RMSNorm)


class Inkling2LMHead(LinearCrossEntropyModule):
    """The LM head with fp32 logits: bf16 operands, fp32 accumulation and output, as the
    engine's fp32 unembed computes them. Rounding logits of magnitude ~30 to bf16 before
    the soft cap would cost up to 0.06 nats per token."""

    def forward(self, input_, weight=None, runtime_gather_output=None, **kwargs):
        assert not kwargs.get("output_cross_entropy_loss"), "Inkling2 computes logprobs from logits"
        assert not runtime_gather_output and not self.gather_output
        weight = self.weight if weight is None else weight
        input_ = copy_to_tensor_model_parallel_region(input_, group=self.tp_group)
        return F.linear(input_.float(), weight.float()), None


class Inkling2GPTModel(GPTModel):
    """GPTModel with the normalized embedding and soft-capped, vocab-masked fp32 logits."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.pre_process:
            self.embed_norm = Inkling2RMSNorm(self.config, self.config.hidden_size, self.config.inkling2.rms_norm_eps)
        if self.post_process:
            # GPTModel hardcodes the output layer class; rebinding keeps `output_layer.weight`.
            assert type(self.output_layer) is LinearCrossEntropyModule
            self.output_layer.__class__ = Inkling2LMHead

    def _preprocess(self, input_ids, position_ids, decoder_input=None, *args, **kwargs):
        if decoder_input is None and self.pre_process:
            decoder_input = self.embed_norm(self.embedding(input_ids=input_ids, position_ids=position_ids))
        return super()._preprocess(input_ids, position_ids, decoder_input, *args, **kwargs)

    def _scale_logits(self, logits):
        inkling2 = self.config.inkling2
        logits = softcap_logits(logits, inkling2.final_logit_softcapping)
        vocab_start = self.output_layer.output_size_per_partition * self.pg_collection.tp.rank()
        padded = torch.arange(logits.shape[-1], device=logits.device) + vocab_start >= inkling2.unpadded_vocab_size
        # The engine masks with -inf; a finite minimum gives the same probabilities (exp
        # underflows to 0) and keeps p * log p finite in the entropy metric.
        return logits.masked_fill(padded, torch.finfo(logits.dtype).min)


def inkling2_model_provider(pre_process=True, post_process=True, vp_stage=None, *, args: TrainerConfig):
    text_cfg = load_inkling2_text_config(args.hf_checkpoint)
    config = build_inkling2_config(text_cfg, args.backend)
    config.inkling2 = replace(config.inkling2, expert_fake_quant=_expert_formats(args.hf_checkpoint, text_cfg))
    return Inkling2GPTModel(
        config=config,
        transformer_layer_spec=get_inkling2_block_spec(config),
        vocab_size=text_cfg["vocab_size"],
        max_sequence_length=args.backend.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        position_embedding_type="none",
        share_embeddings_and_output_weights=False,
        vp_stage=vp_stage,
    )
