import copy

import torch
import torch.nn as nn
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELinear,
    TERowParallelLinear,
)
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_layer import TransformerLayer, get_transformer_layer_offset

from miles.kernels.attention.delta_rule import DeltaRuleHeads, KimiDeltaRule
from miles_plugins.models.kimi_k3.ops import KimiRMSNorm, attn_res_aggregate
from miles_plugins.models.kimi_k3.pipeline import bank_num_rows, pack_stage_boundary, unpack_stage_boundary
from miles_plugins.models.linear_attn import KimiDeltaAttention, LinearAttentionLayer


def _mark_tp_replicated(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.sum_gradients_across_tp_domain = True


def _linear(module: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    output, bias = module(inputs)
    assert bias is None
    return output


class KimiK3Attention(MegatronModule):
    """K3's MLA layers; the KDA layers are :class:`KimiK3KDAAttention`."""

    is_kda = False

    def __init__(
        self,
        config,
        layer_number: int,
        cp_comm_type: str | None = None,
        pg_collection=None,
        name: str | None = None,
    ) -> None:
        super().__init__(config=config)
        del name  # build_module forwards the module path; K3 constructs its submodules directly
        self.cp_comm_type = cp_comm_type

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        else:
            assert hasattr(pg_collection, "tp") and hasattr(pg_collection, "cp")
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp
        self.tp_size = self.tp_group.size()
        self.cp_group = pg_collection.cp
        self.cp_size = self.cp_group.size()
        self.sequence_parallel = config.sequence_parallel
        self.linear_config = copy.copy(config)
        self.linear_config.sequence_parallel = False

        self.layer_idx = layer_number - 1
        assert layer_number not in config.kimi_kda_layers, "K3's KDA layers are KimiK3KDAAttention"
        self._init_mla(config)

    def _duplicated_linear(self, input_size: int, output_size: int) -> TELinear:
        return TELinear(
            input_size,
            output_size,
            config=self.linear_config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

    def _column_linear(self, input_size: int, output_size: int) -> TEColumnParallelLinear:
        return TEColumnParallelLinear(
            input_size,
            output_size,
            config=self.linear_config,
            init_method=self.config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.tp_group,
        )

    def _row_linear(self, input_size: int, output_size: int) -> TERowParallelLinear:
        return TERowParallelLinear(
            input_size,
            output_size,
            config=self.linear_config,
            init_method=self.config.init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.tp_group,
        )

    def _init_mla(self, config) -> None:
        hidden_size = config.hidden_size
        device = torch.cuda.current_device()
        dtype = config.params_dtype
        self.num_heads = config.num_attention_heads
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = self.num_heads // self.tp_size
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_head_dim
        self.qk_extra_head_dim = config.qk_pos_emb_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_extra_head_dim

        self.q_a_proj = self._duplicated_linear(hidden_size, self.q_lora_rank)
        self.q_a_layernorm = KimiRMSNorm(self.q_lora_rank, config.layernorm_epsilon, device=device, dtype=dtype)
        self.q_b_proj = self._column_linear(self.q_lora_rank, self.num_heads * self.q_head_dim)
        self.kv_a_proj_with_mqa = self._duplicated_linear(hidden_size, self.kv_lora_rank + self.qk_extra_head_dim)
        self.kv_a_layernorm = KimiRMSNorm(self.kv_lora_rank, config.layernorm_epsilon, device=device, dtype=dtype)
        self.kv_b_proj = self._column_linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        )
        self.g_proj = self._column_linear(hidden_size, self.num_heads * self.v_head_dim)
        self.o_proj = self._row_linear(self.num_heads * self.v_head_dim, hidden_size)

        # Megatron's MLA core carries CP and varlen; only its module assumes the rotated 64-wide field K3 lacks
        self.core_attention = TEDotProductAttention(
            config=self.config,
            layer_number=self.layer_idx + 1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            softmax_scale=self.q_head_dim**-0.5,
            k_channels=self.q_head_dim,
            v_channels=self.v_head_dim,
            cp_comm_type=self.cp_comm_type,
            pg_collection=self.pg_collection,
        )

    def _forward_mla(
        self,
        hidden_states: torch.Tensor,
        packed_seq_params: PackedSeqParams | None,
    ) -> torch.Tensor:
        # stay in TE's sbhd [s, b, ...] layout so q/k/v need no transpose; the KDA kernels want [b, s, ...]
        x = hidden_states
        query = _linear(
            self.q_b_proj,
            self.q_a_layernorm(_linear(self.q_a_proj, x)),
        )
        query = query.view(*query.shape[:-1], self.local_num_heads, self.q_head_dim)

        compressed_kv = _linear(self.kv_a_proj_with_mqa, x)
        kv_latent, key_extra = torch.split(
            compressed_kv,
            [self.kv_lora_rank, self.qk_extra_head_dim],
            dim=-1,
        )
        key_value = _linear(self.kv_b_proj, self.kv_a_layernorm(kv_latent))
        key_value = key_value.view(
            *key_value.shape[:-1],
            self.local_num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        key_nope, value = torch.split(
            key_value,
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )
        # TE classifies the qkv layout from strides, so the sliced value needs canonical ones too
        value = value.contiguous()
        key_extra = copy_to_tensor_model_parallel_region(key_extra, group=self.tp_group)
        key_extra = key_extra.unsqueeze(-2).expand(*key_nope.shape[:-1], -1)
        key = torch.cat((key_nope, key_extra), dim=-1)

        # thd packing wants 3D [t, h, d]; mirror Megatron's Attention.forward squeeze/reshape
        is_thd = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        if is_thd:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)
        # TE returns the head dims already fused, so there is no flatten here.
        output = self.core_attention(
            query,
            key,
            value,
            None,
            packed_seq_params=packed_seq_params,
            attn_mask_type=AttnMaskType.causal,
        )
        if is_thd:
            output = output.reshape(output.size(0), 1, -1)
        output = output * torch.sigmoid(_linear(self.g_proj, x))
        return _linear(self.o_proj, output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        key_value_states: torch.Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        rotary_pos_emb: torch.Tensor | None = None,
        rotary_pos_cos: torch.Tensor | None = None,
        rotary_pos_sin: torch.Tensor | None = None,
        rotary_pos_cos_sin: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: int | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        del (
            attention_mask,
            key_value_states,
            inference_context,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            rotary_pos_cos_sin,
            attention_bias,
            sequence_len_offset,
            kwargs,
        )
        if self.sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states,
                tensor_parallel_output_grad=False,
                group=self.tp_group,
            )
        output = self._forward_mla(hidden_states, packed_seq_params)
        if self.sequence_parallel:
            output = scatter_to_sequence_parallel_region(output, group=self.tp_group)
        return output, None


class KimiK3KDAAttention(LinearAttentionLayer):
    """K3's KDA layers on the shared head-sharded delta-rule layer. The ``self_attention`` constructor
    signature K3's layer spec builds with; the input norm is the transformer layer's, so none here."""

    is_kda = True

    def __init__(
        self,
        config,
        layer_number: int,
        cp_comm_type: str | None = None,
        pg_collection=None,
        name: str | None = None,
    ) -> None:
        del cp_comm_type, name
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        assert layer_number in config.kimi_kda_layers
        heads = DeltaRuleHeads(
            num_k_heads=config.kimi_linear_num_heads,
            num_v_heads=config.kimi_linear_num_heads,
            head_k_dim=config.kimi_linear_head_dim,
            head_v_dim=config.kimi_linear_head_dim,
        )
        core = KimiDeltaAttention(
            config,
            heads,
            KimiDeltaRule(config.kimi_kda_gate_lower_bound),
            config.kimi_linear_conv_kernel_size,
            config.layernorm_epsilon,
            pg_collection.tp,
        )
        super().__init__(config, core, nn.Identity(), pg_collection, allgather_cp=False)


class KimiK3TransformerLayer(TransformerLayer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert self.config.hidden_dropout == 0.0, "Kimi K3 requires hidden dropout 0"

        hidden_size = self.config.hidden_size
        eps = self.config.layernorm_epsilon
        device = torch.cuda.current_device()
        dtype = self.config.params_dtype
        self.attn_res_block_size = self.config.kimi_attn_res_block_size
        self.self_attention_res_norm = KimiRMSNorm(hidden_size, eps, device=device, dtype=dtype)
        self.self_attention_res_proj = nn.Linear(hidden_size, 1, bias=False, device=device, dtype=dtype)
        self.mlp_res_norm = KimiRMSNorm(hidden_size, eps, device=device, dtype=dtype)
        self.mlp_res_proj = nn.Linear(hidden_size, 1, bias=False, device=device, dtype=dtype)
        _mark_tp_replicated(self.self_attention_res_norm)
        _mark_tp_replicated(self.self_attention_res_proj)
        _mark_tp_replicated(self.mlp_res_norm)
        _mark_tp_replicated(self.mlp_res_proj)

        if self.layer_number == self.config.num_layers:
            self.output_attn_res_norm = KimiRMSNorm(hidden_size, eps, device=device, dtype=dtype)
            self.output_attn_res_proj = nn.Linear(hidden_size, 1, bias=False, device=device, dtype=dtype)
            _mark_tp_replicated(self.output_attn_res_norm)
            _mark_tp_replicated(self.output_attn_res_proj)

        # stage entry/exit layers from the per-rank offsets; VPP is rejected, so vp_stage is None
        pp_size = self.config.pipeline_model_parallel_size
        stage_starts = {get_transformer_layer_offset(self.config, None, r) for r in range(pp_size)}
        layer_idx = self.layer_number - 1
        self.is_stage_entry = layer_idx in stage_starts and layer_idx > 0
        self.is_stage_exit = (layer_idx + 1) in stage_starts and layer_idx + 1 < self.config.num_layers

    @staticmethod
    def _add_bias(output_with_bias: tuple[torch.Tensor, torch.Tensor | None]) -> torch.Tensor:
        output, bias = output_with_bias
        return output if bias is None else output + bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        rotary_pos_emb: torch.Tensor | None = None,
        rotary_pos_cos: torch.Tensor | None = None,
        rotary_pos_sin: torch.Tensor | None = None,
        rotary_pos_cos_sin: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del context_mask, kwargs
        layer_idx = self.layer_number - 1

        if context is not None:
            prefix_sum = hidden_states
            block_residual = context
        elif layer_idx == 0:
            prefix_sum = hidden_states
            block_residual = hidden_states.new_empty(*hidden_states.shape[:-1], 0, hidden_states.shape[-1])
        else:
            assert self.is_stage_entry, "Attention-residual snapshot bank is missing"
            prefix_sum, block_residual = unpack_stage_boundary(
                hidden_states,
                self.config.hidden_size,
                bank_num_rows(layer_idx, self.attn_res_block_size),
            )

        if block_residual.shape[-2] > 0:
            attention_input = attn_res_aggregate(
                prefix_sum,
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.input_layernorm,
            )
        else:
            attention_input = self.input_layernorm(prefix_sum)

        is_block_write_layer = layer_idx % self.attn_res_block_size == 0
        if is_block_write_layer:
            block_residual = torch.cat((block_residual, prefix_sum.unsqueeze(-2)), dim=-2)

        attention_output = self._add_bias(
            self.self_attention(
                attention_input,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )
        )
        prefix_sum = attention_output if is_block_write_layer else prefix_sum + attention_output

        mlp_input = attn_res_aggregate(
            prefix_sum,
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
            self.pre_mlp_layernorm,
        )
        mlp_kwargs = {"padding_mask": padding_mask}
        if self.is_moe_layer:
            mlp_kwargs["input_ids"] = input_ids
        mlp_output = self._add_bias(self.mlp(mlp_input, **mlp_kwargs))
        prefix_sum = prefix_sum + mlp_output

        if self.layer_number == self.config.num_layers:
            prefix_sum = attn_res_aggregate(
                prefix_sum,
                block_residual,
                self.output_attn_res_proj,
                self.output_attn_res_norm,
                nn.Identity(),
            )

        if self.is_stage_exit:
            return pack_stage_boundary(prefix_sum, block_residual), block_residual
        return prefix_sum, block_residual
