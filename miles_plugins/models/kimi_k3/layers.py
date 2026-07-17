import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.modules import FusedRMSNormGated, ShortConvolution
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TELinear, TERowParallelLinear
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group, make_sharded_tensors_for_checkpoint

from miles.backends.megatron_utils.fp32_param_utils import mark_param_dtype
from miles_plugins.models.kimi_k3.ops import KimiRMSNorm, attn_res_aggregate, sglang_kda


def _mark_tp_replicated(module: nn.Module, *, reduction: str = "average") -> None:
    assert reduction in ("average", "sum")
    for parameter in module.parameters():
        setattr(parameter, f"{reduction}_gradients_across_tp_domain", True)


def _linear(module: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    output, bias = module(inputs)
    assert bias is None
    return output


class KimiK3ShortConvolution(ShortConvolution):
    def __init__(self, *args, tp_group, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tp_group = tp_group
        mark_param_dtype(self.weight, torch.float32)
        set_tensor_model_parallel_attributes(self.weight, True, 0, 1)

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: dict | None = None,
    ) -> ShardedStateDict:
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        return make_sharded_tensors_for_checkpoint(
            self.state_dict(prefix="", keep_vars=True),
            prefix,
            {"weight": 0},
            sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )


class KimiK3Attention(MegatronModule):
    def __init__(
        self,
        config,
        layer_number: int,
        cp_comm_type: str | None = None,
        pg_collection=None,
    ) -> None:
        super().__init__(config=config)
        del cp_comm_type

        assert config.context_parallel_size == 1, "Kimi K3 attention initially requires CP=1"
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp"])
        else:
            assert hasattr(pg_collection, "tp")
        self.tp_group = pg_collection.tp
        self.tp_size = self.tp_group.size()
        self.sequence_parallel = config.sequence_parallel
        self.linear_config = copy.copy(config)
        self.linear_config.sequence_parallel = False

        self.layer_idx = layer_number - 1
        self.is_kda = layer_number in config.kimi_kda_layers
        if self.is_kda:
            self._init_kda(config)
        else:
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

    def _init_kda(self, config) -> None:
        hidden_size = config.hidden_size
        device = torch.cuda.current_device()
        dtype = config.params_dtype
        self.num_heads = config.kimi_linear_num_heads
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = self.num_heads // self.tp_size
        self.head_dim = config.kimi_linear_head_dim
        self.projection_size = self.num_heads * self.head_dim
        self.local_projection_size = self.local_num_heads * self.head_dim

        self.q_proj = self._column_linear(hidden_size, self.projection_size)
        self.k_proj = self._column_linear(hidden_size, self.projection_size)
        self.v_proj = self._column_linear(hidden_size, self.projection_size)
        self.q_conv1d = KimiK3ShortConvolution(
            hidden_size=self.local_projection_size,
            kernel_size=config.kimi_linear_conv_kernel_size,
            activation="silu",
            device=device,
            dtype=dtype,
            tp_group=self.tp_group,
        )
        self.k_conv1d = KimiK3ShortConvolution(
            hidden_size=self.local_projection_size,
            kernel_size=config.kimi_linear_conv_kernel_size,
            activation="silu",
            device=device,
            dtype=dtype,
            tp_group=self.tp_group,
        )
        self.v_conv1d = KimiK3ShortConvolution(
            hidden_size=self.local_projection_size,
            kernel_size=config.kimi_linear_conv_kernel_size,
            activation="silu",
            device=device,
            dtype=dtype,
            tp_group=self.tp_group,
        )

        self.f_a_proj = self._duplicated_linear(hidden_size, self.head_dim)
        self.f_b_proj = self._column_linear(self.head_dim, self.projection_size)
        self.b_proj = self._column_linear(hidden_size, self.num_heads)
        self.g_proj = self._column_linear(hidden_size, self.projection_size)

        self.A_log = nn.Parameter(torch.empty(self.local_num_heads, dtype=torch.float32, device=device))
        self.dt_bias = nn.Parameter(torch.empty(self.local_projection_size, dtype=torch.float32, device=device))
        mark_param_dtype(self.A_log, torch.float32)
        mark_param_dtype(self.dt_bias, torch.float32)
        set_tensor_model_parallel_attributes(self.A_log, True, 0, 1)
        set_tensor_model_parallel_attributes(self.dt_bias, True, 0, 1)

        self.o_norm = FusedRMSNormGated(
            self.head_dim,
            eps=config.layernorm_epsilon,
            activation="sigmoid",
            device=device,
            dtype=dtype,
        )
        _mark_tp_replicated(self.o_norm, reduction="sum")
        self.o_proj = self._row_linear(self.projection_size, hidden_size)
        self.gate_lower_bound = config.kimi_kda_gate_lower_bound

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

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: dict | None = None,
    ) -> ShardedStateDict:
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        if not self.is_kda:
            return sharded_state_dict

        metadata = ensure_metadata_has_dp_cp_group(metadata)
        sharded_state_dict.update(
            make_sharded_tensors_for_checkpoint(
                {"A_log": self.A_log, "dt_bias": self.dt_bias},
                prefix,
                {"A_log": 0, "dt_bias": 0},
                sharded_offsets,
                tp_group=self.tp_group,
                dp_cp_group=metadata["dp_cp_group"],
            )
        )
        return sharded_state_dict

    def _forward_kda(
        self,
        hidden_states: torch.Tensor,
        packed_seq_params: PackedSeqParams | None,
    ) -> torch.Tensor:
        x = hidden_states.transpose(0, 1)
        cu_seqlens = packed_seq_params.cu_seqlens_q if packed_seq_params is not None else None

        q, _ = self.q_conv1d(
            x=_linear(self.q_proj, x),
            output_final_state=False,
            cu_seqlens=cu_seqlens,
        )
        k, _ = self.k_conv1d(
            x=_linear(self.k_proj, x),
            output_final_state=False,
            cu_seqlens=cu_seqlens,
        )
        v, _ = self.v_conv1d(
            x=_linear(self.v_proj, x),
            output_final_state=False,
            cu_seqlens=cu_seqlens,
        )

        q = rearrange(q, "b s (h d) -> b s h d", h=self.local_num_heads)
        k = rearrange(k, "b s (h d) -> b s h d", h=self.local_num_heads)
        v = rearrange(v, "b s (h d) -> b s h d", h=self.local_num_heads)
        forget_gate = rearrange(
            _linear(self.f_b_proj, _linear(self.f_a_proj, x)),
            "b s (h d) -> b s h d",
            h=self.local_num_heads,
        )
        beta = _linear(self.b_proj, x).float().sigmoid()

        output = sglang_kda(
            q=q,
            k=k,
            v=v,
            g=forget_gate,
            beta=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            lower_bound=self.gate_lower_bound,
            cu_seqlens=cu_seqlens,
        )

        gate = rearrange(
            _linear(self.g_proj, x),
            "b s (h d) -> b s h d",
            h=self.local_num_heads,
        )
        output = self.o_norm(output.reshape(-1, self.head_dim), gate.reshape(-1, self.head_dim))
        output = output.view(*gate.shape).flatten(-2)
        return _linear(self.o_proj, output).transpose(0, 1)

    def _causal_mla(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        if cu_seqlens is None:
            return F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                is_causal=True,
                scale=self.q_head_dim**-0.5,
            ).transpose(1, 2)

        assert query.shape[0] == 1, "Packed Kimi K3 MLA expects batch size 1"
        outputs = []
        boundaries = cu_seqlens.tolist()
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            q = query[:, start:end].transpose(1, 2)
            k = key[:, start:end].transpose(1, 2)
            v = value[:, start:end].transpose(1, 2)
            outputs.append(
                F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    is_causal=True,
                    scale=self.q_head_dim**-0.5,
                ).transpose(1, 2)
            )
        return torch.cat(outputs, dim=1)

    def _forward_mla(
        self,
        hidden_states: torch.Tensor,
        packed_seq_params: PackedSeqParams | None,
    ) -> torch.Tensor:
        x = hidden_states.transpose(0, 1)
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
        key_extra = copy_to_tensor_model_parallel_region(key_extra, group=self.tp_group)
        key_extra = key_extra.unsqueeze(-2).expand(*key_nope.shape[:-1], -1)
        key = torch.cat((key_nope, key_extra), dim=-1)

        cu_seqlens = packed_seq_params.cu_seqlens_q if packed_seq_params is not None else None
        output = self._causal_mla(query, key, value, cu_seqlens)
        output = output.flatten(-2) * torch.sigmoid(_linear(self.g_proj, x))
        return _linear(self.o_proj, output).transpose(0, 1)

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
        output = (
            self._forward_kda(hidden_states, packed_seq_params)
            if self.is_kda
            else self._forward_mla(hidden_states, packed_seq_params)
        )
        if self.sequence_parallel:
            output = scatter_to_sequence_parallel_region(output, group=self.tp_group)
        return output, None


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
        _mark_tp_replicated(self.self_attention_res_norm, reduction="sum")
        _mark_tp_replicated(self.self_attention_res_proj, reduction="sum")
        _mark_tp_replicated(self.mlp_res_norm, reduction="sum")
        _mark_tp_replicated(self.mlp_res_proj, reduction="sum")

        if self.layer_number == self.config.num_layers:
            self.output_attn_res_norm = KimiRMSNorm(hidden_size, eps, device=device, dtype=dtype)
            self.output_attn_res_proj = nn.Linear(hidden_size, 1, bias=False, device=device, dtype=dtype)
            _mark_tp_replicated(self.output_attn_res_norm, reduction="sum")
            _mark_tp_replicated(self.output_attn_res_proj, reduction="sum")

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
        prefix_sum = hidden_states

        if context is None:
            assert layer_idx == 0, "Attention-residual snapshot bank is missing"
            block_residual = hidden_states.new_empty(*hidden_states.shape[:-1], 0, hidden_states.shape[-1])
        else:
            block_residual = context

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

        return prefix_sum, block_residual
