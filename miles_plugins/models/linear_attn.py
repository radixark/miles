"""Head-sharded delta-rule attention (GDN, KDA) as Megatron modules.

Tensor parallelism shards heads. :class:`LinearAttentionLayer` is the ``self_attention`` drop-in: HF
input norm, one TP collective in (identity/all-reduce, or all-gather/reduce-scatter under sequence
parallelism), context-parallel zigzag relayout, and the row-parallel ``out_proj`` collective out.
:class:`DeltaRuleAttention` runs this rank's heads: the model's input projections, the short conv
over the group-major q/k/v (see ``megatron_to_hf.gdn_layout``), the fla kernel chosen by the
:class:`DeltaRule`, and a gated RMSNorm whose replicated weight has its gradient summed across TP.
Projections are plain bf16 linears on sharded parameters, so ``--fp8`` training leaves this layer in
bf16 as before; subclasses declare them under the HF names, one linear per contiguous output.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.modules.fused_norm_gate import rms_norm_gated
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group, make_sharded_tensors_for_checkpoint

from miles.backends.megatron_utils.fp32_param_utils import mark_param_dtype
from miles.kernels.attention.delta_rule import DeltaRule, DeltaRuleHeads, short_conv
from miles_plugins.models.cp_utils import build_fla_cp_context, packed_shard_to_zigzag, zigzag_to_packed_shard


WEIGHT_LAYOUT_VERSION = 1


class Projections(NamedTuple):
    """This rank's projections: ``qkv`` ``[b, s, Gl * group_qkv_dim]`` group-major, ``gate``
    ``[b, s, Hl * hv]``, ``beta_logits`` ``[b, s, Hl]``, ``decay`` ``[b, s, Hl]`` (GDN) or
    ``[b, s, Hl * hv]`` (KDA)."""

    qkv: torch.Tensor
    gate: torch.Tensor
    beta_logits: torch.Tensor
    decay: torch.Tensor


class _ShardedShortConvolution(ShortConvolution):
    """The short conv over this rank's channels, as a module holding the TP-sharded weight."""

    def __init__(self, *args, tp_group, **kwargs):
        super().__init__(*args, **kwargs)
        self.tp_group = tp_group
        set_tensor_model_parallel_attributes(self.weight, True, 0, 1)

    def forward(self, x: torch.Tensor, cu_seqlens=None, cp_context=None) -> torch.Tensor:
        return short_conv(x, self.weight[:, 0], self.activation, self.backend, cu_seqlens, cp_context)

    def sharded_state_dict(self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None):
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        return make_sharded_tensors_for_checkpoint(
            self.state_dict(prefix="", keep_vars=True),
            prefix,
            {"weight": 0},
            sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )


class DeltaRuleAttention(MegatronModule, ABC):
    """This rank's heads: projections -> conv -> rule -> gated norm. ``out_proj`` holds this rank's
    columns; :class:`LinearAttentionLayer` applies it and the TP reduction. Subclasses create their
    projections in ``_build_projections`` with :meth:`sharded_linear` and map them in ``project``."""

    def __init__(
        self, config, heads: DeltaRuleHeads, rule: DeltaRule, conv_kernel_size: int, norm_eps: float, tp_group
    ):
        super().__init__(config=config)
        self.tp_group = tp_group
        self.heads = heads
        self.local = heads.local(tp_group.size())
        self.rule = rule
        self.conv_kernel_size = conv_kernel_size
        self.norm_eps = norm_eps
        device = torch.cuda.current_device()
        dtype = config.params_dtype

        self._sharded_params: dict[str, int] = {}
        self.register_buffer(
            "weight_layout_version", torch.tensor([WEIGHT_LAYOUT_VERSION], dtype=torch.int32, device=device)
        )
        self._build_projections()
        with get_cuda_rng_tracker().fork():
            self.conv1d = _ShardedShortConvolution(
                hidden_size=self.local.num_k_heads * self.local.group_qkv_dim,
                kernel_size=conv_kernel_size,
                bias=False,
                activation="silu",
                device=device,
                dtype=dtype,
                tp_group=tp_group,
            )
            self.A_log = nn.Parameter(
                torch.empty(self.local.num_v_heads, dtype=torch.float32, device=device).uniform_(1, 16).log_()
            )
        mark_param_dtype(self.A_log, torch.float32)
        self.dt_bias = nn.Parameter(
            torch.ones(rule.dt_bias_size(self.local), dtype=rule.dt_bias_dtype or dtype, device=device)
        )
        if rule.dt_bias_dtype is not None:
            mark_param_dtype(self.dt_bias, rule.dt_bias_dtype)
        self._mark_sharded("A_log", self.A_log, dim=0)
        self._mark_sharded("dt_bias", self.dt_bias, dim=0)
        self.norm = FusedRMSNormGated(
            heads.head_v_dim, eps=norm_eps, activation=rule.norm_activation, device=device, dtype=dtype
        )
        self.out_proj = nn.Linear(self.local.value_dim, config.hidden_size, bias=False, device=device, dtype=dtype)
        with get_cuda_rng_tracker().fork():
            config.output_layer_init_method(self.out_proj.weight)
        self._mark_sharded("out_proj.weight", self.out_proj.weight, dim=1)

    def _mark_sharded(self, name: str, param: nn.Parameter, dim: int) -> None:
        set_tensor_model_parallel_attributes(param, True, dim, 1)
        self._sharded_params[name] = dim

    def sharded_linear(self, name: str, input_size: int, local_output_size: int) -> nn.Linear:
        """This rank's row shard of a head-sharded projection; the input already went through the TP
        collective, so the linear itself communicates nothing."""
        linear = nn.Linear(
            input_size,
            local_output_size,
            bias=False,
            device=torch.cuda.current_device(),
            dtype=self.config.params_dtype,
        )
        with get_cuda_rng_tracker().fork():
            self.config.init_method(linear.weight)
        self._mark_sharded(f"{name}.weight", linear.weight, dim=0)
        return linear

    def sharded_state_dict(self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None):
        sharded = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        params = dict(self.named_parameters())
        sharded.update(
            make_sharded_tensors_for_checkpoint(
                {name: params[name] for name in self._sharded_params},
                prefix,
                self._sharded_params,
                sharded_offsets,
                tp_group=self.tp_group,
                dp_cp_group=metadata["dp_cp_group"],
            )
        )
        return sharded

    @abstractmethod
    def _build_projections(self) -> None: ...

    @abstractmethod
    def project(self, x: torch.Tensor) -> Projections: ...

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor | None, cp_context=None) -> torch.Tensor:
        """x ``[b, s, hidden]``, TP collective already applied -> ``[b, s, local value_dim]``."""
        batch, seq_len, _ = x.shape
        local = self.local
        qkv, gate, beta_logits, decay = self.project(x)
        mixed = self.conv1d(qkv, cu_seqlens=cu_seqlens, cp_context=cp_context)
        q, k, v = mixed.view(batch, seq_len, local.num_k_heads, -1).split(
            [local.head_k_dim, local.head_k_dim, local.group_value_dim], dim=-1
        )
        v = v.reshape(batch, seq_len, local.num_v_heads, local.head_v_dim)
        core = self.rule(
            q, k, v, beta_logits, decay, self.A_log, self.dt_bias, cu_seqlens=cu_seqlens, cp_context=cp_context
        )
        weight = copy_to_tensor_model_parallel_region(self.norm.weight, group=self.tp_group)
        core = rms_norm_gated(
            core.reshape(-1, self.heads.head_v_dim),
            gate.reshape(-1, self.heads.head_v_dim),
            weight,
            self.norm.bias,
            self.rule.norm_activation,
            eps=self.norm_eps,
        )
        return core.reshape(batch, seq_len, -1)


class KimiDeltaAttention(DeltaRuleAttention):
    """KDA in the Kimi-K3 / GLM-5.3-flash layout: ``q_proj`` / ``k_proj`` / ``v_proj`` fused into one
    group-major ``in_proj_qkv``, ``g_proj`` (output gate), ``b_proj`` (beta), and the low-rank forget gate
    ``f_b_proj(f_a_proj(x))``. ``f_a_proj`` is replicated and feeds head-sharded ``f_b_proj``, so its weight
    passes the TP copy op like the norm."""

    def _build_projections(self):
        hidden, local = self.config.hidden_size, self.local
        self.in_proj_qkv = self.sharded_linear("in_proj_qkv", hidden, local.num_k_heads * local.group_qkv_dim)
        self.g_proj = self.sharded_linear("g_proj", hidden, local.value_dim)
        self.b_proj = self.sharded_linear("b_proj", hidden, local.num_v_heads)
        self.f_a_proj = nn.Linear(
            hidden,
            self.heads.head_v_dim,
            bias=False,
            device=torch.cuda.current_device(),
            dtype=self.config.params_dtype,
        )
        self.config.init_method(self.f_a_proj.weight)
        self.f_b_proj = self.sharded_linear("f_b_proj", self.heads.head_v_dim, local.value_dim)

    def project(self, x):
        f_a_weight = copy_to_tensor_model_parallel_region(self.f_a_proj.weight, group=self.tp_group)
        return Projections(self.in_proj_qkv(x), self.g_proj(x), self.b_proj(x), self.f_b_proj(F.linear(x, f_a_weight)))


class LinearAttentionLayer(MegatronModule):
    """``self_attention`` drop-in. ``allgather_cp``: the data pipeline already hands each CP rank a
    contiguous shard (``--allgather-cp``), so no zigzag relayout."""

    def __init__(
        self,
        config,
        linear_attn: DeltaRuleAttention,
        input_layernorm: nn.Module,
        pg_collection: ProcessGroupCollection,
        allgather_cp: bool,
    ):
        super().__init__(config=config)
        self.tp_group = pg_collection.tp
        self.cp_group = pg_collection.cp
        self.cp_size = self.cp_group.size()
        self.sequence_parallel = config.sequence_parallel
        self.allgather_cp = allgather_cp
        self.input_layernorm = input_layernorm
        self.linear_attn = linear_attn
        for param in self.input_layernorm.parameters():
            param.sequence_parallel = self.sequence_parallel

    def _global_cu_seqlens(self, hidden_states, packed_seq_params):
        if packed_seq_params is not None and packed_seq_params.cu_seqlens_q is not None:
            return packed_seq_params.cu_seqlens_q
        total = hidden_states.shape[0] * self.cp_size
        return torch.tensor([0, total], dtype=torch.int32, device=hidden_states.device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset=None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        x = self.input_layernorm(hidden_states)
        if self.sequence_parallel:
            x = gather_from_sequence_parallel_region(x, tensor_parallel_output_grad=True, group=self.tp_group)
        else:
            x = copy_to_tensor_model_parallel_region(x, group=self.tp_group)

        global_cu_seqlens = self._global_cu_seqlens(x, packed_seq_params)
        relayout = self.cp_size > 1 and not self.allgather_cp
        if relayout:
            x = zigzag_to_packed_shard(x, global_cu_seqlens, self.cp_group, self.cp_group.rank(), self.cp_size)
        cp_context = None
        cu_seqlens = global_cu_seqlens
        if self.cp_size > 1:
            cp_context = build_fla_cp_context(
                global_cu_seqlens, self.cp_group, self.linear_attn.conv_kernel_size, x.device
            )
            cu_seqlens = cp_context.cu_seqlens

        core = self.linear_attn(x.transpose(0, 1), cu_seqlens, cp_context).transpose(0, 1)

        if relayout:
            core = packed_shard_to_zigzag(core, global_cu_seqlens, self.cp_group, self.cp_group.rank(), self.cp_size)
        output = self.linear_attn.out_proj(core)
        if self.sequence_parallel:
            output = reduce_scatter_to_sequence_parallel_region(output, group=self.tp_group)
        else:
            output = reduce_from_tensor_model_parallel_region(output, group=self.tp_group)
        return output, None
