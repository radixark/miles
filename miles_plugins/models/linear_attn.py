"""Head-sharded linear attention (KDA, GDN) as Megatron modules.

:class:`LinearAttentionLayer` is the ``self_attention`` drop-in: input norm, one TP collective in
(identity / all-reduce, or all-gather / reduce-scatter under sequence parallelism), the context-parallel
zigzag relayout, and the row-parallel ``out_proj`` collective out. :class:`LinearAttention` runs this
rank's heads: the model's input projections, the short conv over the group-major q/k/v (see
``megatron_to_hf.linear_attn_layout``), the family's recurrence through fla, and a gated RMSNorm whose
replicated weight has its gradient summed across TP. A family subclass supplies the recurrence; a model
subclass declares the projections under the HF names, plain bf16 linears on sharded parameters, so
``--fp8`` training leaves this layer in bf16.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from miles.backends.megatron_utils.megatron_to_hf.linear_attn_layout import LinearAttnHeads
from miles_plugins.models.cp_utils import build_fla_cp_context, packed_shard_to_zigzag, zigzag_to_packed_shard

try:
    from fla.modules import FusedRMSNormGated
    from fla.modules.fused_norm_gate import rms_norm_gated
except ImportError:
    pass

INT32_ELEMENTS = 2**31 - 1
_CHUNK_ELEMENTS = 2**30
_CHANNEL_ALIGN = 128


def gdn_kernel(backend: str):
    if backend == "fla":
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        except ImportError as exc:
            raise ImportError("GDN backend 'fla' requires flash-linear-attention.") from exc
        return chunk_gated_delta_rule

    if backend == "flashqla":
        try:
            from flash_qla import chunk_gated_delta_rule
        except ImportError as exc:
            raise ImportError(
                "GDN backend 'flashqla' requires FlashQLA. Install it from https://github.com/QwenLM/FlashQLA."
            ) from exc
        return chunk_gated_delta_rule

    raise ValueError(f"Unsupported GDN backend: {backend}")


def kda_kernel():
    try:
        from fla.ops.kda import chunk_kda
    except ImportError as exc:
        raise ImportError("KDA requires flash-linear-attention >= 0.5 (fla.ops.kda).") from exc
    return chunk_kda


def kda_recurrence(q, k, v, beta_logits, decay, A_log, dt_bias, *, gate_lower_bound, cu_seqlens, cp_context):
    """q/k ``[b, s, H, hk]``, v ``[b, s, H, hv]``, decay ``[b, s, H * hv]`` (the low-rank forget gate,
    gated inside the kernel) -> ``[b, s, H, hv]``."""
    boundaries = {"cp_context": cp_context} if cp_context is not None else {"cu_seqlens": cu_seqlens}
    out, _ = kda_kernel()(
        q=q,
        k=k,
        v=v,
        g=decay.reshape(v.shape),
        beta=beta_logits.float().sigmoid(),
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=gate_lower_bound,
        transpose_state_layout=True,
        **boundaries,
    )
    return out


class _ContiguousGrad(torch.autograd.Function):
    """Identity whose backward hands on a contiguous gradient. The gradient of one channel chunk of a
    concatenation is a strided view whose row stride is the full width; fla's conv backward indexes it
    with int32 and overflows past 2**31 elements."""

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad.contiguous()


def causal_short_conv(x, weight, activation, backend, cu_seqlens=None, cp_context=None):
    """Depthwise causal conv of ``x`` ``[b, s, C]`` with ``weight`` ``[C, K]`` through fla. fla indexes with
    int32, so inputs past 2**31 elements (long sequences at low TP) run as channel chunks of at most
    2**30; the conv is per channel, so this is exact."""
    from fla.modules.conv.causal_conv1d import causal_conv1d

    tokens = x.shape[0] * x.shape[1]
    kwargs = {
        "bias": None,
        "activation": activation,
        "backend": backend,
        "cu_seqlens": cu_seqlens,
        "cp_context": cp_context,
    }
    if tokens * x.shape[-1] <= INT32_ELEMENTS:
        return causal_conv1d(x=x, weight=weight, **kwargs)[0]
    width = max(_CHANNEL_ALIGN, _CHUNK_ELEMENTS // tokens // _CHANNEL_ALIGN * _CHANNEL_ALIGN)
    chunks = [
        _ContiguousGrad.apply(causal_conv1d(x=chunk.contiguous(), weight=chunk_weight, **kwargs)[0])
        for chunk, chunk_weight in zip(x.split(width, dim=-1), weight.split(width), strict=True)
    ]
    return torch.cat(chunks, dim=-1)


class ShardedShortConv(nn.Conv1d):
    """Depthwise causal conv (SiLU, no bias) over this rank's channels, holding the TP-sharded weight.
    Built like fla's ``ShortConvolution``, so weights and checkpoints match it."""

    def __init__(self, channels: int, kernel_size: int, tp_group, device=None, dtype=None):
        super().__init__(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,
            bias=False,
            padding=kernel_size - 1,
            device=device,
            dtype=dtype,
        )
        self.activation = "silu"
        self.backend = "triton"
        self.tp_group = tp_group
        set_tensor_model_parallel_attributes(self.weight, True, 0, 1)

    def forward(self, x: torch.Tensor, cu_seqlens=None, cp_context=None) -> torch.Tensor:
        return causal_short_conv(x, self.weight[:, 0], self.activation, self.backend, cu_seqlens, cp_context)

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


class Projections(NamedTuple):
    """This rank's projections: ``qkv`` ``[b, s, Gl * group_qkv_dim]`` group-major, ``gate``
    ``[b, s, Hl * hv]``, ``beta_logits`` ``[b, s, Hl]``, ``decay`` ``[b, s, Hl]`` (GDN) or
    ``[b, s, Hl * hv]`` (KDA)."""

    qkv: torch.Tensor
    gate: torch.Tensor
    beta_logits: torch.Tensor
    decay: torch.Tensor


class LinearAttention(MegatronModule, ABC):
    """This rank's heads: projections -> conv -> recurrence -> gated norm. ``out_proj`` holds this
    rank's columns; :class:`LinearAttentionLayer` applies it and the TP reduction. ``dt_bias`` is one
    value per value head, or per value channel when the family sets ``dt_bias_per_channel``."""

    dt_bias_per_channel: bool = False
    dt_bias_dtype: torch.dtype | None = None

    def __init__(
        self,
        config,
        heads: LinearAttnHeads,
        conv_kernel_size: int,
        norm_eps: float,
        tp_group,
        norm_activation: str,
    ):
        super().__init__(config=config)
        self.tp_group = tp_group
        self.heads = heads
        self.local = heads.local(tp_group.size())
        self.conv_kernel_size = conv_kernel_size
        self.norm_eps = norm_eps
        self.norm_activation = norm_activation
        device = torch.cuda.current_device()
        dtype = config.params_dtype

        self._sharded_params: dict[str, int] = {}
        self._build_projections()
        with get_cuda_rng_tracker().fork():
            self.conv1d = ShardedShortConv(
                self.local.num_k_heads * self.local.group_qkv_dim, conv_kernel_size, tp_group, device, dtype
            )
            self.A_log = nn.Parameter(
                torch.empty(self.local.num_v_heads, dtype=torch.float32, device=device).uniform_(1, 16).log_()
            )
        mark_param_dtype(self.A_log, torch.float32)
        dt_bias_size = self.local.value_dim if self.dt_bias_per_channel else self.local.num_v_heads
        self.dt_bias = nn.Parameter(torch.ones(dt_bias_size, dtype=self.dt_bias_dtype or dtype, device=device))
        if self.dt_bias_dtype is not None:
            mark_param_dtype(self.dt_bias, self.dt_bias_dtype)
        self._mark_sharded("A_log", self.A_log, dim=0)
        self._mark_sharded("dt_bias", self.dt_bias, dim=0)
        self.norm = FusedRMSNormGated(
            heads.head_v_dim, eps=norm_eps, activation=norm_activation, device=device, dtype=dtype
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

    @abstractmethod
    def recurrence(self, q, k, v, beta_logits, decay, cu_seqlens, cp_context) -> torch.Tensor:
        """q/k ``[b, s, Gl, hk]``, v ``[b, s, Hl, hv]`` -> ``[b, s, Hl, hv]``."""

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
        core = self.recurrence(q, k, v, beta_logits, decay, cu_seqlens, cp_context)
        weight = copy_to_tensor_model_parallel_region(self.norm.weight, group=self.tp_group)
        core = rms_norm_gated(
            core.reshape(-1, self.heads.head_v_dim),
            gate.reshape(-1, self.heads.head_v_dim),
            weight,
            self.norm.bias,
            self.norm_activation,
            eps=self.norm_eps,
        )
        return core.reshape(batch, seq_len, -1)


class KimiDeltaAttention(LinearAttention):
    """KDA with ``q_proj`` / ``k_proj`` / ``v_proj`` fused into one group-major ``in_proj_qkv``,
    ``g_proj`` (output gate), ``b_proj`` (beta), and the low-rank forget gate ``f_b_proj(f_a_proj(x))``,
    gated inside fla's kernel. ``f_a_proj`` is replicated and feeds head-sharded ``f_b_proj``, so its
    weight passes the TP copy op like the norm."""

    dt_bias_per_channel = True
    dt_bias_dtype = torch.float32

    def __init__(self, config, heads, conv_kernel_size, norm_eps, tp_group, gate_lower_bound: float):
        super().__init__(config, heads, conv_kernel_size, norm_eps, tp_group, norm_activation="sigmoid")
        self.gate_lower_bound = gate_lower_bound

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

    def recurrence(self, q, k, v, beta_logits, decay, cu_seqlens, cp_context):
        return kda_recurrence(
            q,
            k,
            v,
            beta_logits,
            decay,
            self.A_log,
            self.dt_bias,
            gate_lower_bound=self.gate_lower_bound,
            cu_seqlens=cu_seqlens,
            cp_context=cp_context,
        )


class LinearAttentionLayer(MegatronModule):
    """``self_attention`` drop-in. ``allgather_cp``: the data pipeline already hands each CP rank a
    contiguous shard (``--allgather-cp``), so no zigzag relayout."""

    def __init__(
        self,
        config,
        linear_attn: LinearAttention,
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
