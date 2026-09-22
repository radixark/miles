"""Unified head-sharded gated delta-rule (GDN) attention core for the Qwen GDN wrappers.

One module serves Qwen3.5/3.6/3.8 (``in_proj_qkv`` / ``in_proj_z`` / ``in_proj_b`` / ``in_proj_a``)
and Qwen3-Next (``in_proj_qkvz`` / ``in_proj_ba``) and owns parallelism the way the Kimi-K3 KDA
layer does:

* **TP** through Megatron column-parallel projections whose output columns are whole key-head
  groups (a key head with its ``num_v_heads // num_k_heads`` value heads), head-sharded ``conv1d`` /
  ``A_log`` / ``dt_bias``, a replicated per-head ``FusedRMSNormGated`` whose gradient is summed
  across TP, and a row-parallel output projection.  The GDN operator runs once per rank on the
  local heads; there is no replicated GDN compute across TP ranks.
* **SP** is handled by the caller (``HuggingfaceAttention.forward`` gathers the sequence before
  ``hf_forward`` and scatters afterwards); the projections therefore run with
  ``sequence_parallel=False``.
* **CP** through ``build_gdn_cp_context`` (fla state passing); the zigzag <-> packed relayout also
  stays at the ``HuggingfaceAttention`` boundary.

Parameter layout
----------------

The module keeps the HF parameter names, so the Megatron <-> HF mappings stay name-for-name.  Two
tensors are stored in a *head-interleaved* row order instead of HF's flat ``[q_all, k_all, v_all]``
order so that contiguous TP chunks are head shards for every TP size (the same trick Megatron uses
for ``linear_qkv``): Qwen3.5's ``in_proj_qkv.weight`` and both models' ``conv1d.weight``/``bias``.
Their rows are grouped per key head as ``[q_h (head_k_dim), k_h (head_k_dim), v_{h,0..g-1}
(g * head_v_dim)]``.  Qwen3-Next's ``in_proj_qkvz`` / ``in_proj_ba`` already have that head-major
order in HF.  :func:`hf_to_megatron_linear_attn` / :func:`megatron_to_hf_linear_attn` translate
the two tensors; the converters (``megatron_to_hf/qwen3_5.py``, ``qwen3_next.py``) and the
mbridge plugins call them, and the TP split/merge is the ordinary contiguous chunk along the
partition dimension.  Without a Megatron model-parallel config (unit tests) the projections are
plain ``nn.Linear`` and the layout is the TP=1 case of the same scheme.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
except ImportError:  # pragma: no cover - fla is a hard dependency of the GDN wrappers
    FusedRMSNormGated = ShortConvolution = None

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.tensor_parallel.mappings import copy_to_tensor_model_parallel_region
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group, make_sharded_tensors_for_checkpoint

from miles.backends.megatron_utils.fp32_param_utils import mark_param_dtype

from .qwen_gdn_backend import get_chunk_gated_delta_rule

__all__ = [
    "GatedDeltaRuleAttentionCore",
    "GdnLayout",
    "HEAD_INTERLEAVED_PARAMS",
    "hf_to_megatron_linear_attn",
    "megatron_to_hf_linear_attn",
    "merge_linear_attn_param",
    "shard_linear_attn_param",
]

# ``linear_attn`` parameters (HF names) stored head-interleaved in the Megatron model, per HF layout.
HEAD_INTERLEAVED_PARAMS = {
    "qwen3_5": ("in_proj_qkv.weight", "conv1d.weight", "conv1d.bias"),
    "qwen3_next": ("conv1d.weight", "conv1d.bias"),
}
# Partition dimension of every ``linear_attn`` parameter under TP (``None``: replicated).
LINEAR_ATTN_PARTITION_DIM = {
    "in_proj_qkv.weight": 0,
    "in_proj_z.weight": 0,
    "in_proj_b.weight": 0,
    "in_proj_a.weight": 0,
    "in_proj_qkvz.weight": 0,
    "in_proj_ba.weight": 0,
    "conv1d.weight": 0,
    "conv1d.bias": 0,
    "A_log": 0,
    "dt_bias": 0,
    "norm.weight": None,
    "out_proj.weight": 1,
}


@dataclass(frozen=True)
class GdnLayout:
    """Static shape facts of one GDN layer (HF config fields)."""

    hidden_size: int
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel_size: int
    rms_norm_eps: float
    activation: str = "silu"
    hf_layout: str = "qwen3_next"  # or "qwen3_5"

    def __post_init__(self):
        if self.hf_layout not in HEAD_INTERLEAVED_PARAMS:
            raise ValueError(f"unknown hf_layout {self.hf_layout!r}")
        if self.num_v_heads % self.num_k_heads:
            raise ValueError("linear_num_value_heads must be a multiple of linear_num_key_heads")

    @property
    def group(self) -> int:
        return self.num_v_heads // self.num_k_heads

    @property
    def key_dim(self) -> int:
        return self.num_k_heads * self.head_k_dim

    @property
    def value_dim(self) -> int:
        return self.num_v_heads * self.head_v_dim

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim

    @property
    def rows_per_k_head(self) -> int:
        """Rows of one key-head group in the head-interleaved ``[q, k, v]`` order."""
        return 2 * self.head_k_dim + self.group * self.head_v_dim

    @classmethod
    def from_hf_config(cls, text_config, *, hf_layout: str) -> GdnLayout:
        return cls(
            hidden_size=int(text_config.hidden_size),
            num_k_heads=int(text_config.linear_num_key_heads),
            num_v_heads=int(text_config.linear_num_value_heads),
            head_k_dim=int(text_config.linear_key_head_dim),
            head_v_dim=int(text_config.linear_value_head_dim),
            conv_kernel_size=int(text_config.linear_conv_kernel_dim),
            rms_norm_eps=float(text_config.rms_norm_eps),
            activation=str(text_config.hidden_act),
            hf_layout=hf_layout,
        )


# --------------------------------------------------------------------------- #
# HF flat <-> head-interleaved row order
# --------------------------------------------------------------------------- #
def interleave_qkv_rows(layout: GdnLayout, flat: torch.Tensor) -> torch.Tensor:
    """``[q_all, k_all, v_all]`` rows -> per key head ``[q_h, k_h, v_{h,*}]`` rows (any trailing dims)."""
    if flat.shape[0] != layout.conv_dim:
        raise ValueError(f"expected {layout.conv_dim} rows, got {flat.shape[0]}")
    trailing = flat.shape[1:]
    q, k, v = torch.split(flat, [layout.key_dim, layout.key_dim, layout.value_dim], dim=0)
    q = q.reshape(layout.num_k_heads, layout.head_k_dim, *trailing)
    k = k.reshape(layout.num_k_heads, layout.head_k_dim, *trailing)
    v = v.reshape(layout.num_k_heads, layout.group * layout.head_v_dim, *trailing)
    return torch.cat([q, k, v], dim=1).reshape(layout.conv_dim, *trailing).contiguous()


def deinterleave_qkv_rows(layout: GdnLayout, interleaved: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`interleave_qkv_rows`."""
    if interleaved.shape[0] != layout.conv_dim:
        raise ValueError(f"expected {layout.conv_dim} rows, got {interleaved.shape[0]}")
    trailing = interleaved.shape[1:]
    grouped = interleaved.reshape(layout.num_k_heads, layout.rows_per_k_head, *trailing)
    q, k, v = torch.split(grouped, [layout.head_k_dim, layout.head_k_dim, layout.group * layout.head_v_dim], dim=1)
    return torch.cat(
        [
            q.reshape(layout.key_dim, *trailing),
            k.reshape(layout.key_dim, *trailing),
            v.reshape(layout.value_dim, *trailing),
        ],
        dim=0,
    ).contiguous()


def hf_to_megatron_linear_attn(layout: GdnLayout, name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Full (unsharded) HF ``linear_attn.<name>`` tensor -> the Megatron model's row order."""
    if name in HEAD_INTERLEAVED_PARAMS[layout.hf_layout]:
        return interleave_qkv_rows(layout, tensor)
    return tensor


def megatron_to_hf_linear_attn(layout: GdnLayout, name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Full (TP-merged) Megatron ``linear_attn.<name>`` tensor -> HF row order."""
    if name in HEAD_INTERLEAVED_PARAMS[layout.hf_layout]:
        return deinterleave_qkv_rows(layout, tensor)
    return tensor


def shard_linear_attn_param(name: str, full_megatron: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    """This rank's shard of a Megatron-ordered ``linear_attn.<name>`` tensor (contiguous chunk)."""
    dim = LINEAR_ATTN_PARTITION_DIM[name]
    if dim is None or tp_size == 1:
        return full_megatron
    return full_megatron.chunk(tp_size, dim=dim)[tp_rank].contiguous()


def merge_linear_attn_param(name: str, shards: list[torch.Tensor]) -> torch.Tensor:
    """Inverse of :func:`shard_linear_attn_param`."""
    dim = LINEAR_ATTN_PARTITION_DIM[name]
    if dim is None or len(shards) == 1:
        return shards[0]
    return torch.cat(shards, dim=dim)


def hf_linear_attn_to_local(
    layout: GdnLayout, hf: dict[str, torch.Tensor], *, tp_rank: int, tp_size: int
) -> dict[str, torch.Tensor]:
    """Slice one HF ``linear_attn`` state dict into this rank's parameters (Megatron order)."""
    if layout.num_k_heads % tp_size:
        raise ValueError("linear_num_key_heads must be divisible by the TP size")
    return {
        name: shard_linear_attn_param(name, hf_to_megatron_linear_attn(layout, name, value), tp_rank, tp_size)
        for name, value in hf.items()
    }


def local_to_hf_linear_attn(layout: GdnLayout, shards: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Assemble the HF ``linear_attn`` state dict from all TP ranks' local parameters."""
    return {
        name: megatron_to_hf_linear_attn(
            layout, name, merge_linear_attn_param(name, [shard[name] for shard in shards])
        )
        for name in shards[0]
    }


# --------------------------------------------------------------------------- #
# TP-aware submodules
# --------------------------------------------------------------------------- #
class _ShardedShortConvolution(ShortConvolution if ShortConvolution is not None else nn.Module):
    """``ShortConvolution`` over this rank's head-interleaved ``[q, k, v]`` channels."""

    def __init__(self, *args, tp_group, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tp_group = tp_group
        set_tensor_model_parallel_attributes(self.weight, True, 0, 1)
        if self.bias is not None:
            set_tensor_model_parallel_attributes(self.bias, True, 0, 1)

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None
    ) -> ShardedStateDict:
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        return make_sharded_tensors_for_checkpoint(
            self.state_dict(prefix="", keep_vars=True),
            prefix,
            {"weight": 0, "bias": 0},
            sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )


class _TPReplicatedRMSNormGated(FusedRMSNormGated if FusedRMSNormGated is not None else nn.Module):
    """Per-head gated RMSNorm whose replicated weight sees only this rank's heads.

    The weight is identical on every TP rank; its gradient is the sum over all heads, so the local
    partial gradients are all-reduced across TP (identity forward, all-reduce backward) before
    autograd accumulates them.  This does not depend on ``sequence_parallel`` or on Megatron's
    layernorm-gradient pass.
    """

    def __init__(self, *args, tp_group, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tp_group = tp_group

    def forward(
        self, x: torch.Tensor, g: torch.Tensor, residual=None, prenorm: bool = False, residual_in_fp32: bool = False
    ):
        from fla.modules.fused_norm_gate import rms_norm_gated

        weight = self.weight
        if self.tp_group is not None and self.tp_group.size() > 1 and weight is not None:
            weight = copy_to_tensor_model_parallel_region(weight, group=self.tp_group)
        return rms_norm_gated(
            x,
            g,
            weight,
            self.bias,
            self.activation,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )


class _Linear(nn.Linear):
    """``nn.Linear`` with the ``(output, bias)`` return of Megatron's parallel linears."""

    def forward(self, x):  # type: ignore[override]
        return super().forward(x), None


# --------------------------------------------------------------------------- #
# The head-sharded module
# --------------------------------------------------------------------------- #
class GatedDeltaRuleAttentionCore(MegatronModule):
    """Head-sharded GDN layer: projections, short conv, GDN operator, gated norm, output projection.

    ``hidden_states`` is ``[batch, seq, hidden]`` after SP gather and CP relayout; ``cu_seqlens`` are
    this rank's packed boundaries (global ones when no CP).  ``build_gdn_cp_context`` reads
    ``cp_group`` / ``cp_rank`` / ``cp_world_size`` / ``conv_kernel_size`` off the module.

    ``mp_config`` is the Megatron ``TransformerConfig`` of the enclosing layer; with it the
    projections are Megatron column/row-parallel linears over ``tp_group``.  Without it (unit tests
    outside a Megatron process group) they are plain ``nn.Linear`` and ``tp_size == 1``.
    """

    def __init__(
        self,
        layout: GdnLayout,
        *,
        layer_idx: int = 0,
        gdn_backend: str = "fla",
        mp_config=None,
        tp_group=None,
        params_dtype: torch.dtype | None = None,
        a_log_fp32: bool = False,
        device=None,
    ) -> None:
        super().__init__(config=mp_config)
        if ShortConvolution is None:
            raise ImportError("the GDN wrappers require flash-linear-attention (fla.modules)")
        self.layout = layout
        self.layer_idx = layer_idx
        self.gdn_backend = gdn_backend
        self.chunk_gated_delta_rule = get_chunk_gated_delta_rule(gdn_backend)
        self.tp_group = tp_group if mp_config is not None else None
        self.tp_size = self.tp_group.size() if self.tp_group is not None else 1
        if layout.num_k_heads % self.tp_size:
            raise ValueError("linear_num_key_heads must be divisible by the TP size")
        self.tp_sharded = self.tp_size > 1
        # HF-compatible shape facts (build_gdn_cp_context and the FSDP-side helpers read some of these)
        self.hidden_size = layout.hidden_size
        self.num_v_heads = layout.num_v_heads
        self.num_k_heads = layout.num_k_heads
        self.head_k_dim = layout.head_k_dim
        self.head_v_dim = layout.head_v_dim
        self.key_dim = layout.key_dim
        self.value_dim = layout.value_dim
        self.conv_kernel_size = layout.conv_kernel_size
        self.activation = layout.activation
        self.layer_norm_epsilon = layout.rms_norm_eps
        self.local_num_k_heads = layout.num_k_heads // self.tp_size
        self.local_num_v_heads = layout.num_v_heads // self.tp_size
        self.local_key_dim = self.local_num_k_heads * layout.head_k_dim
        self.local_value_dim = self.local_num_v_heads * layout.head_v_dim
        self.local_conv_dim = 2 * self.local_key_dim + self.local_value_dim
        # build_gdn_cp_context reads these off the module
        self.cp_group = None
        self.cp_rank = 0
        self.cp_world_size = 1

        device = device if device is not None else torch.cuda.current_device()
        params_dtype = params_dtype if params_dtype is not None else torch.get_default_dtype()
        self._mp_config = mp_config
        if mp_config is not None:
            # The caller has already gathered the sequence (HuggingfaceAttention.forward); the
            # projections must not gather/reduce-scatter again.
            self._linear_config = copy.copy(mp_config)
            self._linear_config.sequence_parallel = False

        if layout.hf_layout == "qwen3_next":
            self.in_proj_qkvz = self._column(
                layout.num_k_heads * (2 * layout.head_k_dim + 2 * layout.group * layout.head_v_dim)
            )
            self.in_proj_ba = self._column(layout.num_k_heads * 2 * layout.group)
        else:
            self.in_proj_qkv = self._column(layout.conv_dim)
            self.in_proj_z = self._column(layout.value_dim)
            self.in_proj_b = self._column(layout.num_v_heads)
            self.in_proj_a = self._column(layout.num_v_heads)

        self.conv1d = _ShardedShortConvolution(
            hidden_size=self.local_conv_dim,
            kernel_size=layout.conv_kernel_size,
            bias=False,
            activation=layout.activation,
            tp_group=self.tp_group,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.local_num_v_heads))
        A = torch.empty(self.local_num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A).to(torch.float32))
        if a_log_fp32:
            # Qwen3.5 ships A_log in fp32; keep it fp32 through Megatron's mixed-precision wrapper.
            mark_param_dtype(self.A_log, torch.float32)
        set_tensor_model_parallel_attributes(self.dt_bias, True, 0, 1)
        set_tensor_model_parallel_attributes(self.A_log, True, 0, 1)

        self.norm = _TPReplicatedRMSNormGated(
            layout.head_v_dim,
            eps=layout.rms_norm_eps,
            activation=layout.activation,
            device=device,
            dtype=params_dtype,
            tp_group=self.tp_group,
        )
        self.out_proj = self._row(layout.value_dim, layout.hidden_size)

    # -- projections -----------------------------------------------------------------------------
    def _column(self, output_size: int):
        if self._mp_config is None:
            return _Linear(self.layout.hidden_size, output_size, bias=False)
        from megatron.core.extensions.transformer_engine import TEColumnParallelLinear

        return TEColumnParallelLinear(
            self.layout.hidden_size,
            output_size,
            config=self._linear_config,
            init_method=self._mp_config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.tp_group,
        )

    def _row(self, input_size: int, output_size: int):
        if self._mp_config is None:
            return _Linear(input_size, output_size, bias=False)
        from megatron.core.extensions.transformer_engine import TERowParallelLinear

        return TERowParallelLinear(
            input_size,
            output_size,
            config=self._linear_config,
            init_method=self._mp_config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.tp_group,
        )

    # -- checkpointing ---------------------------------------------------------------------------
    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None
    ) -> ShardedStateDict:
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
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

    # -- forward ---------------------------------------------------------------------------------
    def _project(self, hidden_states: torch.Tensor):
        """Return head-interleaved ``[.., local_conv_dim]`` qkv and ``z`` / ``b`` / ``a`` for the local heads."""
        batch, seq_len, _ = hidden_states.shape
        lk, g = self.local_num_k_heads, self.layout.group
        hk, hv = self.layout.head_k_dim, self.layout.head_v_dim
        if self.layout.hf_layout == "qwen3_next":
            qkvz, _ = self.in_proj_qkvz(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            qkvz = qkvz.reshape(batch, seq_len, lk, 2 * hk + 2 * g * hv)
            qkv, z = torch.split(qkvz, [2 * hk + g * hv, g * hv], dim=3)
            b, a = torch.split(ba.reshape(batch, seq_len, lk, 2 * g), [g, g], dim=3)
            return (
                qkv.reshape(batch, seq_len, self.local_conv_dim),
                z.reshape(batch, seq_len, -1),
                b.reshape(batch, seq_len, -1),
                a.reshape(batch, seq_len, -1),
            )
        qkv, _ = self.in_proj_qkv(hidden_states)
        z, _ = self.in_proj_z(hidden_states)
        b, _ = self.in_proj_b(hidden_states)
        a, _ = self.in_proj_a(hidden_states)
        return qkv, z, b, a

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor | None = None, cp_context: Any = None
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        layout = self.layout
        lk, g = self.local_num_k_heads, layout.group
        hk, hv = layout.head_k_dim, layout.head_v_dim

        mixed_qkv, z, b, a = self._project(hidden_states)
        # Depthwise conv over the head-interleaved channels (conv1d.weight rows use the same order).
        conv_cu_seqlens = cp_context.cu_seqlens if cp_context is not None else cu_seqlens
        mixed_qkv, _ = self.conv1d(x=mixed_qkv, cu_seqlens=conv_cu_seqlens, cp_context=cp_context)
        query, key, value = torch.split(
            mixed_qkv.reshape(batch, seq_len, lk, layout.rows_per_k_head), [hk, hk, g * hv], dim=3
        )
        query = query.reshape(batch, seq_len, lk, hk)
        key = key.reshape(batch, seq_len, lk, hk)
        value = value.reshape(batch, seq_len, lk * g, hv)
        z = z.reshape(batch, seq_len, lk * g, hv)

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        gate = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if g > 1 and self.gdn_backend != "loom":
            # The deterministic kernels take grouped value heads directly; fla/flashqla want them expanded.
            query = query.repeat_interleave(g, dim=2)
            key = key.repeat_interleave(g, dim=2)

        if cp_context is not None:
            if self.gdn_backend not in ("fla", "loom"):
                raise NotImplementedError(
                    f"GDN context parallelism requires the 'fla' or 'loom' backend, got {self.gdn_backend!r}."
                )
            core_attn_out, _ = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=gate,
                beta=beta,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cp_context.cu_seqlens,
                cp_context=cp_context,
            )
        else:
            if self.gdn_backend == "flashqla":
                query, key, value, gate, beta = (t.contiguous() for t in (query, key, value, gate, beta))
            core_attn_out, _ = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=gate,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )

        core_attn_out = self.norm(core_attn_out.reshape(-1, hv), z.reshape(-1, hv))
        core_attn_out = core_attn_out.reshape(batch, seq_len, self.local_value_dim)
        output, _ = self.out_proj(core_attn_out)
        return output
