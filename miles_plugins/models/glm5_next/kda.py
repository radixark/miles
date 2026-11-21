import logging

import torch
import torch.nn as nn
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.tensor_parallel.mappings import copy_to_tensor_model_parallel_region
from megatron.core.transformer.module import mark_keep_in_fp32
from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group, make_sharded_tensors_for_checkpoint

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.modules.fused_norm_gate import rms_norm_gated
    from fla.ops.kda import chunk_kda
    from fla.ops.kda.gate import fused_kda_gate
except ImportError:
    FusedRMSNormGated = None
    ShortConvolution = None
    rms_norm_gated = None
    chunk_kda = None
    fused_kda_gate = None

from miles_plugins.models.cp_utils import build_gdn_cp_context
from miles_plugins.models.hf_attention import HuggingfaceAttention
from miles_plugins.models.linear_attn import tp_region_in, tp_region_out

logger = logging.getLogger(__name__)

# head-sharded KDA: TP split dim of each sharded parameter; f_a_proj, g_a_proj and o_norm stay replicated
_TP_AXIS = {
    "q_proj.weight": 0,
    "k_proj.weight": 0,
    "v_proj.weight": 0,
    "b_proj.weight": 0,
    "f_b_proj.weight": 0,
    "g_b_proj.weight": 0,
    "A_log": 0,
    "dt_bias": 0,
    "o_proj.weight": 1,
}
_SHARDED_CONVS = ("q_conv1d", "k_conv1d", "v_conv1d")


def _get_text_config(hf_config):
    return (
        getattr(hf_config, "text_config", None) or hf_config
    )  # config-access-exempt: HF configs may wrap text_config for multimodal checkpoints


class Glm5NextKDA(nn.Module):
    """GLM-5.3 KDA, replicated, or head-sharded over ``tp_group`` when it has more than one rank.

    Head-sharded, each rank holds ``num_heads / TP`` heads: ``q/k/v_proj``, ``b_proj``, ``f_b_proj``, ``g_b_proj``,
    the short convs, ``A_log`` and ``dt_bias`` are split on dim 0, ``o_proj`` on dim 1 (its output is this rank's
    partial sum, which the caller reduces over TP). ``f_a_proj``, ``g_a_proj`` and ``o_norm`` stay replicated but feed
    this rank's heads only, so their gradients are summed over TP. The input is the full sequence, already through
    the TP collective. Parameter names are the replicated module's except the short conv: this rank's q, k and v
    channels are not contiguous in the replicated ``conv1d``, so they run as ``q_conv1d`` / ``k_conv1d`` /
    ``v_conv1d``, which checkpoint as fragments of ``conv1d.weight`` (:meth:`tp_sharded_state_dict`).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int,
        gate_lower_bound: float,
        rms_norm_eps: float,
        tp_group=None,
    ):
        super().__init__()
        if ShortConvolution is None or chunk_kda is None:
            raise ImportError("GLM-5.3 KDA requires flash-linear-attention >= 0.4.2 (fla.ops.kda).")
        tp_size = 1 if tp_group is None else tp_group.size()
        assert num_heads % tp_size == 0, f"{num_heads} KDA heads do not split across TP={tp_size}"
        self.tp_sharded = tp_size > 1
        self.tp_group = tp_group if self.tp_sharded else None
        self.hidden_size = hidden_size
        self.num_heads = num_heads // tp_size
        self.head_dim = head_dim
        self.projection_size = self.num_heads * head_dim
        self.conv_kernel_size = conv_kernel_size
        self.gate_lower_bound = gate_lower_bound

        self.q_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        if self.tp_sharded:
            self.q_conv1d, self.k_conv1d, self.v_conv1d = (self._short_conv(self.projection_size) for _ in range(3))
        else:
            self.conv1d = self._short_conv(3 * self.projection_size)
        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        self.f_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.f_b_proj = nn.Linear(head_dim, self.projection_size, bias=False)
        self.g_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.g_b_proj = nn.Linear(head_dim, self.projection_size, bias=False)
        self.A_log = mark_keep_in_fp32(nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32)))
        self.dt_bias = mark_keep_in_fp32(nn.Parameter(torch.zeros(self.projection_size, dtype=torch.float32)))
        self.o_norm = FusedRMSNormGated(head_dim, eps=rms_norm_eps, activation="sigmoid")
        self.o_proj = nn.Linear(self.projection_size, hidden_size, bias=False)

        if self.tp_sharded:
            params = dict(self.named_parameters())
            for name, axis in _TP_AXIS.items():
                set_tensor_model_parallel_attributes(params[name], True, axis, 1)
            for name in _SHARDED_CONVS:
                set_tensor_model_parallel_attributes(params[f"{name}.weight"], True, 0, 1)
            # o_norm's weight goes through the TP copy op in forward instead
            for name in ("f_a_proj.weight", "g_a_proj.weight"):
                params[name].sum_gradients_across_tp_domain = True

    def _short_conv(self, channels: int):
        return ShortConvolution(hidden_size=channels, kernel_size=self.conv_kernel_size, bias=False, activation="silu")

    def _short_conv_qkv(self, hidden_states, cu_seqlens, cp_context):
        qkv = (self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states))
        if self.tp_sharded:
            convs = (self.q_conv1d, self.k_conv1d, self.v_conv1d)
            return tuple(
                conv(x=x, cu_seqlens=cu_seqlens, cp_context=cp_context)[0] for conv, x in zip(convs, qkv, strict=True)
            )
        mixed_qkv, _ = self.conv1d(x=torch.cat(qkv, dim=-1), cu_seqlens=cu_seqlens, cp_context=cp_context)
        return torch.split(mixed_qkv, [self.projection_size] * 3, dim=-1)

    def _gated_norm(self, x, gate):
        if not self.tp_sharded:
            return self.o_norm(x, gate)
        # replicated o_norm over this rank's heads: the copy op sums its weight gradient over TP
        weight = copy_to_tensor_model_parallel_region(self.o_norm.weight, group=self.tp_group)
        return rms_norm_gated(x, gate, weight, self.o_norm.bias, self.o_norm.activation, eps=self.o_norm.eps)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor):
        cp_context = build_gdn_cp_context(self, cu_seqlens, hidden_states.device)

        conv_cu_seqlens = cp_context.cu_seqlens if cp_context is not None else cu_seqlens
        query, key, value = self._short_conv_qkv(hidden_states, conv_cu_seqlens, cp_context)
        query = query.unflatten(-1, (self.num_heads, self.head_dim))
        key = key.unflatten(-1, (self.num_heads, self.head_dim))
        value = value.unflatten(-1, (self.num_heads, self.head_dim))

        beta = torch.sigmoid(self.b_proj(hidden_states).float())
        forget = self.f_b_proj(self.f_a_proj(hidden_states))
        g = fused_kda_gate(
            forget.unflatten(-1, (self.num_heads, self.head_dim)),
            self.A_log,
            self.dt_bias,
            lower_bound=self.gate_lower_bound,
        )

        if cp_context is not None:
            core_attn_out, _ = chunk_kda(
                query,
                key,
                value,
                g=g,
                beta=beta,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cp_context.cu_seqlens,
                cp_context=cp_context,
            )
        else:
            core_attn_out, _ = chunk_kda(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )

        norm_gate = self.g_b_proj(self.g_a_proj(hidden_states))
        out_shape = core_attn_out.shape
        core_attn_out = self._gated_norm(
            core_attn_out.reshape(-1, self.head_dim),
            norm_gate.reshape(-1, self.head_dim),
        )
        core_attn_out = core_attn_out.reshape(out_shape[0], out_shape[1], -1)
        return self.o_proj(core_attn_out)

    def tp_sharded_state_dict(self, prefix: str, sharded_offsets: tuple, metadata: dict | None) -> dict:
        """Checkpoint entries of the head-sharded parameters, replacing the replicated ones Megatron's default
        builds for this module. Each q/k/v conv is fragment ``j * TP + tp_rank`` of ``3 * TP`` along dim 0 of the
        replicated ``[q | k | v]`` ``conv1d.weight``, so torch_dist checkpoints of either module load into the other.
        """
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        dp_cp_group = metadata["dp_cp_group"]
        params = dict(self.named_parameters())
        sharded = make_sharded_tensors_for_checkpoint(
            {name: params[name] for name in _TP_AXIS},
            prefix,
            _TP_AXIS,
            sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=dp_cp_group,
        )
        tp_size, tp_rank = self.tp_group.size(), self.tp_group.rank()
        prepend_axis_num = len(sharded_offsets)
        dp_cp_rank = dp_cp_group.rank() if dp_cp_group is not None else 0
        for j, name in enumerate(_SHARDED_CONVS):
            sharded[f"{prefix}{name}.weight"] = ShardedTensor.from_rank_offsets(
                f"{prefix}conv1d.weight",
                params[f"{name}.weight"],
                *sharded_offsets,
                (prepend_axis_num, j * tp_size + tp_rank, 3 * tp_size),
                replica_id=(0, 0, dp_cp_rank),
                prepend_axis_num=prepend_axis_num,
            )
        return sharded


class Glm5NextKDAAttention(HuggingfaceAttention):
    """``self_attention`` of a GLM-5.3 KDA layer. ``kda_tp_sharding`` (``--glm5-kda-tp-sharding``) at TP > 1 shards
    the KDA heads over TP and runs the collective pattern of
    :class:`miles_plugins.models.linear_attn.LinearAttentionLayer` instead of every rank running all heads on the
    gathered sequence. Context parallelism is not supported that way yet."""

    hybrid_cp = True

    def __init__(
        self,
        args,
        config,
        layer_number: int,
        cp_comm_type: str = "p2p",
        pg_collection=None,
        name: str | None = None,
        kda_tp_sharding: bool = False,
    ):
        super().__init__(args, config, layer_number, cp_comm_type, pg_collection, name=name)
        text_config = _get_text_config(self.hf_config)
        # the native config parses linear_attn_config into the linear_* fields
        if text_config.linear_lower_bound is None:
            raise ValueError("GLM-5.3 KDA requires gate_lower_bound (safe gate) in the HF config.")
        tp_group = None
        if kda_tp_sharding:
            if pg_collection is None:
                pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
            if pg_collection.tp.size() > 1:
                if pg_collection.cp.size() > 1:
                    raise NotImplementedError("--glm5-kda-tp-sharding does not support context parallelism yet.")
                # MegatronModule.sharded_state_dict builds this module's replicated entries over this group
                self.tp_group = tp_group = pg_collection.tp
        self.kda = Glm5NextKDA(
            hidden_size=text_config.hidden_size,
            num_heads=text_config.linear_num_heads,
            head_dim=text_config.linear_head_dim,
            conv_kernel_size=text_config.linear_conv_kernel_dim,
            gate_lower_bound=text_config.linear_lower_bound,
            rms_norm_eps=text_config.rms_norm_eps,
            tp_group=tp_group,
        )
        if self.kda.tp_sharded and layer_number == 1:
            logger.info(
                f"GLM-5.3 KDA head-sharded over TP: {self.kda.num_heads}/{text_config.linear_num_heads} heads per rank"
            )

    def forward(self, hidden_states, *args, **kwargs):
        if not self.kda.tp_sharded:
            return super().forward(hidden_states, *args, **kwargs)
        sequence_parallel = self.config.sequence_parallel
        x = tp_region_in(hidden_states, sequence_parallel, self.tp_group)
        output = self.kda(x.transpose(0, 1), cu_seqlens=kwargs["packed_seq_params"].cu_seqlens_q).transpose(0, 1)
        return tp_region_out(output, sequence_parallel, self.tp_group), None

    def hf_forward(self, hidden_states, packed_seq_params):
        return self.kda(hidden_states, cu_seqlens=packed_seq_params.cu_seqlens_q)

    def sharded_state_dict(self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None):
        sharded = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        if self.kda.tp_sharded:
            sharded.update(self.kda.tp_sharded_state_dict(f"{prefix}kda.", sharded_offsets, metadata))
        return sharded
