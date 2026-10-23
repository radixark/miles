"""TorchTitan DTensor sharding for K3 TP/SP and expert parallelism."""

import spmd_types as spmd

from torchtitan.models.common.decoder_sharding import (
    colwise_config,
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    rowwise_config,
)
from torchtitan.models.common.moe_sharding import expert_param_placement_sparse
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig

from miles.backends.torchtitan_utils.models.kimi_k3.attention import DeltaAttention
from miles.backends.torchtitan_utils.models.kimi_k3.ops import RMSNorm


def _replicated_state(module):
    names = [name for name, _ in module.named_parameters(recurse=False)]
    names.extend(name for name, _ in module.named_buffers(recurse=False))
    module._sharding_config = ShardingConfig(
        state_shardings={name: dense_param_placement(tp=spmd.R) for name in names}
    )


def _feed_forward(module):
    replicated = dense_activation_placement(tp=spmd.R)
    module._sharding_config = ShardingConfig(
        in_src_shardings={"x": dense_sequence_parallel_placement()}, in_dst_shardings={"x": replicated}
    )
    module.gate_proj._sharding_config = colwise_config()
    module.up_proj._sharding_config = colwise_config()
    module.down_proj._sharding_config = rowwise_config(output_sp=True)


def _attention(module, parallel_dims):
    replicated = dense_activation_placement(tp=spmd.R)
    module._sharding_config = ShardingConfig(
        in_src_shardings={"x": dense_sequence_parallel_placement()}, in_dst_shardings={"x": replicated}
    )
    module.o_proj._sharding_config = rowwise_config(output_sp=True)
    module.g_proj._sharding_config = colwise_config()
    if isinstance(module, DeltaAttention):
        for name in ("q_proj", "k_proj", "v_proj", "f_b_proj", "b_proj"):
            getattr(module, name)._sharding_config = colwise_config()
        _replicated_state(module.f_a_proj)
        for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
            conv = getattr(module, name)
            conv._sharding_config = ShardingConfig(
                state_shardings={key: dense_param_placement(tp=spmd.S(0)) for key, _ in conv.named_parameters()}
            )
        module.core.tp_rank = parallel_dims.get_mesh("tp").get_local_rank()
        module.core._sharding_config = ShardingConfig(
            state_shardings={
                # A_log has 128 checkpoint slots for 96 heads. Keep its padding intact;
                # each rank consumes its active heads and contributes partial gradients.
                "A_log": dense_param_placement(tp=spmd.R),
                "dt_bias": dense_param_placement(tp=spmd.S(0)),
            }
        )
    else:
        for name in ("q_a_proj", "kv_a_proj_with_mqa"):
            _replicated_state(getattr(module, name))
        for name in ("q_b_proj", "kv_b_proj"):
            getattr(module, name)._sharding_config = colwise_config()


def _experts(module, *, enable_tp):
    sequence = dense_sequence_parallel_placement()
    count_layout = dense_activation_placement(tp=spmd.P)
    _replicated_state(module.gate)
    _replicated_state(module.routed_expert_down_proj)
    _replicated_state(module.routed_expert_up_proj)
    if enable_tp:
        _feed_forward(module.shared_experts)
    routed = module.routed_experts
    routed._sharding_config = ShardingConfig(
        in_src_shardings={
            "x_BLD": sequence,
            "topk_scores_BLK": sequence,
            "topk_expert_ids_BLK": sequence,
            "num_local_tokens_per_expert_E": count_layout,
        },
        out_src_shardings=sequence,
        local_map=LocalMapConfig(in_grad_placements=(sequence, sequence, sequence, count_layout)),
    )
    routed.inner_experts._sharding_config = ShardingConfig(
        state_shardings={name: expert_param_placement_sparse() for name, _ in routed.inner_experts.named_parameters()}
    )


def shard_kimi_k3(model, parallel_dims):
    if parallel_dims.spmd_backend != "partial_dtensor":
        raise ValueError("K3 sharding currently requires partial_dtensor")
    if parallel_dims.tp_enabled:
        for module in model.modules():
            if isinstance(module, RMSNorm):
                _replicated_state(module)
        for layer in model.layers.values():
            _replicated_state(layer.self_attention_res_proj)
            _replicated_state(layer.mlp_res_proj)
            _attention(layer.self_attn, parallel_dims)
            if not layer.moe_enabled:
                _feed_forward(layer.mlp)
        _replicated_state(model.output_attn_res_proj)
        replicated = dense_activation_placement(tp=spmd.R)
        sequence = dense_sequence_parallel_placement()
        model.tok_embeddings._sharding_config = ShardingConfig(
            state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
            in_src_shardings={"input": replicated},
            out_src_shardings=dense_activation_placement(tp=spmd.P),
            out_dst_shardings=sequence,
            local_map=LocalMapConfig(in_grad_placements=None),
        )
        model.norm._sharding_config = ShardingConfig(
            state_shardings={"weight": dense_param_placement(tp=spmd.R)},
            in_src_shardings={"x": sequence},
            out_dst_shardings=replicated,
        )
        model.lm_head._sharding_config = colwise_config()
    if parallel_dims.ep_enabled:
        for layer in model.layers.values():
            if layer.moe_enabled:
                _experts(layer.moe, enable_tp=parallel_dims.tp_enabled)
    model.parallelize(parallel_dims)
