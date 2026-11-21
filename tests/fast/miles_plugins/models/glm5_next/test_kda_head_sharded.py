"""GLM-5.3-Flash KDA head-sharded over TP (``--glm5-kda-tp-sharding``) against the replicated module, on CPU.

Two gloo ranks each build a head-sharded ``Glm5NextKDAAttention`` from their part of one replicated
``Glm5NextKDA``; the summed layer output, the input gradient and every weight gradient must equal the replicated
module's, with and without sequence parallelism. Without a TP group, or with a one-rank one, ``Glm5NextKDA`` is
the replicated module. The fla kernels are replaced by plain-torch stand-ins (``kda_tp_utils``).
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from tests.fast.dist_utils import init_gloo, run_multiprocess
from tests.fast.miles_plugins.models.glm5_next.kda_tp_utils import (
    TEXT_CONFIG,
    build_attention,
    build_replicated,
    head_rows,
    head_slice,
    install_cpu_kernels,
    load_shard,
    patch_cpu_kernels,
)

pytest.importorskip("megatron.core")

TP = 2
SEQ_LEN = 12  # two packed sequences, divisible by TP for sequence parallelism
CU_SEQLENS = torch.tensor([0, 5, SEQ_LEN])


def _reference_grad(full, sharded, name: str, tp_rank: int) -> torch.Tensor:
    """The replicated module's gradient for this rank's part of ``name``."""
    rows, heads = head_rows(sharded, tp_rank), head_slice(sharded, tp_rank)
    if name.endswith("_conv1d.weight"):
        offset = ("q_conv1d", "k_conv1d", "v_conv1d").index(name.split(".")[0]) * full.projection_size
        return full.conv1d.weight.grad[offset + rows.start : offset + rows.stop]
    grad = full.get_parameter(name).grad
    if name in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "f_b_proj.weight", "g_b_proj.weight", "dt_bias"):
        return grad[rows]
    if name in ("b_proj.weight", "A_log"):
        return grad[heads]
    if name == "o_proj.weight":
        return grad[:, rows]
    assert name in ("f_a_proj.weight", "g_a_proj.weight", "o_norm.weight"), name
    return grad


def _check_matches_replicated(rank: int, world_size: int, port: int, sequence_parallel: bool) -> None:
    init_gloo(rank, world_size, port=port)
    # Megatron's sequence-parallel gather / reduce-scatter allocate their buffers on the current CUDA device
    torch.cuda.current_device = lambda: "cpu"
    try:
        from miles_plugins.models.glm5_next import kda

        install_cpu_kernels(kda)
        tp_group = dist.new_group(list(range(world_size)))
        torch.manual_seed(0)
        full = build_replicated(kda)
        x = torch.randn(SEQ_LEN, 1, TEXT_CONFIG.hidden_size)  # [s, b, h], the layer's layout

        x_ref = x.clone().requires_grad_(True)
        reference = full(x_ref.transpose(0, 1), CU_SEQLENS).transpose(0, 1)
        reference.square().sum().backward()

        layer = build_attention(kda, tp_group, sequence_parallel=sequence_parallel)
        sharded = layer.kda
        assert sharded.num_heads == TEXT_CONFIG.linear_num_heads // TP
        load_shard(full, sharded, rank)

        local = slice(rank * SEQ_LEN // TP, (rank + 1) * SEQ_LEN // TP) if sequence_parallel else slice(None)
        x_local = x[local].clone().requires_grad_(True)
        output, bias = layer(x_local, packed_seq_params=SimpleNamespace(cu_seqlens_q=CU_SEQLENS))
        assert bias is None
        output.square().sum().backward()

        torch.testing.assert_close(output, reference[local], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(x_local.grad, x_ref.grad[local], rtol=1e-4, atol=1e-5)
        for name, param in sharded.named_parameters():
            grad = param.grad.clone()
            if getattr(param, "sum_gradients_across_tp_domain", False):
                dist.all_reduce(grad, group=tp_group)  # finalize_model_grads does this
            torch.testing.assert_close(
                grad,
                _reference_grad(full, sharded, name, rank),
                rtol=1e-4,
                atol=1e-5,
                msg=lambda m, n=name: f"{n}: {m}",
            )
    finally:
        dist.destroy_process_group()


def _check_sharded_state_dict(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        from miles_plugins.models.glm5_next import kda

        install_cpu_kernels(kda)
        tp_group = dist.new_group(list(range(world_size)))
        dp_cp_groups = [dist.new_group([r]) for r in range(world_size)]
        layer = build_attention(kda, tp_group)
        sharded = layer.kda
        prefix = "decoder.layers.0.self_attention.kda."
        state = layer.sharded_state_dict(
            prefix="decoder.layers.0.self_attention.", metadata={"dp_cp_group": dp_cp_groups[rank]}
        )

        tc = TEXT_CONFIG
        projection = tc.linear_num_heads * tc.linear_head_dim
        local = projection // TP
        for j, conv in enumerate(("q_conv1d", "k_conv1d", "v_conv1d")):
            # fragment j * TP + rank of the replicated module's [q | k | v] conv weight
            tensor = state[f"{prefix}{conv}.weight"]
            assert tensor.key == f"{prefix}conv1d.weight"
            assert tensor.global_shape[0] == 3 * projection
            assert tensor.global_offset[0] == j * projection + rank * local
        for name, axis in kda._TP_AXIS.items():
            tensor = state[f"{prefix}{name}"]
            assert tensor.key == f"{prefix}{name}"
            assert tensor.global_shape[axis] == tensor.local_shape[axis] * TP
            assert tensor.global_offset[axis] == tensor.local_shape[axis] * rank
        for name in ("f_a_proj.weight", "g_a_proj.weight", "o_norm.weight"):
            tensor = state[f"{prefix}{name}"]
            assert tuple(tensor.global_shape) == tuple(tensor.local_shape)
        assert set(state) == {f"{prefix}{name}" for name, _ in sharded.named_parameters()}
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_head_sharded_kda_matches_the_replicated_module(sequence_parallel: bool) -> None:
    run_multiprocess(_check_matches_replicated_sp if sequence_parallel else _check_matches_replicated_tp, TP)


def _check_matches_replicated_tp(rank: int, world_size: int, port: int) -> None:
    _check_matches_replicated(rank, world_size, port, sequence_parallel=False)


def _check_matches_replicated_sp(rank: int, world_size: int, port: int) -> None:
    _check_matches_replicated(rank, world_size, port, sequence_parallel=True)


def test_head_sharded_kda_checkpoints_under_the_replicated_keys() -> None:
    """torch_dist checkpoints of the replicated module load into the head-sharded one (and back)."""
    run_multiprocess(_check_sharded_state_dict, TP)


# the replicated module's parameters in registration order (its own parameters first, as nn.Module lists them)
_REPLICATED_PARAMETERS = {
    "A_log": (4,),
    "dt_bias": (32,),
    "q_proj.weight": (32, 32),
    "k_proj.weight": (32, 32),
    "v_proj.weight": (32, 32),
    "conv1d.weight": (96, 1, 4),
    "b_proj.weight": (4, 32),
    "f_a_proj.weight": (8, 32),
    "f_b_proj.weight": (32, 8),
    "g_a_proj.weight": (8, 32),
    "g_b_proj.weight": (32, 8),
    "o_norm.weight": (8,),
    "o_proj.weight": (32, 32),
}


class _OneRankGroup:
    def size(self) -> int:
        return 1

    def rank(self) -> int:
        return 0


def test_kda_without_a_multi_rank_tp_group_is_the_replicated_module(monkeypatch) -> None:
    """No TP group, or a one-rank one (``--glm5-kda-tp-sharding`` at TP=1), builds the replicated module: its
    parameter names and shapes (one ``conv1d``), no TP attributes, and the same output."""
    from miles_plugins.models.glm5_next import kda

    patch_cpu_kernels(monkeypatch, kda)
    torch.manual_seed(0)
    replicated = build_replicated(kda)
    one_rank = build_replicated(kda, tp_group=_OneRankGroup())
    one_rank.load_state_dict(replicated.state_dict())
    for module in (replicated, one_rank):
        assert not module.tp_sharded
        assert [(name, tuple(p.shape)) for name, p in module.named_parameters()] == list(
            _REPLICATED_PARAMETERS.items()
        )
        for param in module.parameters():
            assert not getattr(param, "tensor_model_parallel", False)
            assert not getattr(param, "sum_gradients_across_tp_domain", False)
    x = torch.randn(1, SEQ_LEN, TEXT_CONFIG.hidden_size)
    torch.testing.assert_close(one_rank(x, CU_SEQLENS), replicated(x, CU_SEQLENS), rtol=0, atol=0)

    layer = build_attention(kda, _OneRankGroup())
    assert not layer.kda.tp_sharded
    assert not hasattr(layer, "tp_group")  # MegatronModule.sharded_state_dict keeps its replicated default
