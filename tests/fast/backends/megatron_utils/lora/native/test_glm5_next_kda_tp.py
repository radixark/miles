"""Native LoRA on GLM-5.3-Flash's head-sharded KDA (``--glm5-kda-tp-sharding``), on CPU.

The nine KDA adapters on two gloo ranks must reproduce the replicated module's adapters exactly (output and every
``A`` / ``B`` gradient, after the TP gradient sums the adapters mark), and the row-parallel ``o_proj`` adapter must
start from the distribution of a full-width ``A`` at any TP size.
"""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from tests.fast.backends.megatron_utils.lora.native.naming_stub import FixedPrefixNaming
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

from miles.backends.megatron_utils.lora.native.config import LoRAConfig  # noqa: E402
from miles.backends.megatron_utils.lora.native.modules.linear import (  # noqa: E402
    LoRAGlobalFanUnreducedRowLinear,
    LoRAPartialReplicatedPlainLinear,
    LoRAShardedPlainColumnLinear,
    iter_adapters,
)
from miles.backends.megatron_utils.lora.native.spec.attention import Glm5NextAttentionSpec  # noqa: E402
from miles.backends.megatron_utils.lora.native.spec.base import AttachContext, ShardLayout  # noqa: E402

TP = 2
SEQ_LEN = 10
CU_SEQLENS = torch.tensor([0, 4, SEQ_LEN])
_KDA_ADAPTERS = ("q", "k", "v", "b", "f_a", "f_b", "g_a", "g_b", "o")
_EXPECTED_ADAPTERS = {
    "q": LoRAShardedPlainColumnLinear,
    "k": LoRAShardedPlainColumnLinear,
    "v": LoRAShardedPlainColumnLinear,
    "b": LoRAShardedPlainColumnLinear,
    "f_a": LoRAPartialReplicatedPlainLinear,
    "f_b": LoRAShardedPlainColumnLinear,
    "g_a": LoRAPartialReplicatedPlainLinear,
    "g_b": LoRAShardedPlainColumnLinear,
    "o": LoRAGlobalFanUnreducedRowLinear,
}


def _context(tp_size: int, tp_rank: int, *, sequence_parallel: bool = False, rank: int = 3, init: str = "xavier"):
    return AttachContext(
        lora=LoRAConfig(rank=rank, alpha=2.0 * rank, dropout=0.0, target_modules=None, a_init_method=init),
        transformer_config=SimpleNamespace(sequence_parallel=sequence_parallel, context_parallel_size=1),
        tp_size=tp_size,
        tp_rank=tp_rank,
        naming=FixedPrefixNaming("model.language_model.layers.0.self_attn."),
    )


def _adapters(block: nn.Module) -> dict:
    return {adapter.param_prefix: adapter for adapter in iter_adapters([block])}


def _ab(adapter):
    prefix = adapter.param_prefix
    return getattr(adapter, f"{prefix}_A"), getattr(adapter, f"{prefix}_B")


def _local_a(adapter, a: torch.Tensor, rows: slice) -> torch.Tensor:
    return a[:, rows] if adapter.layout == ShardLayout.ROW else a


def _local_b(adapter, b: torch.Tensor, rows: slice, heads: slice) -> torch.Tensor:
    if adapter.layout != ShardLayout.COLUMN:
        return b
    return b[heads] if adapter.param_prefix == "b" else b[rows]


def _check_lora_matches_replicated(rank: int, world_size: int, port: int, sequence_parallel: bool) -> None:
    init_gloo(rank, world_size, port=port)
    torch.cuda.current_device = lambda: "cpu"  # Megatron's SP collectives allocate on the current CUDA device
    try:
        from miles_plugins.models.glm5_next import kda

        install_cpu_kernels(kda)
        spec = Glm5NextAttentionSpec()
        tp_group = dist.new_group(list(range(world_size)))
        torch.manual_seed(0)
        full = build_replicated(kda)
        replicated_block = nn.Module()
        replicated_block.kda = full
        assert spec.attach(replicated_block, SimpleNamespace(layer_number=1), _context(1, 0)) == len(_KDA_ADAPTERS)
        reference_adapters = _adapters(replicated_block)
        torch.manual_seed(1)
        with torch.no_grad():
            for adapter in reference_adapters.values():
                for param in _ab(adapter):
                    param.copy_(torch.randn_like(param) * 0.3)  # nonzero B, so every A gets a gradient
        x = torch.randn(SEQ_LEN, 1, TEXT_CONFIG.hidden_size)
        reference = full(x.transpose(0, 1), CU_SEQLENS).transpose(0, 1)
        reference.square().sum().backward()

        layer = build_attention(kda, tp_group, sequence_parallel=sequence_parallel)
        sharded = layer.kda
        load_shard(full, sharded, rank)
        context = _context(world_size, rank, sequence_parallel=sequence_parallel)
        assert spec.attach(layer, SimpleNamespace(layer_number=1), context) == len(_KDA_ADAPTERS)
        adapters = _adapters(layer)
        assert {prefix: type(adapter) for prefix, adapter in adapters.items()} == _EXPECTED_ADAPTERS
        rows, heads = head_rows(sharded, rank), head_slice(sharded, rank)
        with torch.no_grad():
            for prefix, adapter in adapters.items():
                a, b = _ab(adapter)
                ref_a, ref_b = _ab(reference_adapters[prefix])
                a.copy_(_local_a(adapter, ref_a, rows))
                b.copy_(_local_b(adapter, ref_b, rows, heads))

        local = slice(rank * SEQ_LEN // TP, (rank + 1) * SEQ_LEN // TP) if sequence_parallel else slice(None)
        output, _ = layer(x[local].clone(), packed_seq_params=SimpleNamespace(cu_seqlens_q=CU_SEQLENS))
        output.square().sum().backward()
        for param in layer.parameters():  # what reduce_marked_lora_grads does over TP
            if getattr(param, "_lora_grad_sum_group", None) == "tp" and param.grad is not None:
                dist.all_reduce(param.grad, group=tp_group)

        torch.testing.assert_close(output, reference[local], rtol=1e-4, atol=1e-5)
        for prefix, adapter in adapters.items():
            (a, b), (ref_a, ref_b) = _ab(adapter), _ab(reference_adapters[prefix])
            torch.testing.assert_close(
                a.grad,
                _local_a(adapter, ref_a.grad, rows),
                rtol=1e-4,
                atol=1e-5,
                msg=lambda m, p=prefix: f"{p}_A: {m}",
            )
            torch.testing.assert_close(
                b.grad,
                _local_b(adapter, ref_b.grad, rows, heads),
                rtol=1e-4,
                atol=1e-5,
                msg=lambda m, p=prefix: f"{p}_B: {m}",
            )
    finally:
        dist.destroy_process_group()


def _check_lora_matches_replicated_tp(rank: int, world_size: int, port: int) -> None:
    _check_lora_matches_replicated(rank, world_size, port, sequence_parallel=False)


def _check_lora_matches_replicated_sp(rank: int, world_size: int, port: int) -> None:
    _check_lora_matches_replicated(rank, world_size, port, sequence_parallel=True)


@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_head_sharded_kda_adapters_match_the_replicated_adapters(sequence_parallel: bool) -> None:
    run_multiprocess(_check_lora_matches_replicated_sp if sequence_parallel else _check_lora_matches_replicated_tp, TP)


class _FakeGroup:
    """A TP rank's view of its group at build time; the head-sharded KDA only reads its size and rank there."""

    def __init__(self, size: int, rank: int):
        self._size, self._rank = size, rank

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank


# std of one [rank, fan_in] matrix under each init (xavier: fan_out = rank)
_EXPECTED_STD = {
    "xavier": lambda fan_in, rank: math.sqrt(2.0 / (fan_in + rank)),
    "kaiming": lambda fan_in, rank: 1.0 / math.sqrt(3.0 * fan_in),
}


@pytest.mark.parametrize("init", sorted(_EXPECTED_STD))
def test_o_proj_adapter_a_starts_at_the_full_width_distribution(monkeypatch, init: str) -> None:
    """A row-parallel A initialized per shard sees fan-in ``in / TP`` and starts sqrt(TP)-ish too large; the o_proj
    adapter draws the full-width A on every rank (seeded alike, as Megatron seeds the default generator) and keeps
    its columns."""
    from miles_plugins.models.glm5_next import kda

    patch_cpu_kernels(monkeypatch, kda)
    tp_size, lora_rank = 8, 4
    text_config = SimpleNamespace(**{**vars(TEXT_CONFIG), "linear_num_heads": 8, "hidden_size": 64})
    fan_in = text_config.linear_num_heads * text_config.linear_head_dim
    spec = Glm5NextAttentionSpec()
    ratios = []
    for seed in range(4):
        shards = []
        for tp_rank in range(tp_size):
            block = nn.Module()
            block.kda = kda.Glm5NextKDA(
                text_config.hidden_size,
                text_config.linear_num_heads,
                text_config.linear_head_dim,
                text_config.linear_conv_kernel_dim,
                text_config.linear_lower_bound,
                text_config.rms_norm_eps,
                tp_group=_FakeGroup(tp_size, tp_rank),
            )
            torch.manual_seed(seed)
            spec.attach(block, SimpleNamespace(layer_number=1), _context(tp_size, tp_rank, rank=lora_rank, init=init))
            shards.append(_adapters(block)["o"].o_A.detach())
        full_a = torch.cat(shards, dim=1)
        assert full_a.shape == (lora_rank, fan_in)
        assert not torch.equal(shards[0], shards[1]), "every rank must keep different columns"
        ratios.append(full_a.std().item() / _EXPECTED_STD[init](fan_in, lora_rank))
    assert 0.9 < sum(ratios) / len(ratios) < 1.1, ratios
