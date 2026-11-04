"""Native multi-LoRA slots against single LoRA on the same weights — no GPU."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from miles.backends.megatron_utils.lora.native import checkpoint as native_checkpoint
from miles.backends.megatron_utils.lora.native.config import LoRAConfig
from miles.backends.megatron_utils.lora.native.hf_adapter import export_lora_hf_named
from miles.backends.megatron_utils.lora.native.modules.linear import (
    LoRALinear,
    LoRASplitFC1,
    NativeMultiLoRAAdapter,
    adapters_hidden,
    assert_every_adapter_slotted,
    attach_adapter_forward,
    build_adapter,
    iter_adapters,
    iter_slot_adapters,
)
from miles.backends.megatron_utils.lora.native.modules.lm_head import LoRAOutputHead
from miles.backends.megatron_utils.lora.native.spec.base import AttachContext, ProjectionSpec, ShardLayout
from miles.backends.megatron_utils.lora.slots import slice_lora_to_rank

HIDDEN, OUT, MAX_RANK, NUM_SLOTS = 8, 6, 4, 3


class _Host(nn.Module):
    """A Megatron linear's contract: ``(out, bias)``."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight.clone(), requires_grad=False)

    def forward(self, x):
        return x @ self.weight.t(), None


def _context(*, rank, alpha=8.0, num_slots=None, tp_size=1, tp_rank=0):
    return AttachContext(
        lora=LoRAConfig(rank=rank, alpha=alpha, dropout=0.0, target_modules=None, num_slots=num_slots),
        transformer_config=SimpleNamespace(hidden_size=HIDDEN, sequence_parallel=False),
        tp_size=tp_size,
        tp_rank=tp_rank,
        naming=None,
    )


def _column(host, context):
    return LoRALinear(
        hf_prefix="model.layers.0.mlp.",
        projection=ProjectionSpec("up_proj", "up", ShardLayout.COLUMN),
        reference=host.weight,
        context=context,
        in_features=HIDDEN,
        out_features=OUT,
    )


def _row(host, context):
    return LoRALinear(
        hf_prefix="model.layers.0.mlp.",
        projection=ProjectionSpec("down_proj", "down", ShardLayout.ROW),
        reference=host.weight,
        context=context,
        in_features=HIDDEN,
        out_features=OUT,
    )


def _split_fc1(host, context):
    return LoRASplitFC1(
        hf_prefix="model.layers.0.mlp.",
        reference=host.weight,
        context=context,
        projections=(
            ProjectionSpec("gate_proj", "gate", ShardLayout.COLUMN),
            ProjectionSpec("up_proj", "up", ShardLayout.COLUMN),
        ),
        inter_local=OUT // 2,
    )


def _lm_head(host, context):
    return LoRAOutputHead(
        hf_prefix="lm_head.", reference=host.weight, context=context, vocab_local=OUT, unpadded_vocab_size=OUT - 2
    )


ADAPTER_BUILDS = {"column": _column, "row": _row, "split_fc1": _split_fc1, "lm_head": _lm_head}


class _Chunk(nn.Module):
    def __init__(self, host, adapter):
        super().__init__()
        self.host = host
        self.adapter = adapter


def _attached(build, context, base_weight):
    host = _Host(base_weight)
    adapter = build_adapter(lambda: build(host, context), context)
    attach_adapter_forward(host, adapter)
    return _Chunk(host, adapter)


def _single(build, *, rank, alpha, base_weight, seed):
    chunk = _attached(build, _context(rank=rank, alpha=alpha), base_weight)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in chunk.adapter.parameters():
            param.copy_(torch.randn(param.shape, generator=generator))
    return chunk


def _load_slot(
    slot_adapters: NativeMultiLoRAAdapter, slot: int, single: nn.Module, *, rank: int, alpha: float
) -> None:
    slot_adapters.init_slot(slot, rank, alpha)
    with torch.no_grad():
        for (name, slot_param), single_param in zip(
            slot_adapters.slots[slot].named_parameters(), single.parameters(), strict=True
        ):
            if name.endswith("_A"):
                slot_param[:rank].copy_(single_param)
            else:
                slot_param[:, :rank].copy_(single_param)


def _base_weight():
    return torch.randn(OUT, HIDDEN, generator=torch.Generator().manual_seed(0))


def _tokens(num_tokens, seed=1):
    return torch.randn(num_tokens, 1, HIDDEN, generator=torch.Generator().manual_seed(seed)).requires_grad_()


@pytest.mark.parametrize("build", ADAPTER_BUILDS.values(), ids=ADAPTER_BUILDS)
@pytest.mark.parametrize("slot", [0, 2])
@pytest.mark.parametrize("rank", [MAX_RANK, 2])
def test_one_routed_slot_equals_single_lora_bitwise(build, slot, rank):
    base_weight, alpha = _base_weight(), 6.0
    single = _single(build, rank=rank, alpha=alpha, base_weight=base_weight, seed=7)
    multi = _attached(build, _context(rank=MAX_RANK, num_slots=NUM_SLOTS), base_weight)
    for other in range(NUM_SLOTS):
        other_single = _single(build, rank=rank, alpha=3.0, base_weight=base_weight, seed=other)
        _load_slot(multi.adapter, other, other_single.adapter, rank=rank, alpha=3.0)
    _load_slot(multi.adapter, slot, single.adapter, rank=rank, alpha=alpha)
    counts = [0] * NUM_SLOTS
    counts[slot] = 5
    multi.adapter.token_counts_by_slot = tuple(counts)
    x_single, x_multi = _tokens(5), _tokens(5)

    out_single, _ = single.host(x_single)
    out_multi, _ = multi.host(x_multi)
    out_single.square().sum().backward()
    out_multi.square().sum().backward()

    assert torch.equal(out_multi, out_single)
    assert torch.equal(x_multi.grad, x_single.grad)
    for (name, slot_param), single_param in zip(
        multi.adapter.slots[slot].named_parameters(), single.adapter.parameters(), strict=True
    ):
        within = slot_param.grad[:rank] if name.endswith("_A") else slot_param.grad[:, :rank]
        assert torch.equal(within, single_param.grad), name
    for other in set(range(NUM_SLOTS)) - {slot}:
        assert all(torch.count_nonzero(p.grad) == 0 for p in multi.adapter.slots[other].parameters())


@pytest.mark.parametrize("build", ADAPTER_BUILDS.values(), ids=ADAPTER_BUILDS)
def test_mixed_slots_equal_each_slot_alone(build):
    base_weight = _base_weight()
    counts = (3, 0, 4)
    ranks, alphas = (4, 2, 3), (8.0, 2.0, 3.0)
    singles = [
        _single(build, rank=rank, alpha=alpha, base_weight=base_weight, seed=10 + slot)
        for slot, (rank, alpha) in enumerate(zip(ranks, alphas, strict=True))
    ]
    multi = _attached(build, _context(rank=MAX_RANK, num_slots=NUM_SLOTS), base_weight)
    for slot, single in enumerate(singles):
        _load_slot(multi.adapter, slot, single.adapter, rank=ranks[slot], alpha=alphas[slot])
    multi.adapter.token_counts_by_slot = counts
    x = _tokens(sum(counts))

    out, _ = multi.host(x)

    # the GEMMs see fewer rows alone, so their sums may round differently
    for slot, part in enumerate(x.detach().split(counts)):
        expected, _ = singles[slot].host(part)
        torch.testing.assert_close(out.split(counts)[slot], expected)


@pytest.mark.parametrize("build", ADAPTER_BUILDS.values(), ids=ADAPTER_BUILDS)
@pytest.mark.parametrize("rank", [MAX_RANK, 2])
def test_slot_export_equals_single_lora_export_bitwise(build, rank):
    base_weight = _base_weight()
    single = _single(build, rank=rank, alpha=4.0, base_weight=base_weight, seed=3)
    multi = _attached(build, _context(rank=MAX_RANK, num_slots=NUM_SLOTS), base_weight)
    _load_slot(multi.adapter, 1, single.adapter, rank=rank, alpha=4.0)

    expected = export_lora_hf_named(iter_adapters([single]))
    exported = [
        (name, slice_lora_to_rank(name, tensor, rank))
        for name, tensor in export_lora_hf_named(iter_slot_adapters([multi], 1))
    ]

    assert [name for name, _ in exported] == [name for name, _ in expected]
    for (name, tensor), (_, expected_tensor) in zip(exported, expected, strict=True):
        assert torch.equal(tensor, expected_tensor), name


def test_a_slot_claimed_below_max_rank_keeps_its_padding_out_of_training():
    multi = _attached(_column, _context(rank=MAX_RANK, num_slots=NUM_SLOTS), _base_weight())
    multi.adapter.init_slot(0, 2, 4.0)
    with torch.no_grad():
        multi.adapter.slots[0].up_B[:, :2].normal_()
    multi.adapter.token_counts_by_slot = (5, 0, 0)

    out, _ = multi.host(_tokens(5))
    out.square().sum().backward()

    adapter = multi.adapter.slots[0]
    assert torch.count_nonzero(adapter.up_A[2:]) == 0 and torch.count_nonzero(adapter.up_B[:, 2:]) == 0
    assert torch.count_nonzero(adapter.up_A.grad[2:]) == 0 and torch.count_nonzero(adapter.up_B.grad[:, 2:]) == 0
    assert torch.count_nonzero(adapter.up_A.grad[:2]) > 0


def test_a_cleared_slot_adds_nothing():
    base_weight = _base_weight()
    multi = _attached(_column, _context(rank=MAX_RANK, num_slots=NUM_SLOTS), base_weight)
    multi.adapter.init_slot(1, MAX_RANK, 8.0)
    with torch.no_grad():
        multi.adapter.slots[1].up_B.normal_()
    multi.adapter.clear_slot(1)
    multi.adapter.token_counts_by_slot = (0, 5, 0)
    x = _tokens(5)

    out, _ = multi.host(x)

    assert torch.equal(out, x @ base_weight.t())


def test_a_slot_forward_outside_the_routing_fails():
    multi = _attached(_column, _context(rank=MAX_RANK, num_slots=NUM_SLOTS), _base_weight())
    with pytest.raises(AssertionError, match="route_tokens"):
        multi.host(_tokens(5))


def test_a_sequence_parallel_shard_takes_its_window_of_each_slots_tokens():
    multi = _attached(_column, _context(rank=MAX_RANK, num_slots=NUM_SLOTS, tp_size=2, tp_rank=1), _base_weight())
    multi.adapter.token_counts_by_slot = (3, 2, 3)

    assert multi.adapter._local_token_counts(8) == [3, 2, 3]
    assert multi.adapter._local_token_counts(4) == [0, 1, 3]
    with pytest.raises(AssertionError, match="sequence-parallel shard"):
        multi.adapter._local_token_counts(5)


def test_a_multi_lora_model_is_read_per_slot():
    multi = _attached(_column, _context(rank=MAX_RANK, num_slots=NUM_SLOTS), _base_weight())

    assert list(iter_slot_adapters([multi], 2)) == [multi.adapter.slots[2]]
    with pytest.raises(AssertionError, match="iter_slot_adapters"):
        list(iter_adapters([multi]))


def _routed(num_slots=NUM_SLOTS):
    multi = _attached(_column, _context(rank=MAX_RANK, num_slots=num_slots), _base_weight())
    slots = native_checkpoint.NativeAdapterSlots.__new__(native_checkpoint.NativeAdapterSlots)
    slots._model, slots._multi_lora = [multi], True
    return multi, slots


def test_routing_takes_integer_counts_one_per_slot():
    multi, slots = _routed()
    for slot in range(NUM_SLOTS):
        multi.adapter.init_slot(slot, MAX_RANK, 4.0)

    slots.route_tokens(torch.tensor([2, 0, 3]))

    assert multi.adapter.token_counts_by_slot == (2, 0, 3)
    with pytest.raises(AssertionError, match="integers"):
        slots.route_tokens(torch.tensor([2.0, 0.0, 3.0]))
    with pytest.raises(AssertionError, match="2 token counts for 3 slots"):
        slots.route_tokens([2, 3])
    with pytest.raises(AssertionError, match="negative"):
        slots.route_tokens([2, -1, 3])


def test_routing_tokens_to_a_free_slot_fails():
    """A free slot's alpha is 0: its tenant would silently train against the base model."""
    multi, slots = _routed()
    multi.adapter.init_slot(0, MAX_RANK, 4.0)

    slots.route_tokens([5, 0, 0])
    with pytest.raises(AssertionError, match=r"free slots \[1\]"):
        slots.route_tokens([2, 3, 0])


def test_an_adapter_outside_the_slots_fails_a_multi_lora_attach():
    multi = _attached(_column, _context(rank=MAX_RANK, num_slots=NUM_SLOTS), _base_weight())
    assert_every_adapter_slotted([multi])
    multi.stray = _column(_Host(_base_weight()), _context(rank=MAX_RANK))

    with pytest.raises(AssertionError, match="have none"):
        assert_every_adapter_slotted([multi])


@pytest.mark.parametrize("num_slots", [None, NUM_SLOTS], ids=["single", "multi"])
def test_hidden_adapters_leave_only_base_parameters(num_slots):
    chunk = _attached(_column, _context(rank=MAX_RANK, num_slots=num_slots), _base_weight())
    before = list(chunk.state_dict())

    with adapters_hidden([chunk]):
        assert list(chunk.state_dict()) == ["host.weight"]

    assert list(chunk.state_dict()) == before
