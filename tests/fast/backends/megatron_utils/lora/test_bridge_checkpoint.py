"""The row-mapping probe recovers each rank's HF pieces, primaries and shared A from an export alone."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.megatron_utils.lora.bridge_checkpoint import (
    BridgeAdapterSlots,
    BridgeRowMapping,
    probe_bridge_row_mapping,
)
from miles.utils.lora.hf_lora_targets import get_hf_lora_targets

Q, K = "model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj"
TARGETS = get_hf_lora_targets({"model_type": "qwen3", "num_hidden_layers": 1})


def _export(params):
    """Bridge-like: B gathered over the TP split on rows, one A exported under both names."""
    a, b_q, b_k = params

    def gathered(local):
        parts = [torch.empty_like(local) for _ in range(dist.get_world_size())]
        dist.all_gather(parts, local.data)
        return torch.cat(parts)

    yield f"{Q}.lora_A.weight", a.data.clone()
    yield f"{Q}.lora_B.weight", gathered(b_q)
    yield f"{K}.lora_A.weight", a.data.clone()
    yield f"{K}.lora_B.weight", gathered(b_k)


def _probe(rank, world_size, port):
    init_gloo(rank, world_size, port=port)
    params = [
        torch.nn.Parameter(torch.randn(3, 5)),
        torch.nn.Parameter(torch.randn(2, 3)),
        torch.nn.Parameter(torch.randn(2, 3)),
    ]
    before = [p.data.clone() for p in params]
    mapping = probe_bridge_row_mapping(params, lambda: _export(params), TARGETS)

    assert mapping.shared_lora_a_groups == ((K, Q),)
    assert all(torch.equal(p.data, b) for p, b in zip(params, before, strict=True)), "the probe restores the params"
    pieces = {p.key: p for p in mapping.pieces}
    assert set(pieces) == {f"{K}.lora_A", f"{Q}.lora_B", f"{K}.lora_B"}, "a shared A is stored once"
    assert pieces[f"{Q}.lora_B"].offsets == (2 * rank, 0) and pieces[f"{Q}.lora_B"].global_shape == (4, 3)
    flat = torch.cat([p.data.reshape(-1) for p in params])
    assert torch.equal(flat[pieces[f"{K}.lora_B"].local_elements], params[2].data)
    assert torch.equal(flat[pieces[f"{K}.lora_A"].local_elements], params[0].data)
    dist.destroy_process_group()


def test_probe_recovers_pieces_and_shared_a_from_a_tp_split_export():
    run_multiprocess(_probe)


GATE, UP = "model.layers.0.mlp.experts.{}.gate_proj", "model.layers.0.mlp.experts.{}.up_proj"


def _expert_export(params):
    """Fused fc1 on 2 local experts per EP rank: one A per expert exported under gate and up; B rows split gate|up."""
    a, b = params

    def gathered(local):
        parts = [torch.empty_like(local) for _ in range(dist.get_world_size())]
        dist.all_gather(parts, local.data)
        return torch.cat(parts)

    all_a, all_b = gathered(a), gathered(b)
    for expert in range(all_a.shape[0]):
        yield f"{GATE.format(expert)}.lora_A.weight", all_a[expert].clone()
        yield f"{UP.format(expert)}.lora_A.weight", all_a[expert].clone()
        yield f"{GATE.format(expert)}.lora_B.weight", all_b[expert, :3].clone()
        yield f"{UP.format(expert)}.lora_B.weight", all_b[expert, 3:].clone()


def _probe_experts(rank, world_size, port):
    init_gloo(rank, world_size, port=port)
    params = [torch.nn.Parameter(torch.randn(2, 4, 5)), torch.nn.Parameter(torch.randn(2, 6, 4))]
    mapping = probe_bridge_row_mapping(params, lambda: _expert_export(params), TARGETS)
    gate, up = GATE.format("x").replace(".x.", "."), UP.format("x").replace(".x.", ".")
    assert mapping.shared_lora_a_groups == ((gate, up),)
    pieces = {p.key: p for p in mapping.pieces}
    assert set(pieces) == {f"{gate}.lora_A", f"{gate}.lora_B", f"{up}.lora_B"}, "the up copy of a shared A is dropped"
    assert pieces[f"{up}.lora_B"].offsets == (2 * rank, 0, 0) and pieces[f"{up}.lora_B"].global_shape == (4, 3, 4)
    flat = torch.cat([p.data.reshape(-1) for p in params])
    assert torch.equal(flat[pieces[f"{up}.lora_B"].local_elements], params[1].data[:, 3:])
    dist.destroy_process_group()


def test_probe_stacks_ep_split_experts_and_stores_a_shared_expert_a_once():
    run_multiprocess(_probe_experts)


HEAD = "lm_head"


def _trimmed_head_export(params, served_rows):
    """Native-like: B gathered over the vocab split, then trimmed to the served vocab."""
    a, b = params
    parts = [torch.empty_like(b) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, b.data)
    yield f"{HEAD}.lora_A.weight", a.data.clone()
    yield f"{HEAD}.lora_B.weight", torch.cat(parts)[:served_rows]


def _probe_trimmed_head(rank, world_size, port):
    init_gloo(rank, world_size, port=port)
    # vocab 7 padded to 8: rank 1's last row of B is padding
    params = [torch.nn.Parameter(torch.randn(2, 5)), torch.nn.Parameter(torch.randn(4, 2))]
    padding = torch.arange(10 + 3 * 2, 10 + 4 * 2) if rank == 1 else torch.empty(0, dtype=torch.long)
    with pytest.raises(RuntimeError, match="rank 1: AssertionError: .*does not reach 2 of this rank's elements"):
        probe_bridge_row_mapping(params, lambda: _trimmed_head_export(params, 7), TARGETS)

    mapping = probe_bridge_row_mapping(params, lambda: _trimmed_head_export(params, 7), TARGETS, unexported=padding)

    head_b = next(p for p in mapping.pieces if p.key == f"{HEAD}.lora_B")
    assert head_b.global_shape == (7, 2) and head_b.offsets == (4 * rank, 0)
    flat = torch.cat([p.data.reshape(-1) for p in params])
    assert torch.equal(flat[head_b.local_elements], params[1].data[: 4 - rank])
    with pytest.raises(RuntimeError, match="the export reaches elements declared unexported"):
        probe_bridge_row_mapping(params, lambda: _trimmed_head_export(params, 8), TARGETS, unexported=padding)
    dist.destroy_process_group()


def test_probe_exempts_only_the_declared_unexported_elements_from_coverage():
    run_multiprocess(_probe_trimmed_head)


GATE_UP = "model.layers.0.mlp.gate_up_proj"


def _fused_gate_up_export(params):
    """Native's fused gate_up at TP: gate and up gathered separately, then concatenated."""
    (b,) = params

    def gathered(local):
        parts = [torch.empty_like(local) for _ in range(dist.get_world_size())]
        dist.all_gather(parts, local.contiguous())
        return torch.cat(parts)

    half = b.shape[0] // 2
    yield f"{GATE_UP}.lora_B.weight", torch.cat([gathered(b.data[:half]), gathered(b.data[half:])])


def _probe_fused_gate_up(rank, world_size, port):
    init_gloo(rank, world_size, port=port)
    params = [torch.nn.Parameter(torch.randn(4, 3))]
    mapping = probe_bridge_row_mapping(params, lambda: _fused_gate_up_export(params), TARGETS)

    assert [(p.key, p.global_shape, p.offsets) for p in mapping.pieces] == [
        (f"{GATE_UP}.lora_B", (8, 3), (2 * rank, 0)),
        (f"{GATE_UP}.lora_B", (8, 3), (4 + 2 * rank, 0)),
    ]
    flat = params[0].data.reshape(-1)
    assert torch.equal(torch.cat([flat[p.local_elements] for p in mapping.pieces]), params[0].data)
    dist.destroy_process_group()


def test_probe_splits_a_ranks_bands_of_one_key_into_rectangles():
    run_multiprocess(_probe_fused_gate_up)


# process -> slot -> parameter indices it holds; process 0 holds the same in both slots, its peers do not
_HOLDS = [{0: {0}, 1: {0}}, {0: {1}, 1: {2}}, {0: {2}, 1: {1}}]


def _specs_of_two_slots(rank, world_size, port):
    dist.init_process_group(
        "gloo", init_method=f"tcp://localhost:{port}", rank=rank, world_size=world_size, timeout=timedelta(seconds=10)
    )
    slots = BridgeAdapterSlots.__new__(BridgeAdapterSlots)
    slots._specs, slots._multi_lora, slots.slot_parameters = {}, False, lambda slot: []
    slots.row_mapping = BridgeRowMapping(pieces=[], shared_lora_a_groups=(), target_modules=())
    for slot in (0, 1):
        assert slots.local_shard_specs(slot, frozenset(_HOLDS[rank][slot])) == []
    dist.destroy_process_group()


def test_every_process_gathers_shard_specs_for_each_slot():
    run_multiprocess(_specs_of_two_slots, world_size=3)
