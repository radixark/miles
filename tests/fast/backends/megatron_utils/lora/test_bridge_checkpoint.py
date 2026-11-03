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
from miles.utils.distributed_utils import init_gloo_group
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
    init_gloo_group()
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
    init_gloo_group()
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


# process -> slot -> parameter indices it holds; process 0 holds the same in both slots, its peers do not
_HOLDS = [{0: {0}, 1: {0}}, {0: {1}, 1: {2}}, {0: {2}, 1: {1}}]


def _specs_of_two_slots(rank, world_size, port):
    dist.init_process_group(
        "gloo", init_method=f"tcp://localhost:{port}", rank=rank, world_size=world_size, timeout=timedelta(seconds=10)
    )
    init_gloo_group()
    slots = BridgeAdapterSlots.__new__(BridgeAdapterSlots)
    slots._specs, slots._multi_lora, slots.slot_parameters = {}, False, lambda slot: []
    slots.row_mapping = BridgeRowMapping(pieces=[], shared_lora_a_groups=(), target_modules=())
    for slot in (0, 1):
        assert slots.local_shard_specs(slot, frozenset(_HOLDS[rank][slot])) == []
    dist.destroy_process_group()


def test_every_process_gathers_shard_specs_for_each_slot():
    run_multiprocess(_specs_of_two_slots, world_size=3)


GDN = "model.layers.0.linear_attn"
GDN_TARGETS = get_hf_lora_targets(
    {
        "model_type": "qwen3_5_text",
        "num_hidden_layers": 1,
        "layer_types": ["linear_attention"],
        "linear_num_key_heads": 1,
        "linear_key_head_dim": 2,
        "linear_num_value_heads": 1,
        "linear_value_head_dim": 3,
    }
)


def _gdn_export(params):
    """Bridge-like GDN in_proj: the fused in_proj_qkv and in_proj_z export one A."""
    a, b_qkv, b_z = params
    yield f"{GDN}.in_proj_qkv.lora_A.weight", a.data.clone()
    yield f"{GDN}.in_proj_qkv.lora_B.weight", b_qkv.data.clone()
    yield f"{GDN}.in_proj_z.lora_A.weight", a.data.clone()
    yield f"{GDN}.in_proj_z.lora_B.weight", b_z.data.clone()


def _probe_gdn(rank, world_size, port):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    params = [torch.nn.Parameter(torch.randn(4, 6)), torch.nn.Parameter(torch.randn(7, 4))]
    params.append(torch.nn.Parameter(torch.randn(3, 4)))
    mapping = probe_bridge_row_mapping(params, lambda: _gdn_export(params), GDN_TARGETS)
    group = tuple(f"{GDN}.in_proj_{m}" for m in "kqvz")
    assert mapping.shared_lora_a_groups == (group,)
    pieces = {p.key: p for p in mapping.pieces}
    assert set(pieces) == {f"{group[0]}.lora_A"} | {f"{m}.lora_B" for m in group}
    flat = torch.cat([p.data.reshape(-1) for p in params])
    assert torch.equal(flat[pieces[f"{group[0]}.lora_A"].local_elements], params[0].data)
    assert torch.equal(flat[pieces[f"{GDN}.in_proj_k.lora_B"].local_elements], params[1].data[2:4])
    dist.destroy_process_group()


def test_probe_stores_a_fused_gdn_modules_shared_a_under_the_sorted_first_member():
    run_multiprocess(_probe_gdn)


def _export_fails_on_rank_1(params):
    yield f"{Q}.lora_A.weight", params[0].data.clone()
    if dist.get_rank() == 1:
        raise ValueError("rank 1 cannot export")


def _probe_export_fails(rank, world_size, port):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    params = [torch.nn.Parameter(torch.randn(3, 5))]
    with pytest.raises(RuntimeError, match="adapter checkpoint failed: rank 1: ValueError: rank 1 cannot export"):
        probe_bridge_row_mapping(params, lambda: _export_fails_on_rank_1(params), TARGETS)
    dist.destroy_process_group()


def test_an_export_failure_on_one_rank_fails_every_rank():
    """The fake export has no collectives; a rank failing inside one of Bridge's still hangs the others."""
    run_multiprocess(_probe_export_fails)
