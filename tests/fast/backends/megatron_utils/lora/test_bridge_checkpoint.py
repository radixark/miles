from dataclasses import replace
from types import SimpleNamespace

"""The row-mapping probe recovers each rank's HF pieces, primaries and shared A from an export alone."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.megatron_utils.lora.bridge_checkpoint import (
    BridgeAdapterSlots,
    BridgeRowMapping,
    _HfPiece,
    probe_bridge_row_mapping,
)
from miles.backends.training_utils.checkpoint.adapter.adapter_dir import HeadRows
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


def _model(params):
    """The probe swaps parameters where modules register them."""
    holder = torch.nn.Module()
    for index, param in enumerate(params):
        holder.register_parameter(f"p{index}", param)
    return [holder]


def _registered(model):
    """What an export reads: the probe's stand-ins while it runs."""
    return list(model[0].parameters())


def _probe(rank, world_size, port):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    params = [
        torch.nn.Parameter(torch.randn(3, 5)),
        torch.nn.Parameter(torch.randn(2, 3)),
        torch.nn.Parameter(torch.randn(2, 3)),
    ]
    model = _model(params)
    before = [p.data.clone() for p in params]
    # Megatron DDP fills main_grad from a hook on each parameter's grad accumulator, which it keeps alive
    hooked, accumulators = [], [param.expand_as(param).grad_fn.next_functions[0][0] for param in params]
    for accumulator in accumulators:
        accumulator.register_hook(lambda *_: hooked.append(1))
    mapping = probe_bridge_row_mapping(model, params, lambda: _export(_registered(model)), TARGETS)
    sum(p.sum() for p in params).backward()
    assert len(hooked) == len(params), "the probe keeps each parameter's grad accumulator"

    assert mapping.shared_lora_a_groups == ((K, Q),)
    assert all(torch.equal(p.data, b) for p, b in zip(params, before, strict=True)), "the probe restores the params"
    pieces = {p.key: p for p in mapping.pieces}
    assert set(pieces) == {f"{K}.lora_A", f"{Q}.lora_B", f"{K}.lora_B"}, "a shared A is stored once"
    assert pieces[f"{Q}.lora_B"].offsets == (2 * rank, 0) and pieces[f"{Q}.lora_B"].global_shape == (4, 3)
    flat = torch.cat([p.data.reshape(-1) for p in params])
    assert torch.equal(flat[pieces[f"{K}.lora_B"].local_elements], params[2].data)
    assert torch.equal(flat[pieces[f"{K}.lora_A"].local_elements], params[0].data)
    # one index per local element stays resident; int64 would double it
    assert all(p.local_elements.dtype == torch.int32 for p in mapping.pieces)
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
    model = _model(params)
    mapping = probe_bridge_row_mapping(model, params, lambda: _expert_export(_registered(model)), TARGETS)
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
    init_gloo_group()
    # vocab 7 padded to 8: rank 1's last row of B is padding
    params = [torch.nn.Parameter(torch.randn(2, 5)), torch.nn.Parameter(torch.randn(4, 2))]
    model = _model(params)
    padding = torch.arange(10 + 3 * 2, 10 + 4 * 2) if rank == 1 else torch.empty(0, dtype=torch.long)
    with pytest.raises(RuntimeError, match="rank 1: AssertionError: .*does not reach 2 of this rank's elements"):
        probe_bridge_row_mapping(model, params, lambda: _trimmed_head_export(_registered(model), 7), TARGETS)

    mapping = probe_bridge_row_mapping(
        model, params, lambda: _trimmed_head_export(_registered(model), 7), TARGETS, unexported=padding
    )

    head_b = next(p for p in mapping.pieces if p.key == f"{HEAD}.lora_B")
    assert head_b.global_shape == (7, 2) and head_b.offsets == (4 * rank, 0)
    flat = torch.cat([p.data.reshape(-1) for p in params])
    assert torch.equal(flat[head_b.local_elements], params[1].data[: 4 - rank])
    with pytest.raises(RuntimeError, match="the export reaches elements declared unexported"):
        probe_bridge_row_mapping(
            model, params, lambda: _trimmed_head_export(_registered(model), 8), TARGETS, unexported=padding
        )
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
    init_gloo_group()
    params = [torch.nn.Parameter(torch.randn(4, 3))]
    model = _model(params)
    mapping = probe_bridge_row_mapping(model, params, lambda: _fused_gate_up_export(_registered(model)), TARGETS)

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
    model = _model(params)
    mapping = probe_bridge_row_mapping(model, params, lambda: _gdn_export(_registered(model)), GDN_TARGETS)
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
    model = _model(params)
    with pytest.raises(RuntimeError, match="adapter checkpoint failed: rank 1: ValueError: rank 1 cannot export"):
        probe_bridge_row_mapping(model, params, lambda: _export_fails_on_rank_1(_registered(model)), TARGETS)
    dist.destroy_process_group()


def test_an_export_failure_on_one_rank_fails_every_rank():
    """The fake export has no collectives; a rank failing inside one of Bridge's still hangs the others."""
    run_multiprocess(_probe_export_fails)


def _slots_of(pieces, real_rank):
    """Multi-LoRA slots whose probe found ``pieces``, padded to max rank 4; the slot's real rank is ``real_rank``."""
    slots = BridgeAdapterSlots.__new__(BridgeAdapterSlots)
    slots.row_mapping, slots._specs = BridgeRowMapping(pieces, (), ()), {}
    slots.lora_rank = lambda slot: real_rank
    return slots


def _trim_a_tp_split_rank(rank, world_size, port):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    # max rank 4 split over 2 TP ranks: each rank holds 2 rows of A ([rank, in]) and 2 columns of B ([out, rank])
    a = _HfPiece(f"{Q}.lora_A", (4, 3), (2 * rank, 0), torch.arange(6).view(2, 3))
    b = _HfPiece(f"{Q}.lora_B", (5, 4), (0, 2 * rank), torch.arange(6, 16).view(5, 2))
    flat = torch.arange(16.0)

    slots = _slots_of([a, b], real_rank=3)
    specs = [(s.key, s.global_shape, s.offsets, s.local_shape) for s in slots.get_local_shard_specs(1)]
    kept = 2 if rank == 0 else 1
    assert specs == [(a.key, (3, 3), (2 * rank, 0), (kept, 3)), (b.key, (5, 3), (0, 2 * rank), (5, kept))]
    piece_a, piece_b = slots.local_pieces(1, [flat])
    assert torch.equal(piece_a, flat[a.local_elements[:kept]]) and torch.equal(
        piece_b, flat[b.local_elements[:, :kept]]
    )

    # real rank 2: rank 1's rows and columns are all padding, so it holds no piece of the slot
    slots = _slots_of([a, b], real_rank=2)
    assert (slots.get_local_shard_specs(1) == []) == (rank == 1)
    assert (slots.local_pieces(1, [flat]) == []) == (rank == 1)
    dist.destroy_process_group()


def test_a_multi_lora_slot_keeps_only_its_real_rank_of_a_tp_split_rank_axis():
    run_multiprocess(_trim_a_tp_split_rank)


def _slots_with_pieces(keys_and_rows, vocab_size):
    slots = BridgeAdapterSlots.__new__(BridgeAdapterSlots)
    slots.targets = replace(TARGETS, vocab_size=vocab_size)
    pieces = [SimpleNamespace(key=key, global_shape=(rows, 2)) for key, rows in keys_and_rows]
    slots.row_mapping = SimpleNamespace(pieces=pieces)
    return slots


def test_a_trained_head_declares_its_served_rows_and_the_hf_vocab_a_source_may_hold():
    head_b = f"{TARGETS.unembed[0]}.lora_B"
    assert _slots_with_pieces([(head_b, 6)], vocab_size=8)._head_rows() == HeadRows(head_b, max_source_rows=8)
    assert _slots_with_pieces([(f"{K}.lora_B", 4)], vocab_size=8)._head_rows() is None
