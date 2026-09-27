"""Temporary real-CUDA qualification for EP-local routed expert quantization.

Run on eight Blackwell GPUs with the Miles CUDA 13 environment::

    torchrun --standalone --nproc-per-node=8 tests/manual/test_ep_local_expert_quantization.py

The quantizer and collectives are real. Only model parameter enumeration is
replaced with deterministic synthetic weights, avoiding a checkpoint download.
"""

import json
import os
import time
from argparse import Namespace
from collections import Counter
from datetime import timedelta
from types import SimpleNamespace

NVFP4_ENV = {
    "NVTE_NVFP4_4OVER6": "all",
    "NVTE_NVFP4_4OVER6_E4M3_USE_256": "none",
    "NVTE_NVFP4_4OVER6_ERR_MODE": "MSE",
    "NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH": "1",
    "NVTE_USE_FAST_MATH": "0",
    "NVTE_NVFP4_DISABLE_2D_QUANTIZATION": "1",
    "NVTE_NVFP4_DISABLE_RHT": "1",
    "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING": "1",
}
os.environ.update(NVFP4_ENV)

import torch
import torch.distributed as dist

import miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct as direct
from miles.backends.training_utils.parallel import set_parallel_state
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.utils.distributed_utils import get_gloo_group, init_gloo_group

MODEL_NAME = "glmmoedsa"
HIDDEN_SIZE = 64
EXPERT_INTERMEDIATE_SIZE = 48


def _group_info(ranks):
    group = dist.new_group(ranks=ranks, backend="nccl")
    if dist.get_rank() not in ranks:
        return None
    return SimpleNamespace(rank=ranks.index(dist.get_rank()), size=len(ranks), group=group)


def _create_parallel_state(ep_size, edp_size, pp_size):
    """Use expert rank order EP, EDP, PP and ordinary TP2 within each PP stage."""
    ranks_per_pp = ep_size * edp_size
    assert ranks_per_pp * pp_size == dist.get_world_size() == 8
    selected = {}
    group_lists = {
        "ep": [
            [pp * ranks_per_pp + edp * ep_size + ep for ep in range(ep_size)]
            for pp in range(pp_size)
            for edp in range(edp_size)
        ],
        "edp": [
            [pp * ranks_per_pp + edp * ep_size + ep for edp in range(edp_size)]
            for pp in range(pp_size)
            for ep in range(ep_size)
        ],
        "pp": [[pp * ranks_per_pp + lane for pp in range(pp_size)] for lane in range(ranks_per_pp)],
        "tp_dp_cp": [list(range(pp * ranks_per_pp, (pp + 1) * ranks_per_pp)) for pp in range(pp_size)],
        "tp": [list(range(start, start + 2)) for start in range(0, 8, 2)],
        "etp": [[rank] for rank in range(8)],
    }
    for kind, rank_lists in group_lists.items():
        for ranks in rank_lists:
            info = _group_info(ranks)
            if info is not None:
                selected[kind] = info
    state = SimpleNamespace(**selected)
    ep_groups_by_rank = {rank: ranks for ranks in group_lists["ep"] for rank in ranks}
    staged_sources = [
        source
        for edp_source in dist.get_process_group_ranks(state.edp.group)
        for source in ep_groups_by_rank[edp_source]
    ]
    assert dist.get_process_group_ranks(state.tp_dp_cp.group) == staged_sources
    set_parallel_state(state)
    return state


def _args(num_layers, num_experts):
    return Namespace(
        sglang_speculative_algorithm=None,
        custom_model_provider_path=None,
        mtp_num_layers=None,
        num_layers=num_layers,
        num_experts=num_experts,
        q_lora_rank=None,
        vocab_size=64,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=4,
        num_query_groups=4,
        kv_channels=16,
        swiglu=True,
        update_weight_buffer_size=20000,
    )


def _weight(name, shape):
    seed = sum((i + 1) * ord(char) for i, char in enumerate(name))
    values = torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape)
    values = ((values * (seed % 7 + 1)) % 97 - 48) / 17 + (seed % 19 - 9) / 13
    if "linear_fc1" in name:
        values[shape[0] // 2 :] *= 7
    return values.to(torch.bfloat16)


def _full_weights(num_layers, num_experts):
    weights = {}
    for layer in range(num_layers):
        prefix = f"module.module.decoder.layers.{layer}"
        for expert in range(num_experts):
            for projection, shape in (
                ("linear_fc1", (2 * EXPERT_INTERMEDIATE_SIZE, HIDDEN_SIZE)),
                ("linear_fc2", (HIDDEN_SIZE, EXPERT_INTERMEDIATE_SIZE)),
            ):
                name = f"{prefix}.mlp.experts.{projection}.weight{expert}"
                weights[name] = _weight(name, shape)
        for suffix, shape in (
            ("mlp.shared_experts.linear_fc1.weight", (64, HIDDEN_SIZE)),
            ("mlp.shared_experts.linear_fc2.weight", (HIDDEN_SIZE, 32)),
            ("input_layernorm.weight", (HIDDEN_SIZE,)),
        ):
            name = f"{prefix}.{suffix}"
            weights[name] = _weight(name, shape)
    return weights


def _local_weights(full_weights, state, local_experts):
    local = {}
    layer_prefix = f"module.module.decoder.layers.{state.pp.rank}."
    for name, full in full_weights.items():
        if not name.startswith(layer_prefix):
            continue
        if direct.is_routed_expert_param(name):
            expert = int(name.rsplit("weight", 1)[1])
            if expert // local_experts != state.ep.rank:
                continue
            weight = full.clone()
        elif "shared_experts.linear_fc1" in name:
            gate, up = full.chunk(2, dim=0)
            weight = torch.cat([gate.chunk(2, 0)[state.tp.rank], up.chunk(2, 0)[state.tp.rank]])
            weight.tensor_model_parallel = True
            weight.partition_dim = 0
            weight.partition_stride = 2
        elif "shared_experts.linear_fc2" in name:
            weight = full.chunk(2, dim=1)[state.tp.rank].contiguous()
            weight.tensor_model_parallel = True
            weight.partition_dim = 1
            weight.partition_stride = 1
        else:
            weight = full.clone()
        local[name] = weight
    return local


def _raw_bytes(tensor):
    return tensor.contiguous().reshape(-1).view(torch.uint8).cpu()


def _verify_output(buckets, expected):
    actual = {}
    bucket_index = {}
    for index, bucket in enumerate(buckets):
        for name, tensor in bucket:
            assert name not in actual, f"Duplicate output: {name}"
            actual[name] = tensor
            bucket_index[name] = index
    assert actual.keys() == expected.keys(), (
        sorted(actual.keys() - expected.keys()),
        sorted(expected.keys() - actual.keys()),
    )
    for name, tensor in actual.items():
        reference = expected[name]
        assert tensor.dtype == reference.dtype and tensor.shape == reference.shape, name
        assert torch.equal(_raw_bytes(tensor), _raw_bytes(reference)), f"Byte mismatch: {name}"
        if ".experts." in name and name.endswith(".gate_proj.weight"):
            base = name.removesuffix(".gate_proj.weight")
            suffixes = ("weight",)
            if f"{base}.gate_proj.weight_scale_2" in actual:
                gate_scale = actual[f"{base}.gate_proj.weight_scale_2"]
                up_scale = actual[f"{base}.up_proj.weight_scale_2"]
                assert torch.equal(_raw_bytes(gate_scale), _raw_bytes(up_scale)), name
                suffixes += ("weight_scale", "weight_scale_2")
            paired_names = [f"{base}.{role}.{suffix}" for role in ("gate_proj", "up_proj") for suffix in suffixes]
            assert len({bucket_index[item] for item in paired_names}) == 1, base
    return len(actual), sum(tensor.nbytes for tensor in actual.values())


def _expected_edp_owners(local_experts, edp_size):
    """Independent balanced contiguous partition, with larger chunks first."""
    chunk_size, extra = divmod(local_experts, edp_size)
    return [owner for owner in range(edp_size) for _ in range(chunk_size + (owner < extra))]


def _verify_calls(all_calls, full_weights, state, *, local_experts, gather_pp, sender_only):
    totals = Counter()
    ranks_per_pp = state.ep.size * state.edp.size
    expected_owners = _expected_edp_owners(local_experts, state.edp.size)
    expert_calls_per_rank = []
    for rank, calls in enumerate(all_calls):
        totals.update(calls)
        rank_pp = rank // ranks_per_pp
        rank_ep = rank % state.ep.size
        rank_edp = (rank // state.ep.size) % state.edp.size
        expert_calls_per_rank.append(
            sum(count for name, count in calls.items() if direct.is_routed_expert_param(name))
        )
        for name in calls:
            if direct.is_routed_expert_param(name):
                expert = int(name.rsplit("weight", 1)[1])
                layer = int(name.split(".layers.")[1].split(".")[0])
                assert rank_edp == expected_owners[expert % local_experts], (rank, name, expected_owners)
                assert rank_pp == layer and rank_ep == expert // local_experts, (rank, name)
        assert expert_calls_per_rank[-1] == 2 * expected_owners.count(rank_edp), (rank, calls, expected_owners)
    for name in full_weights:
        expected = 1 if direct.is_routed_expert_param(name) or sender_only else (8 if gather_pp else ranks_per_pp)
        assert totals[name] == expected, (name, totals[name], expected)
    if local_experts >= state.edp.size:
        assert all(expert_calls_per_rank), expert_calls_per_rank
    else:
        assert sum(bool(count) for count in expert_calls_per_rank) == state.pp.size * state.ep.size * local_experts
    return {
        "expert_conversions": sum(count for name, count in totals.items() if direct.is_routed_expert_param(name)),
        "other_conversions": sum(count for name, count in totals.items() if not direct.is_routed_expert_param(name)),
        "expert_conversions_per_rank": expert_calls_per_rank,
        "local_expert_edp_owners": expected_owners,
    }


def _verify_rounds(all_rounds, state, local_experts):
    assert len({len(rounds) for rounds in all_rounds}) == 1, [len(rounds) for rounds in all_rounds]
    counts_per_edp = []
    for index in range(len(all_rounds[0])):
        counts = [0] * state.edp.size
        for rank, rounds in enumerate(all_rounds):
            counts[(rank // state.ep.size) % state.edp.size] += rounds[index]
        if local_experts % state.edp.size == 0:
            # Balanced replicas must contribute within the same round, not in
            # separate blocks of rounds that serialize their quantization.
            assert min(counts) > 0 and len(set(counts)) == 1, (index, counts)
        counts_per_edp.append(counts)
    return counts_per_edp


def _run_case(state, *, local_experts=2, gather_pp, sender_only, quantized=True):
    args = _args(state.pp.size, state.ep.size * local_experts)
    full_weights = _full_weights(args.num_layers, args.num_experts)
    weights = _local_weights(full_weights, state, local_experts)
    model = torch.nn.Module()
    model.config = SimpleNamespace(mtp_num_layers=None)
    model.synthetic_weights = weights
    quantization = (
        {"quant_method": "nvfp4", "ignore": ["model.layers.0.mlp.experts.0.down_proj"]} if quantized else None
    )
    materialize = not sender_only or (dist.get_rank() == 0 if gather_pp else state.ep.rank == state.edp.rank == 0)
    calls = Counter()
    round_calls = []
    active_round = None
    original_convert = direct.convert_to_hf
    original_enumeration = direct.named_params_and_buffers
    original_batch = direct.HfWeightIteratorDirect._materialize_expert_batch
    original_metadata_exchange = dist.all_gather_object
    metadata_exchanges = 0
    original_gather_payloads = direct.ExpertGather._gather_payloads
    original_collectives = {
        name: getattr(dist, name) for name in ("broadcast", "all_gather", "all_gather_into_tensor")
    }
    original_new_group = dist.new_group
    payload_phase = False
    payload_collectives = Counter()
    payload_stages = Counter()
    expected_groups = ([state.pp] if gather_pp else []) + [state.tp_dp_cp]
    expected_groups = tuple(group.group for group in expected_groups if group.size > 1)
    stage_names = {state.pp.group: "pp", state.tp_dp_cp.group: "tp_dp_cp"}

    def tracked_convert(args, model_name, name, param, quantization_config=None, packed_weight_basenames=None):
        calls[name] += 1
        if active_round is not None:
            round_calls[active_round] += 1
        return original_convert(args, model_name, name, param, quantization_config, packed_weight_basenames)

    def tracked_batch(iterator, batch, local_weights):
        nonlocal active_round
        active_round = len(round_calls)
        round_calls.append(0)
        try:
            return original_batch(iterator, batch, local_weights)
        finally:
            active_round = None

    def tracked_metadata_exchange(*args, **kwargs):
        nonlocal metadata_exchanges
        metadata_exchanges += 1
        return original_metadata_exchange(*args, **kwargs)

    def tracked_gather_payloads(gather, storage, payloads, local_payload):
        nonlocal payload_phase
        payload_stages[stage_names[gather._group]] += 1
        previous_collectives = payload_collectives.total()
        payload_phase = True
        try:
            handle = original_gather_payloads(gather, storage, payloads, local_payload)
            assert payload_collectives.total() - previous_collectives == bool(storage.numel())
            return handle
        finally:
            payload_phase = False

    def tracked_collective(name):
        def call(*args, **kwargs):
            if payload_phase:
                payload_collectives[name] += 1
            return original_collectives[name](*args, **kwargs)

        return call

    def reject_new_group(*args, **kwargs):
        raise AssertionError("Weight updates must reuse existing process groups")

    direct.convert_to_hf = tracked_convert
    direct.named_params_and_buffers = lambda _args, modules: iter(modules[0].synthetic_weights.items())
    direct.HfWeightIteratorDirect._materialize_expert_batch = tracked_batch
    dist.new_group = reject_new_group
    try:
        iterator = direct.HfWeightIteratorDirect(
            args,
            [model],
            placement=WeightUpdatePlacement(gather_pp=gather_pp),
            model_name=MODEL_NAME,
            quantization_config=quantization,
        )
        for batch in iterator._expert_batches:
            assert tuple(gather._group for gather in batch.gathers) == expected_groups
        expected_stages = Counter(
            stage_names[group] for _batch in iterator._expert_batches for group in expected_groups
        )
        dist.all_gather_object = tracked_metadata_exchange
        direct.ExpertGather._gather_payloads = tracked_gather_payloads
        for name in original_collectives:
            setattr(dist, name, tracked_collective(name))
        dist.barrier()
        started = time.monotonic()
        initial_buckets = list(iterator.iter_hf_weights(weights, materialize=materialize))
        initial_exchanges = metadata_exchanges
        initial_payload_collectives = payload_collectives.copy()
        initial_payload_stages = payload_stages.copy()
        assert initial_exchanges == expected_stages.total()
        assert initial_payload_stages == expected_stages
        initial_calls = calls.copy()
        calls.clear()
        round_calls.clear()
        metadata_exchanges = 0
        payload_collectives.clear()
        payload_stages.clear()
        updated_weights = {name: tensor * -1.25 for name, tensor in weights.items()}
        buckets = list(iterator.iter_hf_weights(updated_weights, materialize=materialize))
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        assert metadata_exchanges == 0, "Stable expert layouts must not be exchanged on later updates"
        assert calls == initial_calls, (calls, initial_calls)
        assert payload_collectives == initial_payload_collectives
        assert payload_stages == expected_stages
    finally:
        direct.convert_to_hf = original_convert
        direct.named_params_and_buffers = original_enumeration
        direct.HfWeightIteratorDirect._materialize_expert_batch = original_batch
        dist.all_gather_object = original_metadata_exchange
        direct.ExpertGather._gather_payloads = original_gather_payloads
        for name, collective in original_collectives.items():
            setattr(dist, name, collective)
        dist.new_group = original_new_group

    # Keeping the first update alive checks that cached layouts never reuse its storage.
    for outputs, multiplier in ((initial_buckets, 1.0), (buckets, -1.25)):
        expected = {}
        if materialize:
            for name, tensor in full_weights.items():
                if not gather_pp and not name.startswith(f"module.module.decoder.layers.{state.pp.rank}."):
                    continue
                expected.update(original_convert(args, MODEL_NAME, name, (tensor * multiplier).cuda(), quantization))
        tensor_count, nbytes = _verify_output(outputs, expected)
    all_calls = [None] * dist.get_world_size()
    dist.all_gather_object(all_calls, calls, group=get_gloo_group())
    counts = _verify_calls(
        all_calls, full_weights, state, local_experts=local_experts, gather_pp=gather_pp, sender_only=sender_only
    )
    all_rounds = [None] * dist.get_world_size()
    dist.all_gather_object(all_rounds, round_calls, group=get_gloo_group())
    round_counts = _verify_rounds(all_rounds, state, local_experts)
    measurements = [None] * dist.get_world_size()
    dist.all_gather_object(measurements, (elapsed, tensor_count, nbytes), group=get_gloo_group())
    return {
        "ep": state.ep.size,
        "edp": state.edp.size,
        "pp": state.pp.size,
        "tp": 2,
        "quantized": quantized,
        "local_experts": local_experts,
        "round_expert_conversions_per_edp": round_counts,
        "gather_pp": gather_pp,
        "sender_only": sender_only,
        "updates": 2,
        **counts,
        "max_iterator_seconds": max(item[0] for item in measurements),
        "output_tensors_per_rank": [item[1] for item in measurements],
        "output_bytes_per_rank": [item[2] for item in measurements],
        "bytewise_oracle": "passed",
        "owner_counts": "passed",
        "gate_up_atomicity": "passed",
        "cached_layout_reuse": "passed",
        "previous_update_storage": "passed",
        "metadata_exchanges_per_update": [initial_exchanges, metadata_exchanges],
        "payload_collectives_per_update": [dict(initial_payload_collectives), dict(payload_collectives)],
        "payload_stages_per_update": [dict(initial_payload_stages), dict(payload_stages)],
        "new_process_groups_during_construction": 0,
        "new_process_groups_per_update": [0, 0],
    }


def _verify_etp_guard(state):
    args = _args(state.pp.size, state.ep.size * 2)
    weights = _local_weights(_full_weights(args.num_layers, args.num_experts), state, 2)
    model = torch.nn.Module()
    model.config = SimpleNamespace(mtp_num_layers=None)
    original_enumeration = direct.named_params_and_buffers
    original_etp = state.etp
    state.etp = SimpleNamespace(size=2, rank=0, group=None)
    direct.named_params_and_buffers = lambda _args, _model: iter(weights.items())
    try:
        kwargs = dict(placement=WeightUpdatePlacement(gather_pp=False), model_name=MODEL_NAME)
        for quantization_config in ({"quant_method": "nvfp4"}, None):
            try:
                direct.HfWeightIteratorDirect(args, [model], quantization_config=quantization_config, **kwargs)
            except ValueError as error:
                assert "expert-tensor-parallel-size 1" in str(error), str(error)
            else:
                raise AssertionError(f"Routed-expert ETP2 was accepted with {quantization_config=}")
    finally:
        state.etp = original_etp
        direct.named_params_and_buffers = original_enumeration


def _verify_payload_layouts():
    def units(rank, revision, mode):
        if mode == "empty" or (mode == "ragged" and rank % 2) or (mode == "single" and rank != 7):
            return []
        count = rank + 1 if mode == "ragged" else 1
        return [
            [
                (f"{rank}.fp64", torch.arange(count, dtype=torch.float64, device="cuda") + rank + revision),
                (f"{rank}.scalar", torch.tensor(rank + revision, dtype=torch.int64, device="cuda")),
                (f"{rank}.odd_tail", torch.tensor([1, 2, 3], dtype=torch.uint8, device="cuda") + revision),
            ]
        ]

    for mode in ("uniform", "ragged", "single", "empty"):
        gather = direct.ExpertGather(group=dist.group.WORLD)
        first = gather(units(dist.get_rank(), 0, mode), device="cuda")
        second = gather(units(dist.get_rank(), 7, mode), device="cuda")
        for actual, revision in ((first, 0), (second, 7)):
            expected = dict(item for rank in range(8) for unit in units(rank, revision, mode) for item in unit)
            _verify_output(actual, expected)


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    init_gloo_group()
    assert dist.get_world_size() == 8, "This qualification requires exactly eight ranks"
    _verify_payload_layouts()
    results = []
    for ep, edp, pp in ((4, 2, 1), (2, 2, 2), (1, 2, 4), (2, 4, 1), (1, 8, 1)):
        state = _create_parallel_state(ep, edp, pp)
        cases = [(2, False, True, True), (2, True, False, True), (2, True, True, True), (2, False, False, True)]
        if (ep, edp, pp) == (2, 2, 2):
            cases += [(3, False, True, True), (3, True, False, True), (2, True, False, False)]
        if edp == 4:
            cases = [(1, False, True, True), (1, True, False, True)]
        if edp == 8:
            # Match 256 experts / EP64 / EDP8 locally: four complete experts
            # shared by eight replicas, without claiming a 512-GPU run.
            cases = [(4, True, True, True)]
        for local_experts, gather_pp, sender_only, quantized in cases:
            result = _run_case(
                state, local_experts=local_experts, gather_pp=gather_pp, sender_only=sender_only, quantized=quantized
            )
            if edp == 8:
                assert result["expert_conversions_per_rank"] == [2, 2, 2, 2, 0, 0, 0, 0], result
                assert result["local_expert_edp_owners"] == [0, 1, 2, 3], result
            results.append(result)
            if dist.get_rank() == 0:
                print("EP_LOCAL_CASE " + json.dumps(result, sort_keys=True), flush=True)
        _verify_etp_guard(state)
        dist.barrier()
    if dist.get_rank() == 0:
        print(
            "EP_LOCAL_SUMMARY "
            + json.dumps(
                {
                    "status": "passed",
                    "cases": len(results),
                    "environment": NVFP4_ENV,
                    "etp2_rejected": True,
                    "unquantized_etp2_rejected": True,
                    "unquantized_etp1_bytewise_oracle": "passed",
                    "edp_work_sharing": "passed",
                    "uneven_local_expert_partition": "passed",
                    "more_edp_ranks_than_local_experts": "passed",
                    "aligned_slab_uniform_ragged_single_empty": "passed",
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(),
                    "results": results,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
