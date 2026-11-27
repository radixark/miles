"""Single-adapter multi-LoRA preserves single-LoRA microbatches across supported layouts."""

from argparse import Namespace
from contextlib import nullcontext
from copy import deepcopy
from unittest.mock import Mock

import pytest
import torch

from miles.backends.megatron_utils.lora import model as lora_model
from miles.backends.training_utils.data import rollout as data_utils
from miles.backends.training_utils.parallel import GroupInfo, ParallelState
from miles.ray.rollout.train_data_conversion import process_rollout_data_shard, split_train_data_by_dp
from miles.tinker.runtime import MilesBackend, _build_train_data
from miles.utils import dp_schedule
from miles.utils.flops_utils import KeyTerm, LayerFlops, ModelFlops


def _parallel_state(dp_size=1, dp_rank=0, cp_size=1):
    trivial = GroupInfo(rank=0, size=1, group=None)
    dp = GroupInfo(rank=dp_rank, size=dp_size, group=None)
    return ParallelState(
        intra_dp=dp,
        intra_dp_cp=dp,
        cp=GroupInfo(rank=0, size=cp_size, group=None),
        tp=trivial,
        pp=trivial,
        ep=trivial,
        etp=trivial,
        indep_dp=trivial,
        vpp_size=1,
        microbatch_group_size_per_vp_stage=None,
    )


def _args(*, dynamic=True, micro_batch_size=1, balance="none"):
    return Namespace(
        qkv_format="thd",
        multimodal_token_space_processing=False,
        global_batch_size=64,
        micro_batch_size=micro_batch_size,
        backend=Namespace(global_batch_size=64, micro_batch_size=micro_batch_size, deterministic_mode=False),
        use_dynamic_global_batch_size=True,
        use_dynamic_batch_size=dynamic,
        max_tokens_per_gpu=10,
        balance_data=balance == "data",
        balance_by_flops=balance == "flops",
        allow_partial_train_step=False,
        enable_sample_ownership_checker=False,
        hf_checkpoint="unused",
        multi_lora_n_adapters=4,
    )


def _slot_datums(lengths, slots=None):
    return [
        (
            3 if slots is None else slots[index],
            {"tokens": [index] * length, "target_tokens": [index] * (length - 1), "target_len": length - 1},
        )
        for index, length in enumerate(lengths)
    ]


def _single_lora_batch(batch):
    single = {key: value for key, value in batch.items() if key not in ("adapter_slots", "loss_fn", "loss_fn_config")}
    return dict(single, rollout_ids=list(range(len(batch["tokens"]))))


def _shards(args, batch, dp_size, cp_size=1):
    config = _parallel_state(dp_size=dp_size, cp_size=cp_size).train_parallel_config(
        supports_precomputed_schedule=True
    )
    return [process_rollout_data_shard(args, shard) for shard in split_train_data_by_dp(args, batch, config)]


def _sample_order(args, shard):
    iterators, counts = data_utils.get_data_iterator(args, model=None, rollout_data=shard)
    order = [iterators[0].get_next(["sample_indices"])["sample_indices"] for _ in range(sum(counts))]
    return order, counts


@pytest.fixture
def trainer_stubs(monkeypatch):
    flops = ModelFlops(layers=(LayerFlops(10.0, (KeyTerm(1.0),)),), lm_head=0.0)
    monkeypatch.setattr(dp_schedule, "_model_flops", lambda checkpoint: flops)
    monkeypatch.setattr(data_utils.current_platform, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(lora_model, "setup_train_iteration_config", Mock())
    monkeypatch.setattr(lora_model, "DumperMegatronUtil", Mock())
    monkeypatch.setattr(lora_model, "accumulate_gradients", lambda model: nullcontext())
    monkeypatch.setattr(lora_model, "aggregate_train_losses", lambda *args: {})


@pytest.mark.parametrize("dp_size", [1, 2, 4])
@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("balance", ["none", "data", "flops"])
@pytest.mark.parametrize("dynamic,micro_batch_size", [(True, 1), (False, 1), (False, 2)])
@pytest.mark.parametrize("forward_only", [False, True])
async def test_runtime_to_forward_pass_matches_single_lora(
    monkeypatch, trainer_stubs, dp_size, cp_size, balance, dynamic, micro_batch_size, forward_only
):
    """Run the gateway's real padding/translation/split path and compare every trainer rank."""
    args = _args(dynamic=dynamic, micro_batch_size=micro_batch_size, balance=balance)
    lengths = [3, 3, 6, 6, 2, 2, 11, 4, 8, 4, 2, 2, 3, 3, 5, 5]
    # Uneven sample counts expose gratuitous DP padding. Static batches also
    # exercise the final partial microbatch that the single-LoRA scheduler accepts.
    if dynamic or micro_batch_size == 2:
        lengths = lengths[:-1]
    lengths = [length * cp_size for length in lengths]
    datums = _slot_datums(lengths)
    native = _shards(args, _single_lora_batch(_build_train_data(datums)), dp_size, cp_size)
    backend = MilesBackend(None, "unused", lora_layout=None, dp_size=dp_size)

    async def call_trainer(method, batch_id, batch):
        assert len(batch["tokens"]) == len(datums), "padding must not alter a schedulable batch"
        multi = _shards(args, batch, dp_size, cp_size)
        outputs = []
        for rank, (expected_shard, actual_shard) in enumerate(zip(native, multi, strict=True)):
            parallel = _parallel_state(dp_size, rank, cp_size)
            monkeypatch.setattr(data_utils, "get_parallel_state", lambda parallel=parallel: parallel)
            monkeypatch.setattr(lora_model, "get_parallel_state", lambda parallel=parallel: parallel)
            expected, counts = _sample_order(args, expected_shard)
            assert actual_shard["num_microbatches"] == counts
            assert "num_rollouts" not in actual_shard, "Tinker does not use rollout loss normalization"
            actual = []

            def run_pass(args, dumper, iterators, model, count, *, actual=actual, **kwargs):
                assert kwargs["forward_only"] == forward_only
                assert kwargs["num_rollouts"] is None
                actual.extend(iterators[0].get_next(["sample_indices"])["sample_indices"] for _ in range(count))
                return [
                    {"per_datum": [{"sample_index": i, "loss": 0.0, "logprobs": torch.tensor([0.0])} for i in mb]}
                    for mb in actual
                ]

            monkeypatch.setattr(lora_model, "run_forward_backward_pass", run_pass)
            outputs.append(
                lora_model.run_forward_backward(
                    args, batch_id, [Mock()], Mock(), actual_shard, forward_only=forward_only
                )
            )
            assert actual == expected
        return outputs

    monkeypatch.setattr(backend, "_call_trainer", call_trainer)
    forward = backend.forward_only if forward_only else backend.forward_backward
    outputs = await forward(0, datums, "cross_entropy", {})
    assert len(outputs) == len(datums)


@pytest.mark.parametrize("dp_size", [1, 2, 4])
@pytest.mark.parametrize("dynamic", [False, True])
def test_mixed_adapters_sort_within_microbatches_without_changing_membership(
    monkeypatch, trainer_stubs, dp_size, dynamic
):
    args = _args(dynamic=dynamic, micro_batch_size=2)
    slots = [2, 0, 1, 2, 0, 1, 3, 0] * 2
    batch = _build_train_data(_slot_datums([3, 3, 6, 6, 2, 5, 2, 4] * 2, slots))
    native = _shards(args, _single_lora_batch(batch), dp_size)
    multi = _shards(args, dict(batch, loss_fn="cross_entropy"), dp_size)
    for rank, (expected_shard, actual_shard) in enumerate(zip(native, multi, strict=True)):
        monkeypatch.setattr(data_utils, "get_parallel_state", lambda rank=rank: _parallel_state(dp_size, rank))
        expected, counts = _sample_order(args, expected_shard)
        original_schedule = deepcopy(actual_shard["micro_batch_indices"])
        actual, actual_counts = _sample_order(args, actual_shard)
        assert actual_counts == counts
        assert actual == [sorted(mb, key=lambda i: slots[i]) for mb in expected]
        assert actual_shard["micro_batch_indices"] == original_schedule


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("balance", ["none", "data", "flops"])
@pytest.mark.parametrize("media_key", ["multimodal_train_inputs", "multimodal_inputs"])
def test_multimodal_uses_the_same_local_fallback(monkeypatch, trainer_stubs, dynamic, balance, media_key):
    args = _args(dynamic=dynamic, balance=balance)
    batch = _build_train_data(_slot_datums([3, 5, 3, 5, 3, 5, 3, 5]))
    batch[media_key] = [None] * len(batch["tokens"])
    native = _shards(args, _single_lora_batch(batch), 2)
    multi = _shards(args, dict(batch, loss_fn="cross_entropy"), 2)
    for rank, (expected_shard, actual_shard) in enumerate(zip(native, multi, strict=True)):
        monkeypatch.setattr(data_utils, "get_parallel_state", lambda rank=rank: _parallel_state(2, rank))
        monkeypatch.setattr(data_utils.GeneralPGUtil, "create", lambda group: Mock())
        assert "micro_batch_indices" not in expected_shard and "micro_batch_indices" not in actual_shard
        assert _sample_order(args, actual_shard) == _sample_order(args, expected_shard)


def test_static_alignment_errors_match_single_lora():
    args = _args(dynamic=False, micro_batch_size=2)
    batch = _build_train_data(_slot_datums([3] * 5))
    with pytest.raises(AssertionError, match="static path") as single:
        _shards(args, _single_lora_batch(batch), 2)
    with pytest.raises(AssertionError, match="static path") as multi:
        _shards(args, dict(batch, loss_fn="cross_entropy"), 2)
    assert str(single.value) == str(multi.value)


def test_work_unit_keeps_all_datums_without_rollout_audit_metadata():
    args = _args()
    args.enable_sample_ownership_checker = True
    batch = _build_train_data(_slot_datums([3, 4, 5, 6, 7]))
    shards = _shards(args, dict(batch, loss_fn="cross_entropy"), 2)
    assert sorted(i for shard in shards for i in shard["sample_indices"]) == list(range(5))


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("balance", ["none", "data", "flops"])
def test_separate_work_units_match_grouped_rollout_step_boundaries(monkeypatch, trainer_stubs, dynamic, balance):
    args = _args(dynamic=dynamic, micro_batch_size=2, balance=balance)
    datums = _slot_datums([3, 3, 6, 6, 2, 2, 11, 4])
    single_batch = _single_lora_batch(_build_train_data(datums))
    single_batch.update(rollout_ids=[0, 0, 1, 2, 3, 3, 4, 5], dynamic_global_batch_size=3)
    single = _shards(args, single_batch, 2)
    work_units = [
        _shards(args, dict(_build_train_data(datums[start : start + 4]), loss_fn="cross_entropy"), 2)
        for start in (0, 4)
    ]
    for rank in range(2):
        monkeypatch.setattr(data_utils, "get_parallel_state", lambda rank=rank: _parallel_state(2, rank))
        expected, counts = _sample_order(args, single[rank])
        assert single[rank]["num_rollouts"] == [3, 3]
        actual = []
        actual_counts = []
        for start, shards in zip((0, 4), work_units, strict=True):
            order, work_counts = _sample_order(args, shards[rank])
            actual.extend([[start + i for i in mb] for mb in order])
            actual_counts.extend(work_counts)
        assert actual == expected
        assert actual_counts == counts
