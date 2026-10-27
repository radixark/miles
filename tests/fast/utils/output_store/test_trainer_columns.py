"""A trainer shard built through the output store carries the inline path's training columns.

Each case rolls out a batch both ways (``test_plan_vs_inline.run_case``), converts and splits
each batch across DP ranks, and has the store shards materialize their columns from the stored
objects. Every column of every shard must equal the inline shard's: values, dtypes and types.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tests.fast.fixtures.driver_fakes import FakeObjectStore
from tests.fast.ray.rollout.conftest import make_args
from tests.fast.train_parallel_config_utils import make_train_parallel_config
from tests.fast.utils.output_store.test_plan import fake_store_refs
from tests.fast.utils.output_store.test_plan_vs_inline import LAYERS, run_case

from miles.ray.rollout.train_data_conversion import (
    convert_samples_to_train_data,
    process_rollout_data_shard,
    split_train_data_by_dp,
    split_train_data_by_dp_raw,
)
from miles.rollout.generate_utils import tool_call_utils
from miles.utils import object_store
from miles.utils.data import process_rollout_data
from miles.utils.object_store import ObjectStoreGetResult
from miles.utils.output_store.plan import MASK_SUPPORT_LOGPROBS, StorePlanError
from miles.utils.output_store.trainer_columns import materialize_store_columns

DP_SIZE = 2


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    monkeypatch.setattr(tool_call_utils, "tokenize_tool_responses", lambda messages, tokenizer: list(messages))
    fake_store_refs(monkeypatch)


def reader(objects: dict, calls: list):
    """``OutputStore.read_many`` over plain arrays, recording what it was asked for."""

    def read(requests):
        calls.append([(ref.key, tuple(names)) for ref, names in requests])
        return [SimpleNamespace(**{name: objects[ref.key][name] for name in names}) for ref, names in requests]

    return read


def rollout(seed: int):
    rng = np.random.default_rng(seed)
    launch_mode, k = [("selected", 0), ("support", int(rng.integers(1, 5))), ("support", 0)][seed % 3]
    mode = "support" if launch_mode == "support" and k else "selected"
    routed_experts = bool(rng.random() < 0.5)
    inline, store, objects = [], [], {}
    for index in range(int(rng.integers(DP_SIZE, 7))):
        # A no-output abort is retried, never trained.
        a, b, objs = run_case(rng, mode=mode, k=k, routed_experts=routed_experts, key_prefix=f"{index}/", aborts=False)
        for sample in (a, b):
            sample.index, sample.reward = index, float(index)
        inline.append(a)
        store.append(b)
        objects |= objs
    args = make_args(
        rewards_normalization=False,
        balance_data=False,
        rollout_top_logprobs_num=k,
        rollout_sampling_logprobs_mode=launch_mode,
        loss_type="score_centering" if seed % 2 else "policy_loss",
        ci_test=True,
        use_rollout_routing_replay=routed_experts,
        num_layers=LAYERS,
        # The trainer's TrainerConfig keeps the indexer stream count under backend.
        backend=SimpleNamespace(rollout_indexer_topk_num_streams=None),
    )
    return args, inline, store, objects, mode


def assert_same_column(store, inline, name):
    assert len(store) == len(inline), name
    for s, i in zip(store, inline, strict=True):
        assert type(s) is type(i), (name, type(s), type(i))
        if isinstance(i, torch.Tensor):
            assert s.dtype == i.dtype and torch.equal(s, i), name
        elif isinstance(i, np.ndarray):
            assert s.dtype == i.dtype, name
            np.testing.assert_array_equal(s, i, err_msg=name)
        else:
            assert s == i, name


@pytest.mark.parametrize("seed", range(120))
def test_store_shards_carry_the_inline_columns(seed):
    args, inline, store, objects, mode = rollout(seed)
    inline_shards = split_train_data_by_dp_raw(
        args, convert_samples_to_train_data(args, inline, {}, None, None), dp_size=DP_SIZE
    )
    store_shards = split_train_data_by_dp_raw(
        args, convert_samples_to_train_data(args, store, {}, None, None), dp_size=DP_SIZE
    )

    for inline_shard, store_shard in zip(inline_shards, store_shards, strict=True):
        calls = []
        materialize_store_columns(args, store_shard, read=reader(objects, calls))

        assert store_shard.keys() == inline_shard.keys()
        for name, column in inline_shard.items():
            assert_same_column(store_shard[name], column, name)
        # One read per shard, each object once; selected requests stored no support log-probs.
        (requests,) = calls
        assert len({key for key, _ in requests}) == len(requests)
        if mode == "selected":
            assert all(MASK_SUPPORT_LOGPROBS not in names for _, names in requests)


def test_a_shard_without_plans_is_left_alone():
    shard = {"tokens": [[1, 2]], "response_lengths": [1]}
    # The column alone decides: no args field is read.
    materialize_store_columns(SimpleNamespace(), shard, read=None)
    assert shard == {"tokens": [[1, 2]], "response_lengths": [1]}


def test_plans_without_a_reader_or_their_objects_fail_without_retries():
    args, _, store, objects, _ = rollout(1)
    data = convert_samples_to_train_data(args, store, {}, None, None)
    with pytest.raises(StorePlanError, match="no output store reader"):
        materialize_store_columns(args, dict(data), read=None)
    # A deleted object: an FT retry would read the same nothing.
    missing = {key: arrays for key, arrays in objects.items() if not key.startswith("0/")}
    with pytest.raises(StorePlanError, match="cannot read this shard's output store objects"):
        materialize_store_columns(args, dict(data), read=reader(missing, []))


def test_ci_test_catches_a_sampled_log_prob_the_candidates_disagree_with():
    args, _, store, objects, mode = rollout(1)
    assert mode == "support"
    tampered = next(sample for sample in store if sample.response_length)
    tampered.rollout_log_probs[-1] += 0.5
    data = convert_samples_to_train_data(args, store, {}, None, None)
    with pytest.raises(ValueError, match="same sampler distribution"):
        materialize_store_columns(args, data, read=reader(objects, []))


@pytest.mark.parametrize("delay_split", [True, False])
def test_every_rank_materializes_its_own_shard(monkeypatch, delay_split):
    monkeypatch.setattr(object_store, "_INSTANCE", FakeObjectStore())
    args, inline, store, objects, _ = rollout(4)
    args.delay_split_train_data_by_dp = delay_split
    config = make_train_parallel_config(dp_size=DP_SIZE)
    inline_shards = split_train_data_by_dp(
        args, convert_samples_to_train_data(args, inline, {}, None, None), train_parallel_config=config
    )
    data = convert_samples_to_train_data(args, store, {}, None, None)
    put = object_store.get_instance().put
    ref = (
        put(value=data) if delay_split else [put(value=shard) for shard in split_train_data_by_dp(args, data, config)]
    )

    for rank in range(DP_SIZE):
        calls = []
        shard, result = process_rollout_data(
            args,
            ref,
            dp_rank=rank,
            train_parallel_config=config,
            witness_info=None,
            output_store_read=reader(objects, calls),
        )
        expected = process_rollout_data_shard(args, dict(inline_shards[rank]))
        assert shard.keys() == expected.keys()
        for name, column in expected.items():
            assert_same_column(shard[name], column, name)
        # The stored value keeps its plans: an FT retry reads them again.
        assert "rollout_store_plan" in result.value and len(calls) == 1


def test_a_failed_store_read_releases_the_shard(monkeypatch):
    store = FakeObjectStore()
    released = []
    get = store.get
    monkeypatch.setattr(store, "get", lambda ref: ObjectStoreGetResult(get(ref).value, release_fn=released.append))
    monkeypatch.setattr(object_store, "_INSTANCE", store)
    args, _, samples, _, _ = rollout(4)
    args.delay_split_train_data_by_dp = False
    config = make_train_parallel_config(dp_size=DP_SIZE)
    data = convert_samples_to_train_data(args, samples, {}, None, None)
    ref = [store.put(value=shard) for shard in split_train_data_by_dp(args, data, config)]

    def read(requests):
        raise TimeoutError("output store read timed out")

    with pytest.raises(TimeoutError):
        process_rollout_data(
            args, ref, dp_rank=0, train_parallel_config=config, witness_info=None, output_store_read=read
        )
    # The caller never got the result to release: an FT retry would leak the shard's buffer.
    assert len(released) == 1
