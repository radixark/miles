"""Trainer side of the replay data: chunked, deduplicated store reads and the columns training reads."""

from argparse import Namespace

import numpy as np
import pytest
from tests.fast.fixtures.output_store_fixtures import b64, make_store, mask_arrays, put_response

from miles.backends.training_utils.data import replay
from miles.backends.training_utils.data.replay import materialize_replay_columns, read_store_objects
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data, split_train_data_by_dp_raw
from miles.utils.replay_index import ReplayDataError, ReplayIndex
from miles.utils.types import Sample

LAYERS, TOPK = 3, 2
SUPPORTS = [[7, 3], [8, 9]]
LOGPROBS = [[-0.1, -2.0], [-0.3, -1.5]]


def _args(**overrides) -> Namespace:
    values = dict(
        ci_test=False,
        enable_sample_ownership_checker=False,
        loss_type="policy_loss",
        rollout_top_logprobs_num=2,
        rollout_sampling_logprobs_mode="support",
        use_sampling_support_replay=True,
        use_rollout_routing_replay=True,
        use_rollout_indexer_replay=False,
        advantage_estimator="grpo",
        rewards_normalization=False,
        reward_key=None,
        use_dynamic_global_batch_size=False,
        balance_data=False,
        output_store_local_buffer_size="1gb",
        # --use-output-store turns it on: only the train ranks read store objects.
        enable_delay_token_replay_assembly=True,
    )
    return Namespace(**(values | overrides))


def _rows(n: int, start: int = 1) -> np.ndarray:
    return np.arange(start, start + n * LAYERS * TOPK, dtype=np.int32).reshape(n, LAYERS, TOPK)


def _sample(index: int, replay_index: ReplayIndex) -> Sample:
    return Sample(
        index=index,
        group_index=0,
        tokens=[1, 2, 7, 8],
        response_length=2,
        loss_mask=[1, 1],
        # The sampled tokens' log-probs, which the rollout reads from the response.
        rollout_log_probs=[-0.1, -0.3],
        reward=1.0,
        status=Sample.Status.COMPLETED,
        replay_index=replay_index,
    )


def _parse(meta_info: dict, *, via_store: bool) -> ReplayIndex:
    return ReplayIndex.from_meta_info(
        meta_info,
        via_store=via_store,
        routed_experts_rows=3,
        num_layers=LAYERS,
        indexer_topk_rows=None,
        indexer_num_streams=None,
        return_sampling_mask=True,
        support_logprobs=True,
        num_output_tokens=2,
        aborted=False,
    )


def _inline(start: int) -> ReplayIndex:
    meta = {
        "routed_experts": b64(_rows(3, start)),
        "output_token_sampling_mask": SUPPORTS,
        "output_token_sampling_logprobs": LOGPROBS,
    }
    return _parse(meta, via_store=False)


def _stored(store, start: int) -> ReplayIndex:
    ref = put_response(store, routed_experts=_rows(3, start), **mask_arrays(SUPPORTS, LOGPROBS))
    return _parse({"output_store_ref": ref, "output_token_sampling_mask_length": 2}, via_store=True)


@pytest.fixture
def store(monkeypatch):
    store, backend = make_store()
    monkeypatch.setattr(replay, "get_output_store_reader", lambda: store)
    return store, backend


def _train_columns(args, samples) -> dict:
    data = convert_samples_to_train_data(args, samples, {}, None, None)
    materialize_replay_columns(args, data)
    return data


def _assert_same_columns(actual: dict, expected: dict) -> None:
    keys = ("rollout_routed_experts", "rollout_sampling_mask_ids", "rollout_sampling_mask_offsets")
    for key in (*keys, "rollout_log_probs", "rollout_topk_token_ids", "rollout_topk_log_probs"):
        assert len(actual[key]) == len(expected[key]), key
        for a, b in zip(actual[key], expected[key], strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=key)
            assert np.asarray(a).dtype == np.asarray(b).dtype, key


def test_store_and_mixed_batches_train_on_the_same_columns_as_inline(store):
    store, _ = store
    inline = _train_columns(_args(), [_sample(0, _inline(1)), _sample(1, _inline(50))])
    mixed = _train_columns(_args(), [_sample(0, _inline(1)), _sample(1, _stored(store, 50))])
    stored = _train_columns(_args(), [_sample(0, _stored(store, 1)), _sample(1, _stored(store, 50))])

    _assert_same_columns(mixed, inline)
    _assert_same_columns(stored, inline)
    assert "replay_index" not in stored and "rollout_sampling_mask_lengths" not in stored
    np.testing.assert_array_equal(inline["rollout_routed_experts"][1], _rows(3, 50))
    assert inline["rollout_topk_token_ids"][0].tolist() == [[7, 3], [8, 9]]
    assert inline["rollout_log_probs"][0] == [-0.1, -0.3]


def test_reads_each_object_field_once_and_only_the_fields_used(store):
    store, backend = store
    ref = put_response(store, routed_experts=_rows(3), indexer_topk=_rows(3, 9), **mask_arrays(SUPPORTS, LOGPROBS))
    meta = {"output_store_ref": ref, "output_token_sampling_mask_length": 2}
    index = _parse(meta, via_store=True)
    # Two samples (v2 leaves sharing a turn) index the same object.
    fetched = read_store_objects([index, index.strip_tail(1)], buffer_bytes=1 << 20)

    (read,) = backend.reads
    assert sorted(name.rsplit("/", 1)[1] for name, _ in read) == [
        "output_token_sampling_logprobs",
        "output_token_sampling_mask_lengths",
        "output_token_sampling_mask_token_ids",
        "routed_experts",
    ]
    assert len(fetched) == 4


def test_reads_in_chunks_that_fit_the_buffer_and_copies_each_out(store):
    store, backend = store
    indexes = [_stored(store, start) for start in (1, 50, 100)]
    largest = max(spec.nbytes for index in indexes for ref in index.store_refs() for spec in ref.fields)
    fetched = read_store_objects(indexes, buffer_bytes=largest + 8)

    assert len(backend.reads) > 1
    assert all(sum(nbytes for _, nbytes in read) <= largest + 8 for read in backend.reads)
    assert sum(len(read) for read in backend.reads) == len(fetched) == 3 * 4
    for value in fetched.values():
        assert value.flags.owndata or value.base is None or isinstance(value.base, np.ndarray)


def test_object_larger_than_the_buffer_fails_before_any_read(store):
    store, backend = store
    index = _stored(store, 1)
    with pytest.raises(ReplayDataError, match="raise --output-store-local-buffer-size to at least 72"):
        read_store_objects([index], buffer_bytes=64)
    assert backend.reads == []


def test_missing_object_fails_loudly_with_its_key(store):
    store, backend = store
    index = _stored(store, 1)
    backend.parts.clear()
    with pytest.raises(ReplayDataError, match="failed to read 1 output store objects .*miles-output-store/test/"):
        read_store_objects([index], buffer_bytes=1 << 20)


def test_all_zero_routed_experts_fail_on_the_trainer(store):
    store, _ = store
    ref = put_response(store, routed_experts=np.zeros((3, LAYERS, TOPK), dtype=np.int32))
    index = ReplayIndex.from_meta_info(
        {"output_store_ref": ref},
        via_store=True,
        routed_experts_rows=3,
        num_layers=LAYERS,
        indexer_topk_rows=None,
        indexer_num_streams=None,
        return_sampling_mask=False,
        support_logprobs=False,
        num_output_tokens=2,
        aborted=False,
    )
    with pytest.raises(ReplayDataError, match="all zeros"):
        _train_columns(_args(use_sampling_support_replay=False, rollout_top_logprobs_num=0), [_sample(0, index)])


def test_shards_read_their_own_rows_whether_split_before_or_after_transport(store):
    store, backend = store
    samples = [_sample(i, _stored(store, 1 + 40 * i)) for i in range(4)]
    args = _args()
    data = convert_samples_to_train_data(args, samples, {}, None, None)
    shards = split_train_data_by_dp_raw(args, dict(data), dp_size=2)

    reads_before = len(backend.reads)
    for shard in shards:
        partition = shard["partition"]
        materialize_replay_columns(args, shard)
        for row, sample_index in enumerate(partition):
            np.testing.assert_array_equal(shard["rollout_routed_experts"][row], _rows(3, 1 + 40 * sample_index))
    read_keys = {name.rsplit("/", 1)[0] for read in backend.reads[reads_before:] for name, _ in read}
    assert len(read_keys) == 4


def test_ci_validation_runs_on_trainer_derived_candidates(store):
    store, _ = store
    data = _train_columns(_args(ci_test=True), [_sample(0, _stored(store, 1))])
    assert data["rollout_topk_token_ids"][0].tolist() == [[7, 3], [8, 9]]


def test_sampled_token_outside_its_support_fails_with_the_sample(store):
    store, _ = store
    sample = _sample(5, _stored(store, 1))
    sample.tokens = [1, 2, 7, 4]
    with pytest.raises(ReplayDataError, match=r"sample_index=5\): sampled token 4 is absent"):
        _train_columns(_args(), [sample])
