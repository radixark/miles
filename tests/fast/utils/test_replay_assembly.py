"""--enable-delay-token-replay-assembly moves the replay assembly to the train ranks, not its output."""

import math
from copy import deepcopy

import numpy as np
import pybase64
import pytest
import torch
from tests.fast.ray.rollout.conftest import make_args

from miles.backends.training_utils.data.replay import materialize_replay_columns
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data, split_train_data_by_dp_raw
from miles.utils.replay_index import ReplayIndex, sampled_logprobs_from_meta_info
from miles.utils.types import Sample

LAYERS, TOPK, INDEXER_TOPK, K = 2, 2, 4, 3


def _b64(array: np.ndarray) -> str:
    return pybase64.b64encode(np.ascontiguousarray(array, dtype=np.int32).tobytes()).decode("ascii")


def _response(supports: list[list[int]], *, rows: int, seed: int) -> dict:
    """A support-mode SGLang response with R3 and indexer top-k rows for the whole sequence."""
    rng = np.random.default_rng(seed)
    return {
        "output_token_sampling_mask": supports,
        "output_token_sampling_logprobs": [[math.log(1 / len(row))] * len(row) for row in supports],
        "routed_experts": _b64(rng.integers(1, 8, size=(rows, LAYERS, TOPK))),
        "indexer_topk": _b64(rng.integers(0, 50, size=(rows, LAYERS, INDEXER_TOPK))),
        "indexer_topk_num_layers": LAYERS,
    }


def _append_response(sample: Sample, output: list[int], supports: list[list[int]], *, seed: int) -> None:
    """Record one generation call as the /generate path does."""
    rows = len(sample.tokens) + len(output) - 1
    meta_info = _response(supports, rows=rows, seed=seed)
    new = ReplayIndex.from_meta_info(
        meta_info,
        via_store=False,
        routed_experts_rows=rows,
        num_layers=LAYERS,
        indexer_topk_rows=rows,
        indexer_num_streams=None,
        return_sampling_mask=True,
        support_logprobs=True,
        num_output_tokens=len(output),
        aborted=False,
    )
    sample.replay_index = sample.replay_index.append_response(new, response_length=sample.response_length)
    sample.rollout_log_probs += sampled_logprobs_from_meta_info(
        meta_info, output_token_ids=output, support_logprobs=True, aborted=False
    )
    sample.tokens = sample.tokens + output
    sample.response_length += len(output)
    sample.loss_mask += [1] * len(output)


def _tool_turn_sample(index: int) -> Sample:
    """Two generated turns around a tool observation, whose rows are forced."""
    sample = Sample(
        index=index,
        group_index=index // 2,
        tokens=[1, 2, 3 + index],
        response_length=0,
        loss_mask=[],
        rollout_log_probs=[],
        reward=float(index),
        status=Sample.Status.COMPLETED,
    )
    _append_response(sample, [10, 11 + index], [[10, 30], [31, 11 + index, 32]], seed=index)
    observation = [20, 21]
    sample.replay_index = sample.replay_index.append_forced_tokens(
        len(observation), response_length=sample.response_length
    )
    sample.tokens = sample.tokens + observation
    sample.response_length += len(observation)
    sample.loss_mask += [0] * len(observation)
    sample.rollout_log_probs += [0.0] * len(observation)
    _append_response(sample, [12], [[12, 33]], seed=100 + index)
    return sample


def _assert_same(expected, actual, where: str) -> None:
    assert type(actual) is type(expected), where
    if isinstance(expected, np.ndarray):
        assert (actual.dtype, actual.shape) == (expected.dtype, expected.shape), where
        np.testing.assert_array_equal(actual, expected, err_msg=where)
    elif isinstance(expected, torch.Tensor):
        assert (actual.dtype, actual.shape) == (expected.dtype, expected.shape), where
        assert torch.equal(actual, expected), where
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), where
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _assert_same(e, a, f"{where}[{i}]")
    else:
        assert actual == expected, where


@pytest.mark.parametrize("loss_type", ["policy_loss", "score_centering"])
def test_delayed_shards_match_the_rollout_assembled_shards(loss_type):
    samples = [_tool_turn_sample(i) for i in range(4)]
    kwargs = dict(
        rewards_normalization=False,
        rollout_top_logprobs_num=K,
        rollout_sampling_logprobs_mode="support",
        loss_type=loss_type,
        use_rollout_routing_replay=True,
        use_rollout_indexer_replay=True,
        ci_test=True,
    )
    args = make_args(**kwargs, enable_delay_token_replay_assembly=False)
    delayed_args = make_args(**kwargs, enable_delay_token_replay_assembly=True)
    data = convert_samples_to_train_data(args, deepcopy(samples), {}, None, None)
    delayed = convert_samples_to_train_data(delayed_args, deepcopy(samples), {}, None, None)
    assert "replay_index" not in data and "replay_index" in delayed
    # Score centering shares the support IDs in both places; other losses keep the dense candidates.
    assert ("rollout_topk_lengths" in data) == (loss_type == "score_centering")

    shards = split_train_data_by_dp_raw(args, data, dp_size=2)
    for rank, (shard, delayed_shard) in enumerate(
        zip(shards, split_train_data_by_dp_raw(delayed_args, delayed, dp_size=2), strict=True)
    ):
        materialize_replay_columns(args, shard)
        materialize_replay_columns(delayed_args, delayed_shard)
        assert delayed_shard.keys() == shard.keys()
        for key in shard:
            _assert_same(shard[key], delayed_shard[key], f"rank {rank} {key}")
