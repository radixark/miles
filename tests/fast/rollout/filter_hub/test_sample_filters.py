from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

import logging
from argparse import Namespace

import pytest

from miles.rollout.filter_hub import sample_filters
from miles.rollout.filter_hub.sample_filters import mask_truncated_zero_reward
from miles.utils.function_registry import load_function
from miles.utils.types import Sample

ARGS = Namespace(reward_key=None)


def make_sample(*, reward=0.0, status=Sample.Status.COMPLETED, exit_status=None) -> Sample:
    return Sample(reward=reward, status=status, metadata={} if exit_status is None else {"exit_status": exit_status})


@pytest.mark.parametrize(
    "sample",
    [
        make_sample(status=Sample.Status.TRUNCATED),
        make_sample(exit_status="TimeLimitExceeded"),
        make_sample(exit_status="SequenceLengthLimitExceeded"),
    ],
    ids=["status_truncated", "time_limit", "context_limit"],
)
def test_cut_off_zero_reward_is_masked(sample: Sample) -> None:
    mask_truncated_zero_reward(ARGS, [[sample]])

    assert sample.remove_sample


@pytest.mark.parametrize(
    "sample",
    [
        make_sample(reward=1.0, status=Sample.Status.TRUNCATED),  # grader passed the patch written before the cut
        make_sample(reward=1.0, exit_status="TimeLimitExceeded"),
        make_sample(reward=0.0),  # finished and wrong: a real negative
        make_sample(reward=0.0, exit_status="Submitted"),
        make_sample(reward=0.0, exit_status="AgentError"),  # infra failures are a different filter's call
        make_sample(reward={"score": 0.0}, exit_status="TimeLimitExceeded"),  # dict reward, no --reward-key
    ],
    ids=[
        "truncated_correct",
        "time_limit_correct",
        "completed_wrong",
        "submitted_wrong",
        "agent_error",
        "dict_reward",
    ],
)
def test_other_samples_stay_in_the_loss(sample: Sample) -> None:
    mask_truncated_zero_reward(ARGS, [[sample]])

    assert not sample.remove_sample


def test_reward_key_is_respected() -> None:
    args = Namespace(reward_key="score")
    passed = make_sample(reward={"score": 1.0}, exit_status="TimeLimitExceeded")
    failed = make_sample(reward={"score": 0.0}, exit_status="TimeLimitExceeded")

    mask_truncated_zero_reward(args, [[passed, failed]])

    assert [passed.remove_sample, failed.remove_sample] == [False, True]


def test_dict_reward_without_reward_key_warns_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(sample_filters, "_warned_dict_reward", False)

    with caplog.at_level(logging.WARNING, logger=sample_filters.__name__):
        for _ in range(2):
            mask_truncated_zero_reward(ARGS, [[make_sample(reward={"score": 0.0}, exit_status="TimeLimitExceeded")]])

    assert sum("--reward-key is not set" in r.getMessage() for r in caplog.records) == 1


def test_loadable_by_rollout_sample_filter_path() -> None:
    fn = load_function("miles.rollout.filter_hub.sample_filters.mask_truncated_zero_reward")

    assert fn is mask_truncated_zero_reward
