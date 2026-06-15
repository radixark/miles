from types import SimpleNamespace

from miles.rollout.filter_hub.dynamic_sampling_filters import (
    AGENTIC_INTERMEDIATE_TRUNCATED_KEY,
    check_no_aborted_or_truncated_and_reward_nonzero_std,
)
from miles.utils.types import Sample


def _args():
    return SimpleNamespace(reward_key=None)


def _sample(status=Sample.Status.COMPLETED, reward=1.0, metadata=None):
    return Sample(status=status, reward=reward, metadata=metadata or {})


def test_filter_rejects_agentic_intermediate_truncated_group():
    samples = [
        _sample(reward=1.0),
        _sample(
            status=Sample.Status.TRUNCATED,
            reward=0.0,
            metadata={AGENTIC_INTERMEDIATE_TRUNCATED_KEY: True},
        ),
    ]

    output = check_no_aborted_or_truncated_and_reward_nonzero_std(_args(), samples)

    assert not output.keep
    assert output.reason == "group_has_agentic_intermediate_truncated"


def test_filter_rejects_plain_truncated_group():
    samples = [
        _sample(reward=1.0),
        _sample(status=Sample.Status.TRUNCATED, reward=0.0),
    ]

    output = check_no_aborted_or_truncated_and_reward_nonzero_std(_args(), samples)

    assert not output.keep
    assert output.reason == "group_has_truncated"
