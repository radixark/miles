from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-fast")

from miles.rollout.generate_hub.agentic_tool_call import _take_samples_until_terminal
from miles.utils.types import Sample


def _sample(status: Sample.Status) -> Sample:
    return Sample(status=status)


def test_take_samples_until_terminal_keeps_completed_samples():
    samples = [_sample(Sample.Status.COMPLETED), _sample(Sample.Status.COMPLETED)]

    assert _take_samples_until_terminal(samples) == samples


def test_take_samples_until_terminal_keeps_first_terminal_sample():
    samples = [
        _sample(Sample.Status.COMPLETED),
        _sample(Sample.Status.TRUNCATED),
        _sample(Sample.Status.COMPLETED),
    ]

    assert _take_samples_until_terminal(samples) == samples[:2]
