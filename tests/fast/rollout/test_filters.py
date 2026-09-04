from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

from argparse import Namespace

from miles.rollout.filter_hub.base_types import DynamicFilterOutput, FilterOutput, iter_samples
from miles.rollout.filter_hub.common_filters import check_no_aborted, check_reward_nonzero_std, group_staleness
from miles.rollout.filter_hub.dynamic_sampling_filters import (
    check_reward_nonzero_std as legacy_check_reward_nonzero_std,
)
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall


def make_sample(
    *,
    reward=1.0,
    status=Sample.Status.COMPLETED,
    weight_versions=(),
) -> Sample:
    return Sample(
        reward=reward,
        status=status,
        weight_versions=[
            WeightVersionsPerCall(spans=[WeightVersionSpan(version=version, abs_start=0, abs_end=1)])
            for version in weight_versions
        ],
    )


def test_dynamic_filter_output_is_a_compatibility_alias():
    assert DynamicFilterOutput is FilterOutput


def test_dynamic_sampling_filters_reexports_common_implementation():
    assert legacy_check_reward_nonzero_std is check_reward_nonzero_std


def test_iter_samples_preserves_flat_and_mixed_nested_order():
    samples = [Sample(index=index) for index in range(4)]

    assert list(iter_samples(samples)) == samples
    assert list(iter_samples([samples[0], samples[1:3], samples[3]])) == samples


def test_common_filter_returns_structured_drop():
    assert check_no_aborted(
        Namespace(reward_key=None),
        [make_sample(status=Sample.Status.ABORTED)],
        ignored=True,
    ) == FilterOutput(keep=False, reason="group_has_aborted")


def test_group_staleness_uses_oldest_version_across_nested_samples():
    group = [
        make_sample(weight_versions=("9",)),
        [make_sample(weight_versions=("4", "8")), make_sample(weight_versions=())],
    ]

    assert group_staleness(group, current_version=10) == 6
    assert group_staleness(group, current_version=None) is None
    assert group_staleness([make_sample()], current_version=10) is None
    assert group_staleness([make_sample(weight_versions=("12",))], current_version=10) == -2
