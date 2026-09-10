from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

import warnings
from argparse import Namespace

import pytest

from miles.rollout.filter_hub import dynamic_sampling_filters
from miles.rollout.filter_hub.base_types import DynamicFilterOutput, FilterOutput, iter_samples
from miles.rollout.filter_hub.common_filters import (
    apply_aborted_filter,
    apply_missing_reward_filter,
    apply_preput_filters,
    apply_reward_nonzero_std_filter,
    group_staleness,
)
from miles.utils.function_registry import load_function
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


@pytest.mark.parametrize(
    ("legacy_name", "canonical"),
    [
        ("check_no_aborted", apply_aborted_filter),
        ("check_reward_nonzero_std", apply_reward_nonzero_std_filter),
    ],
)
def test_legacy_filters_forward_without_runtime_deprecation_warnings(monkeypatch, legacy_name, canonical):
    legacy = load_function(f"miles.rollout.filter_hub.dynamic_sampling_filters.{legacy_name}")
    assert legacy.__name__ == legacy_name
    assert canonical.__name__ in legacy.__deprecated__
    assert not hasattr(canonical, "__deprecated__")

    args, samples, extra = Namespace(reward_key=None), [make_sample()], object()
    result = FilterOutput(keep=False, reason="unchanged")

    def replacement(forwarded_args, forwarded_samples, **kwargs):
        assert forwarded_args is args
        assert forwarded_samples is samples
        assert kwargs == {"extra": extra}
        return result

    monkeypatch.setattr(dynamic_sampling_filters, canonical.__name__, replacement)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert legacy(args, samples, extra=extra) is result

    error = ValueError("filter failed")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(dynamic_sampling_filters, canonical.__name__, fail)
    with pytest.raises(ValueError) as raised:
        legacy(args=args, samples=samples, extra=extra)
    assert raised.value is error


@pytest.mark.parametrize(
    ("legacy_name", "samples", "expected"),
    [
        ("check_no_aborted", [make_sample()], FilterOutput(keep=True)),
        (
            "check_no_aborted",
            [[make_sample(status=Sample.Status.ABORTED)]],
            FilterOutput(keep=False, reason="group_has_aborted"),
        ),
        (
            "check_reward_nonzero_std",
            [make_sample(reward=1.0), make_sample(reward=2.0)],
            FilterOutput(keep=True),
        ),
        (
            "check_reward_nonzero_std",
            [[make_sample(reward=1.0), make_sample(reward=1.0)]],
            FilterOutput(keep=False, reason="zero_std_1.0"),
        ),
    ],
)
def test_legacy_filters_preserve_structured_results(legacy_name, samples, expected):
    legacy = load_function(f"miles.rollout.filter_hub.dynamic_sampling_filters.{legacy_name}")
    result = legacy(Namespace(reward_key=None), samples, ignored=True)
    assert isinstance(result, DynamicFilterOutput)
    assert result == expected


def test_iter_samples_preserves_flat_and_mixed_nested_order():
    samples = [Sample(index=index) for index in range(4)]

    assert list(iter_samples(samples)) == samples
    assert list(iter_samples([samples[0], samples[1:3], samples[3]])) == samples


@pytest.mark.parametrize(
    ("filter_fn", "group", "expected"),
    [
        (
            apply_aborted_filter,
            [make_sample(status=Sample.Status.ABORTED)],
            FilterOutput(keep=False, reason="group_has_aborted"),
        ),
        (
            apply_missing_reward_filter,
            [make_sample(reward=None)],
            FilterOutput(keep=False, reason="group_has_missing_reward"),
        ),
    ],
)
def test_common_filters_return_structured_drop(filter_fn, group, expected):
    assert filter_fn(Namespace(reward_key=None), group, ignored=True) == expected


@pytest.mark.parametrize("reward", [None, 1.0], ids=["missing-reward", "numeric-reward"])
def test_preput_aborted_wins_before_missing_reward_and_dynamic_filter(reward):
    custom_calls = []

    def dynamic_filter(*args, **kwargs):
        custom_calls.append((args, kwargs))
        return True

    output = apply_preput_filters(
        Namespace(reward_key=None),
        dynamic_filter,
        [make_sample(reward=reward, status=Sample.Status.ABORTED)],
    )

    assert output == FilterOutput(keep=False, reason="group_has_aborted")
    assert custom_calls == []


@pytest.mark.parametrize(
    ("reward_key", "group"),
    [
        (None, [make_sample(reward=None)]),
        ("score", [make_sample(reward={"score": None})]),
        (None, [make_sample(), [make_sample(reward=None)]]),
    ],
    ids=["raw", "selected", "nested"],
)
def test_preput_missing_reward_wins_before_dynamic_filter(reward_key, group):
    custom_calls = []

    def dynamic_filter(*args, **kwargs):
        custom_calls.append((args, kwargs))
        return True

    output = apply_preput_filters(
        Namespace(reward_key=reward_key),
        dynamic_filter,
        group,
    )

    assert output == FilterOutput(keep=False, reason="group_has_missing_reward")
    assert custom_calls == []


def test_preput_selected_reward_missing_key_propagates():
    custom_calls = []

    def dynamic_filter(*args, **kwargs):
        custom_calls.append((args, kwargs))
        return True

    with pytest.raises(KeyError, match="score"):
        apply_preput_filters(
            Namespace(reward_key="score"),
            dynamic_filter,
            [make_sample(reward={"other": 1.0})],
        )

    assert custom_calls == []


def test_preput_preserves_filter_output_identity():
    output = FilterOutput(keep=False, reason="custom_reason")

    assert apply_preput_filters(Namespace(reward_key=None), lambda *args, **kwargs: output, [make_sample()]) is output


def test_preput_normalizes_legacy_false():
    output = apply_preput_filters(Namespace(reward_key=None), lambda *args, **kwargs: False, [make_sample()])

    assert output == FilterOutput(keep=False)


def test_preput_passes_kwargs_and_returns_keep_output():
    args = Namespace(reward_key=None)
    group = [make_sample()]
    marker = object()
    calls = []

    def dynamic_filter(received_args, received_group, **kwargs):
        calls.append((received_args, received_group, kwargs))
        return FilterOutput(keep=True)

    output = apply_preput_filters(
        args,
        dynamic_filter,
        group,
        marker=marker,
    )

    assert output == FilterOutput(keep=True)
    assert calls == [(args, group, {"marker": marker})]


def test_preput_propagates_dynamic_filter_exception():
    def dynamic_filter(*args, **kwargs):
        raise RuntimeError("dynamic filter failed")

    with pytest.raises(RuntimeError, match="dynamic filter failed"):
        apply_preput_filters(
            Namespace(reward_key=None),
            dynamic_filter,
            [make_sample()],
        )


def test_group_staleness_uses_oldest_version_across_nested_samples():
    group = [
        make_sample(weight_versions=("9",)),
        [make_sample(weight_versions=("4", "8")), make_sample(weight_versions=())],
    ]

    assert group_staleness(group, current_version=10) == 6
    assert group_staleness(group, current_version=None) is None
    assert group_staleness([make_sample()], current_version=10) is None
    assert group_staleness([make_sample(weight_versions=("12",))], current_version=10) == -2
