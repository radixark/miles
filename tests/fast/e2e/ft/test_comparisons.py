from pathlib import Path
from typing import Any

import pytest
from tests.e2e.ft.conftest_ft import comparisons
from tests.e2e.ft.conftest_ft.app import BASELINE_SIDE, TARGET_SIDE

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME
from miles.utils.test_utils.comparisons.dumps import INPUT_TENSORS_ALLOW_FAILED_PATTERN, INPUT_TENSORS_SKIP_PATTERN

_BASELINE_DIR = "/dumps/baseline"
_TARGET_DIR = "/dumps/target"
_MIN_TRAINED_ROLLOUTS = 2

_PRIMITIVES: tuple[str, ...] = (
    "assert_reconfigure_events",
    "assert_every_metric_is_classified",
    "compare_metrics",
    "compare_dumps",
    "compare_inference_engine_checksums",
    "assert_engine_weights_moved",
    "assert_gradients_were_nonzero",
)


@pytest.fixture
def recorded_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[dict[str, Any]]]:
    calls: dict[str, list[dict[str, Any]]] = {name: [] for name in _PRIMITIVES}

    def make_recorder(name: str):
        def recorder(*args: Any, **kwargs: Any) -> None:
            calls[name].append(dict(args=args, kwargs=kwargs))

        return recorder

    for name in _PRIMITIVES:
        monkeypatch.setattr(comparisons, name, make_recorder(name))

    return calls


def _compare(*, expected_metric_deltas: dict[str, list[float]] | None = None) -> None:
    comparisons.compare_deterministic_sides(
        baseline_dir=_BASELINE_DIR,
        target_dir=_TARGET_DIR,
        min_trained_rollouts=_MIN_TRAINED_ROLLOUTS,
        expected_metric_deltas=expected_metric_deltas,
    )


class TestCompareDeterministicSides:
    def test_metrics_are_compared_bitwise_over_the_compared_prefixes(
        self, recorded_calls: dict[str, list[dict[str, Any]]]
    ) -> None:
        """A nonzero tolerance would let a genuinely nondeterministic run pass this scenario."""
        _compare()

        assert [call["kwargs"] for call in recorded_calls["compare_metrics"]] == [
            dict(
                baseline_dir=_BASELINE_DIR,
                target_dir=_TARGET_DIR,
                rtol=0.0,
                atol=0.0,
                key_prefixes=list(comparisons.COMPARED_METRIC_PREFIXES),
                exclude_keys=[],
                expected_deltas={},
            )
        ]

    def test_scenario_proven_metric_deltas_are_forwarded_without_dropping_the_keys(
        self, recorded_calls: dict[str, list[dict[str, Any]]]
    ) -> None:
        """A counter exception remains an exact comparison whose expected delta comes from the scenario."""
        expected = {"rollout/weight_version/max": [2.0]}

        _compare(expected_metric_deltas=expected)

        assert recorded_calls["compare_metrics"][0]["kwargs"]["expected_deltas"] == expected

    def test_dumps_are_compared_at_zero_relative_difference_with_the_input_tensor_patterns(
        self, recorded_calls: dict[str, list[dict[str, Any]]]
    ) -> None:
        """The threshold and the skip/allow patterns are what make this a bitwise comparison rather than a smoke test."""
        _compare()

        assert [call["kwargs"] for call in recorded_calls["compare_dumps"]] == [
            dict(
                baseline_dir=_BASELINE_DIR,
                target_dir=_TARGET_DIR,
                diff_thresholds=[(".*", "rel <= 0")],
                allow_skipped_pattern=INPUT_TENSORS_SKIP_PATTERN,
                allow_failed_pattern=INPUT_TENSORS_ALLOW_FAILED_PATTERN,
            )
        ]

    def test_every_metric_is_classified_per_side_against_the_prefix_partition(
        self, recorded_calls: dict[str, list[dict[str, Any]]]
    ) -> None:
        """An unclassified metric is one nobody compares, so both sides must be checked against both prefix sets."""
        _compare()

        assert [(call["args"], call["kwargs"]) for call in recorded_calls["assert_every_metric_is_classified"]] == [
            (
                (side_dir,),
                dict(compared=comparisons.COMPARED_METRIC_PREFIXES, ignored=comparisons.UNCOMPARED_METRIC_PREFIXES),
            )
            for side_dir in (_BASELINE_DIR, _TARGET_DIR)
        ]

    def test_each_side_is_asserted_to_have_moved_weights_and_nonzero_gradients_exactly_once(
        self, recorded_calls: dict[str, list[dict[str, Any]]]
    ) -> None:
        """Two identical sides that both trained nothing would compare equal, so each side needs its own witness."""
        _compare()

        assert [call["kwargs"] for call in recorded_calls["assert_engine_weights_moved"]] == [
            dict(side=BASELINE_SIDE, dump_dir=_BASELINE_DIR),
            dict(side=TARGET_SIDE, dump_dir=_TARGET_DIR),
        ]
        assert [call["kwargs"] for call in recorded_calls["assert_gradients_were_nonzero"]] == [
            dict(side=BASELINE_SIDE, dump_dir=_BASELINE_DIR, min_trained_rollouts=_MIN_TRAINED_ROLLOUTS),
            dict(side=TARGET_SIDE, dump_dir=_TARGET_DIR, min_trained_rollouts=_MIN_TRAINED_ROLLOUTS),
        ]

    def test_both_sides_are_required_to_have_reconfigured_never(
        self, recorded_calls: dict[str, list[dict[str, Any]]]
    ) -> None:
        """A side that healed took a different code path, and comparing it bitwise proves nothing about determinism."""
        _compare()

        assert [(call["args"], call["kwargs"]) for call in recorded_calls["assert_reconfigure_events"]] == [
            ((Path(side_dir) / EVENTS_DIRNAME,), dict(expected=[])) for side_dir in (_BASELINE_DIR, _TARGET_DIR)
        ]

    def test_the_engine_checksums_of_the_two_sides_are_compared_once(
        self, recorded_calls: dict[str, list[dict[str, Any]]]
    ) -> None:
        """Matching metrics and dumps still allow the two runs to have served different weights."""
        _compare()

        assert [call["kwargs"] for call in recorded_calls["compare_inference_engine_checksums"]] == [
            dict(baseline_dir=_BASELINE_DIR, target_dir=_TARGET_DIR)
        ]


class TestMetricPrefixPartition:
    def test_the_compared_and_ignored_prefixes_do_not_overlap(self) -> None:
        """A prefix in both sets would be compared and excused at once, hiding whichever answer is wrong."""
        assert not set(comparisons.COMPARED_METRIC_PREFIXES) & set(comparisons.UNCOMPARED_METRIC_PREFIXES)
