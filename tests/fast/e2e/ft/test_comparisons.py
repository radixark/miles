from pathlib import Path
from typing import Any

import pytest
from tests.e2e.ft.conftest_ft import comparisons
from tests.e2e.ft.conftest_ft.app import BASELINE_SIDE, TARGET_SIDE

from tests.fast.e2e.ft.event_fakes import _reconfigure, _write_events
from tests.utils.soak.ft.checkers.reconfigure import ReconfigureInfo

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME
from miles.utils.test_utils.comparisons.dumps import INPUT_TENSORS_ALLOW_FAILED_PATTERN, INPUT_TENSORS_SKIP_PATTERN

_BASELINE_DIR = "/dumps/baseline"
_TARGET_DIR = "/dumps/target"
_MIN_TRAINED_ROLLOUTS = 2
_HEAL_AT_2 = ReconfigureInfo(rollout_id=2, src_cell_index=0, healed_cell_indices=[1], alive_cell_indices_after=[0, 1])

_PRIMITIVES: tuple[str, ...] = (
    "assert_reconfigure_events",
    "assert_metrics_classified",
    "compare_metrics",
    "compare_dumps",
    "compare_inference_engine_checksums",
    "assert_engine_weights_moved",
    "assert_gradients_nonzero",
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


def _compare(
    *, exclude_keys: list[str] | None = None, expected_target_reconfigures: list[ReconfigureInfo] | None = None
) -> None:
    comparisons.compare_deterministic_sides(
        baseline_dir=_BASELINE_DIR,
        target_dir=_TARGET_DIR,
        min_trained_rollouts=_MIN_TRAINED_ROLLOUTS,
        expected_target_reconfigures=expected_target_reconfigures or [],
        exclude_keys=exclude_keys,
    )


def _compare_dirs(root: Path, *, expected: list[ReconfigureInfo]) -> None:
    comparisons.compare_deterministic_sides(
        baseline_dir=str(root / "baseline"),
        target_dir=str(root / "target"),
        min_trained_rollouts=_MIN_TRAINED_ROLLOUTS,
        expected_target_reconfigures=expected,
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
            )
        ]

    def test_a_scenario_may_exclude_named_keys_without_loosening_anything_else(
        self, recorded_calls: dict[str, list[dict[str, Any]]]
    ) -> None:
        """Only the keys a scenario names are dropped; every remaining key keeps the zero-tolerance comparison."""
        excluded = ["rollout/weight_version/max"]

        _compare(exclude_keys=excluded)

        assert [call["kwargs"] for call in recorded_calls["compare_metrics"]] == [
            dict(
                baseline_dir=_BASELINE_DIR,
                target_dir=_TARGET_DIR,
                rtol=0.0,
                atol=0.0,
                key_prefixes=list(comparisons.COMPARED_METRIC_PREFIXES),
                exclude_keys=excluded,
            )
        ]

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

        assert [(call["args"], call["kwargs"]) for call in recorded_calls["assert_metrics_classified"]] == [
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
        assert [call["kwargs"] for call in recorded_calls["assert_gradients_nonzero"]] == [
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


class TestTargetReconfigures:
    def test_a_declared_target_sequence_is_checked_on_the_target_while_the_baseline_stays_quiet(
        self, recorded_calls: dict[str, list[dict[str, Any]]]
    ) -> None:
        """A faulted target must heal exactly as declared while the baseline must still never reconfigure."""
        _compare(expected_target_reconfigures=[_HEAL_AT_2])

        assert [(call["args"], call["kwargs"]) for call in recorded_calls["assert_reconfigure_events"]] == [
            ((Path(_BASELINE_DIR) / EVENTS_DIRNAME,), dict(expected=[])),
            ((Path(_TARGET_DIR) / EVENTS_DIRNAME,), dict(expected=[_HEAL_AT_2])),
        ]

    def test_the_declared_healing_passes_against_real_event_logs(
        self, comparison_primitives_but_reconfigure: list[str], tmp_path: Path
    ) -> None:
        """A target log with exactly the declared healing and a silent baseline must pass every check."""
        _write_events(tmp_path / "baseline" / EVENTS_DIRNAME, [])
        _write_events(tmp_path / "target" / EVENTS_DIRNAME, [_reconfigure(rollout_id=2, healed=[1], alive=[0, 1])])

        _compare_dirs(tmp_path, expected=[_HEAL_AT_2])

        assert "assert_gradients_nonzero" in comparison_primitives_but_reconfigure

    @pytest.mark.parametrize(
        "baseline_events,target_events",
        [
            (
                [_reconfigure(rollout_id=2, healed=[1], alive=[0, 1])],
                [_reconfigure(rollout_id=2, healed=[1], alive=[0, 1])],
            ),
            ([], []),
            ([], [_reconfigure(rollout_id=3, healed=[1], alive=[0, 1])]),
            ([], [_reconfigure(rollout_id=2, healed=[0], alive=[0, 1], src=1)]),
            (
                [],
                [
                    _reconfigure(rollout_id=2, healed=[1], alive=[0, 1]),
                    _reconfigure(rollout_id=4, healed=[1], alive=[0, 1]),
                ],
            ),
        ],
    )
    def test_any_other_reconfigure_history_fails_before_the_bitwise_comparison(
        self,
        comparison_primitives_but_reconfigure: list[str],
        tmp_path: Path,
        baseline_events: list[Any],
        target_events: list[Any],
    ) -> None:
        """A healing baseline, a missing, late, misplaced or extra healing must each fail the comparison."""
        _write_events(tmp_path / "baseline" / EVENTS_DIRNAME, baseline_events)
        _write_events(tmp_path / "target" / EVENTS_DIRNAME, target_events)

        with pytest.raises(AssertionError, match="CellReconfigureEvent sequence mismatch"):
            _compare_dirs(tmp_path, expected=[_HEAL_AT_2])

        assert comparison_primitives_but_reconfigure == []


class TestMetricPrefixPartition:
    def test_the_compared_and_ignored_prefixes_do_not_overlap(self) -> None:
        """A prefix in both sets would be compared and excused at once, hiding whichever answer is wrong."""
        assert not set(comparisons.COMPARED_METRIC_PREFIXES) & set(comparisons.UNCOMPARED_METRIC_PREFIXES)
