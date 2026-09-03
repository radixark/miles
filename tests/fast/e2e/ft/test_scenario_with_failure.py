import json
import shlex
from pathlib import Path

import pytest
from tests.e2e.ft.conftest_ft import scenario_with_failure
from tests.e2e.ft.conftest_ft.modes import MODES
from tests.e2e.ft.conftest_ft.scenario_with_failure import (
    _DIFF_THRESHOLDS,
    _FAULT_ROLLOUT_ID,
    _FIRST_INJECTED_ROLLOUT_ID,
    _FIRST_POST_FAULT_ROLLOUT_ID,
    _POST_FAULT_DIFF_THRESHOLDS,
    _assert_event_analysis,
    _build_baseline_args,
    _build_target_args,
    _diff_thresholds_for_rollout,
)


def _option_value(args: str, option: str) -> str:
    tokens = shlex.split(args)
    return tokens[tokens.index(option) + 1]


def test_real_rollout_injection_starts_at_fault_rollout() -> None:
    """Real-rollout training data injection must cover the fault rollout itself."""
    args = _build_target_args(
        MODES["dp2_cp2_real_rollout_dense"],
        "/tmp/target/phase_b",
        enable_dumper=False,
    )
    actions = json.loads(_option_value(args, "--ci-ft-test-actions"))

    assert _FIRST_INJECTED_ROLLOUT_ID == _FAULT_ROLLOUT_ID
    assert _FIRST_POST_FAULT_ROLLOUT_ID == _FAULT_ROLLOUT_ID + 1
    assert int(_option_value(args, "--ci-inject-rollout-data-start-rollout-id")) == _FAULT_ROLLOUT_ID
    assert {action["at_rollout"] for action in actions} == {_FAULT_ROLLOUT_ID}


def test_fault_rollout_keeps_strict_tensor_thresholds() -> None:
    """The fault rollout must stay strict while measured post-fault floors start later."""
    mode = MODES["dp2_cp2_real_rollout_dense"]

    assert _diff_thresholds_for_rollout(mode, _FAULT_ROLLOUT_ID) is _DIFF_THRESHOLDS
    assert _diff_thresholds_for_rollout(mode, _FIRST_POST_FAULT_ROLLOUT_ID) is _POST_FAULT_DIFF_THRESHOLDS


def test_baseline_uses_fault_tolerant_topology_without_fault_actions() -> None:
    """The baseline must isolate fault execution while retaining the target topology."""
    args = shlex.split(
        _build_baseline_args(
            MODES["dp2_cp2_real_rollout_dense"],
            "/tmp/baseline/phase_b",
            enable_dumper=False,
        )
    )

    assert "--use-fault-tolerance" in args
    assert "--ft-components" in args
    assert "--ci-ft-test-actions" not in args
    assert "--ci-inject-rollout-data-path" not in args


def test_final_event_analysis_failure_fails_the_comparison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A witness issue from the final rollout must fail the post-run semantic check."""
    calls: list[Path] = []

    def fake_run_analysis(event_dir: Path) -> list[object]:
        calls.append(event_dir)
        return ["witness mismatch"] if event_dir.parts[-3:-1] == ("target", "phase_b") else []

    monkeypatch.setattr(scenario_with_failure, "run_analysis", fake_run_analysis)

    with pytest.raises(AssertionError, match="target/phase_b.*witness mismatch"):
        _assert_event_analysis(str(tmp_path))

    assert calls == [
        tmp_path / "baseline" / "phase_a" / "events",
        tmp_path / "baseline" / "phase_b" / "events",
        tmp_path / "target" / "phase_a" / "events",
        tmp_path / "target" / "phase_b" / "events",
    ]


def test_fake_rollout_does_not_inject_recorded_data() -> None:
    """Fake-rollout scenarios must keep using their generated deterministic fixtures."""
    args = _build_target_args(
        MODES["dp2_cp2_pp2"],
        "/tmp/target/phase_b",
        enable_dumper=False,
    )

    assert "--ci-inject-rollout-data-start-rollout-id" not in shlex.split(args)
