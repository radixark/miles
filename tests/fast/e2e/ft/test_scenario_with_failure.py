import json
import shlex

from tests.e2e.ft.conftest_ft.modes import MODES
from tests.e2e.ft.conftest_ft.scenario_with_failure import (
    _DIFF_THRESHOLDS,
    _FAULT_ROLLOUT_ID,
    _FIRST_INJECTED_ROLLOUT_ID,
    _FIRST_POST_FAULT_ROLLOUT_ID,
    _POST_FAULT_DIFF_THRESHOLDS,
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


def test_comparison_uses_deterministic_collectives_without_changing_baseline_topology() -> None:
    """Both sides must share deterministic collectives while baseline remains normal DP."""
    mode = MODES["dp2_cp2_real_rollout_dense"]
    baseline_args = shlex.split(_build_baseline_args(mode, "/tmp/baseline/phase_b", enable_dumper=False))
    target_args = shlex.split(_build_target_args(mode, "/tmp/target/phase_b", enable_dumper=False))

    for args in (baseline_args, target_args):
        assert "--deterministic-mode" in args
        assert "--debug-deterministic-collective" in args
    assert "--use-fault-tolerance" not in baseline_args
    assert "--use-fault-tolerance" in target_args


def test_fake_rollout_does_not_inject_recorded_data() -> None:
    """Fake-rollout scenarios must keep using their generated deterministic fixtures."""
    args = _build_target_args(
        MODES["dp2_cp2_pp2"],
        "/tmp/target/phase_b",
        enable_dumper=False,
    )

    tokens = shlex.split(args)

    assert "--ci-inject-rollout-data-start-rollout-id" not in tokens
    assert "--debug-deterministic-collective" not in tokens
