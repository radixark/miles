import json
import shlex

from tests.e2e.ft.conftest_ft.modes import MODES
from tests.e2e.ft.conftest_ft.scenario_with_failure import (
    _DIFF_THRESHOLDS,
    _FAULT_ROLLOUT_ID,
    _build_baseline_args,
    _build_target_args,
)


def _option_value(args: str, option: str) -> str:
    tokens = shlex.split(args)
    return tokens[tokens.index(option) + 1]


def test_real_rollout_trains_on_target_generated_data_with_production_math() -> None:
    """The live target must train its own rollout through the production math path."""
    args = _build_target_args(
        MODES["dp2_cp2_real_rollout_dense"],
        "/tmp/target/phase_b",
        enable_dumper=False,
    )
    actions = json.loads(_option_value(args, "--ci-ft-test-actions"))

    assert {action["at_rollout"] for action in actions} == {_FAULT_ROLLOUT_ID}
    tokens = shlex.split(args)
    assert "--ci-inject-rollout-data-path" not in tokens
    assert "--ci-inject-rollout-data-start-rollout-id" not in tokens
    assert "--ci-inject-rollout-data-min-match-ratio" not in tokens
    assert "--debug-deterministic-collective" not in tokens
    assert "--clip-grad" not in tokens
    assert "--use-dynamic-batch-size" in tokens
    assert _option_value(args, "--max-tokens-per-gpu") == "32768"


def test_fault_scenario_has_no_post_fault_threshold_exception() -> None:
    """The fault and post-fault rollouts must share the strict tensor thresholds."""
    assert all("3e-3" not in predicate for _, predicate in _DIFF_THRESHOLDS)


def test_baseline_remains_normal_dp_while_target_uses_ft() -> None:
    """The comparison must retain its intentional normal-DP versus FT contract."""
    mode = MODES["dp2_cp2_real_rollout_dense"]
    baseline_args = _build_baseline_args(
        mode,
        "/tmp/baseline/phase_b",
        enable_dumper=False,
    )
    target_args = _build_target_args(
        mode,
        "/tmp/target/phase_b",
        enable_dumper=False,
    )

    baseline_tokens = shlex.split(baseline_args)
    target_tokens = shlex.split(target_args)
    assert "--use-fault-tolerance" not in baseline_tokens
    assert "--ft-components" not in baseline_tokens
    assert "--ci-ft-test-actions" not in baseline_tokens
    assert "--use-fault-tolerance" in target_tokens
    assert "--ft-components" in target_tokens
    assert "--ci-ft-test-actions" in target_tokens
