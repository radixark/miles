import json
import shlex

from tests.e2e.ft.conftest_ft.modes import MODES
from tests.e2e.ft.conftest_ft.scenario_with_failure import (
    FAULT_ROLLOUT_ID,
    FIRST_INJECTED_ROLLOUT_ID,
    _build_target_args,
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

    assert FIRST_INJECTED_ROLLOUT_ID == FAULT_ROLLOUT_ID
    assert int(_option_value(args, "--ci-inject-rollout-data-start-rollout-id")) == FAULT_ROLLOUT_ID
    assert {action["at_rollout"] for action in actions} == {FAULT_ROLLOUT_ID}


def test_fake_rollout_does_not_inject_recorded_data() -> None:
    """Fake-rollout scenarios must keep using their generated deterministic fixtures."""
    args = _build_target_args(
        MODES["dp2_cp2_pp2"],
        "/tmp/target/phase_b",
        enable_dumper=False,
    )

    assert "--ci-inject-rollout-data-start-rollout-id" not in shlex.split(args)
