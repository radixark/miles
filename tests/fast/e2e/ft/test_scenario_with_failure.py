import shlex

from tests.e2e.ft.conftest_ft.modes import MODES
from tests.e2e.ft.conftest_ft.scenario_with_failure import (
    _WITH_FAILURE_ACTIONS,
    NUM_PHASE_A_STEPS,
    _build_baseline_args,
    _build_target_args,
)

_REAL_ROLLOUT_MODE = "dp2_cp2_real_rollout_dense"
_DEBUG_DATA_MODE = "dp2_cp2_tp2_ep2"


def _option_value(tokens: list[str], option: str) -> str:
    return tokens[tokens.index(option) + 1]


def _phase_b_tokens(mode_name: str, side: str) -> list[str]:
    build = _build_baseline_args if side == "baseline" else _build_target_args
    return shlex.split(build(MODES[mode_name], f"/tmp/{side}/phase_b", enable_dumper=False))


def test_real_rollout_sides_share_deterministic_collective_and_inactive_clipping() -> None:
    """Both sides of the live-rollout comparison must fold reductions identically with clipping inactive."""
    for side in ("baseline", "target"):
        tokens = _phase_b_tokens(_REAL_ROLLOUT_MODE, side)
        assert "--debug-deterministic-collective" in tokens
        assert _option_value(tokens, "--clip-grad") == "10.0"


def test_debug_data_modes_keep_the_production_collective() -> None:
    """Modes without live rollout keep the production reduction path and default clipping."""
    for side in ("baseline", "target"):
        tokens = _phase_b_tokens(_DEBUG_DATA_MODE, side)
        assert "--debug-deterministic-collective" not in tokens
        assert "--clip-grad" not in tokens


def test_injection_starts_after_the_fault_rollout() -> None:
    """Only post-fault rollouts may inject baseline data; the fault rollout trains live samples."""
    fault_rollout_ids = {action["at_rollout"] for action in _WITH_FAILURE_ACTIONS}
    target_tokens = _phase_b_tokens(_REAL_ROLLOUT_MODE, "target")
    start = int(_option_value(target_tokens, "--ci-inject-rollout-data-start-rollout-id"))

    assert fault_rollout_ids == {NUM_PHASE_A_STEPS + 1}
    assert start == NUM_PHASE_A_STEPS + 2
    assert "--ci-inject-rollout-data-path" not in _phase_b_tokens(_REAL_ROLLOUT_MODE, "baseline")
