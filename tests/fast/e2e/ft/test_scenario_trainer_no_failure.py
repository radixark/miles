import shlex

from tests.e2e.ft.conftest_ft.modes import MODES
from tests.e2e.ft.conftest_ft.scenario_trainer_no_failure import _build_baseline_args, _build_target_args

_REAL_ROLLOUT_MODE = "kill_train__dp2_cp2__moe_5layer"
_FAKE_ROLLOUT_MODE = "kill_train__dp2_cp2_tp2_ep2__fake_rollout__moe_5layer"


def _option_value(tokens: list[str], option: str) -> str:
    return tokens[tokens.index(option) + 1]


def _tokens(mode_name: str, side: str) -> list[str]:
    build = _build_baseline_args if side == "baseline" else _build_target_args
    return shlex.split(build(MODES[mode_name], f"/dumps/run/{side}", enable_dumper=False))


def test_real_rollout_sides_share_deterministic_collective_and_inactive_clipping() -> None:
    """Both sides must fold reductions identically so live samples match without replaying data."""
    for side in ("baseline", "target"):
        tokens = _tokens(_REAL_ROLLOUT_MODE, side)
        assert "--debug-deterministic-collective" in tokens
        assert _option_value(tokens, "--clip-grad") == "10.0"
        assert "--ci-inject-rollout-data-path" not in tokens


def test_fake_rollout_modes_keep_the_production_collective() -> None:
    """Modes without live rollout keep the production reduction path and default clipping."""
    for side in ("baseline", "target"):
        tokens = _tokens(_FAKE_ROLLOUT_MODE, side)
        assert "--debug-deterministic-collective" not in tokens
        assert "--clip-grad" not in tokens
