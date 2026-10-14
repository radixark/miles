from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from tests.e2e.ft.conftest_ft.modes import MODES
from tests.e2e.ft.conftest_ft.scenario_rollout_deterministic import (
    NUM_ROLLOUTS,
    TERMINAL_FAULT_FREE_ROLLOUTS,
    _build_args,
    _rollout_fault_injection_enabled,
)

_BASE = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)


def test_rollout_deterministic_uses_the_shared_deterministic_recipe_without_true_on_policy(tmp_path: Path) -> None:
    """The rollout-FT comparison must use pure deterministic inference rather than true-on-policy."""
    args = _build_args(MODES["kill_rollout__dp4__colocate"], dump_dir=str(tmp_path))

    assert "--sglang-enable-deterministic-inference " in args
    assert "--sglang-attention-backend flashinfer " in args
    assert '"SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "false"' in args
    assert "--rollout-health-check-interval 1.0 " in args
    assert "--deterministic-mode " in args
    assert "--context-parallel-size " not in args
    assert "--true-on-policy-mode" not in args
    assert "--sglang-true-on-policy-contract" not in args
    assert "--true-on-policy-contract" not in args
    assert "--sglang-attention-backend fa3" not in args
    assert "--recompute-logprobs-via-prefill" not in args


def test_rollout_fault_window_closes_before_the_final_rollouts(monkeypatch: pytest.MonkeyPatch) -> None:
    """The deterministic recovery tail admits no new rollout fault."""
    completed_rollouts = [(rollout_id, _BASE) for rollout_id in range(NUM_ROLLOUTS - TERMINAL_FAULT_FREE_ROLLOUTS)]

    def read_completed_rollouts(dump_dir: str) -> list[tuple[int, datetime]]:
        return completed_rollouts

    monkeypatch.setattr(
        "tests.e2e.ft.conftest_ft.scenario_rollout_deterministic.read_rollout_completion_times",
        read_completed_rollouts,
    )

    assert not _rollout_fault_injection_enabled("/dump")


def test_rollout_fault_window_uses_the_latest_completed_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rollout ID watermark, not the number of metric events, starts the recovery tail."""
    completed_rollouts = [
        (NUM_ROLLOUTS - TERMINAL_FAULT_FREE_ROLLOUTS - 1, _BASE),
        (NUM_ROLLOUTS - TERMINAL_FAULT_FREE_ROLLOUTS - 1, _BASE + timedelta(seconds=1)),
    ]

    def read_completed_rollouts(dump_dir: str) -> list[tuple[int, datetime]]:
        return completed_rollouts

    monkeypatch.setattr(
        "tests.e2e.ft.conftest_ft.scenario_rollout_deterministic.read_rollout_completion_times",
        read_completed_rollouts,
    )

    assert not _rollout_fault_injection_enabled("/dump")
