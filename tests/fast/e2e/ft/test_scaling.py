import shlex
from pathlib import Path

import pytest

from tests.e2e.ft.conftest_ft.scaling import _build_train_args, compute_scaling_mode


class TestScalingHealthChecks:
    def test_short_scaling_run_probes_trainer_before_admitting_resizes(self) -> None:
        """A short scaling run must observe a real heartbeat before its resize window closes."""
        tokens = shlex.split(_build_train_args(mode=compute_scaling_mode(("train",)), dump_dir=Path("/dumps/scaling")))
        option = "--trainer-heartbeat-checker-first-wait"
        assert tokens[tokens.index(option) + 1] == "0"
        assert "--trainer-heartbeat-checker-interval" not in tokens


class TestScalingObjectStore:
    def test_shrinking_one_cell_retains_a_second_rollout_data_replica(self) -> None:
        """Removing one storage contributor must leave a rollout data replica."""
        tokens = shlex.split(_build_train_args(mode=compute_scaling_mode(("train",)), dump_dir=Path("/dumps/scaling")))
        option = "--mooncake-replica-num"
        assert tokens[tokens.index(option) + 1] == "2"


class TestInferenceScalingWeightTransfer:
    def test_rollout_scaling_uses_partial_target_weight_transfer(self) -> None:
        """Rollout cell replacement requires weight transfer that can exclude a failed target."""
        tokens = shlex.split(
            _build_train_args(mode=compute_scaling_mode(("rollout",)), dump_dir=Path("/dumps/scaling"))
        )
        option = "--update-weight-transfer-mode"
        assert tokens[tokens.index(option) + 1] == "p2p"
        assert "--colocate" not in tokens
        assert "--sglang-remote-instance-weight-loader-start-seed-via-transfer-engine" in tokens


class TestScalingTrainerStartup:
    @pytest.mark.parametrize(("component", "expected"), [("train", "2"), ("rollout", "1")])
    def test_startup_waits_for_the_deployed_trainer_cells(self, component: str, expected: str) -> None:
        """Ordinary data parallel workers share one cell when trainer FT is disabled."""
        tokens = shlex.split(
            _build_train_args(mode=compute_scaling_mode((component,)), dump_dir=Path("/dumps/scaling"))
        )
        option = "--trainer-init-expected-num-cells"
        assert tokens[tokens.index(option) + 1] == expected


class TestScalingRolloutHealthChecks:
    def test_short_rollouts_refresh_health_before_the_resize_window_closes(self) -> None:
        """Weight updates must not keep engine readiness unknown throughout a short resize window."""
        tokens = shlex.split(
            _build_train_args(mode=compute_scaling_mode(("rollout",)), dump_dir=Path("/dumps/scaling"))
        )
        option = "--rollout-health-check-interval"
        assert float(tokens[tokens.index(option) + 1]) == 1.0
