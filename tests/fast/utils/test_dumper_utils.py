import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from miles.utils import dumper_utils
from miles.utils.dumper_utils import DumperMegatronUtil, DumperPhase


class TestWrapForwardStepWithStepping:

    @pytest.fixture()
    def setup(self):
        inner = MagicMock(return_value=("output", "loss_fn"))
        wrapped = dumper_utils._wrap_forward_step_with_stepping(inner)
        mock_dumper = MagicMock()
        return inner, wrapped, mock_dumper

    @pytest.mark.parametrize(("n_calls", "expected_steps"), [(1, 0), (2, 1), (5, 4)])
    def test_step_called_n_minus_1_times(self, setup, n_calls: int, expected_steps: int) -> None:
        _inner, wrapped, mock_dumper = setup
        with patch("miles.utils.dumper_utils.dumper", mock_dumper):
            for _ in range(n_calls):
                wrapped("iter", "model")
        assert mock_dumper.step.call_count == expected_steps

    def test_passes_args_and_returns_result(self, setup) -> None:
        inner, wrapped, mock_dumper = setup
        with patch("miles.utils.dumper_utils.dumper", mock_dumper):
            result = wrapped("my_iter", "my_model", extra=True)
        inner.assert_called_once_with("my_iter", "my_model", extra=True)
        assert result == ("output", "loss_fn")


def test_sglang_env_includes_startup_dumper_settings() -> None:
    args = SimpleNamespace(
        dumper_enable=False,
        dumper_inference=["enable=true", "non_intrusive_mode=all"],
        dumper_source_patcher_config_inference="/tmp/patcher.yaml",
    )

    env = dumper_utils.get_sglang_env(args)

    assert env == {
        "DUMPER_SERVER_PORT": "reuse",
        "DUMPER_NON_INTRUSIVE_MODE": "all",
        "DUMPER_SOURCE_PATCHER_CONFIG": "/tmp/patcher.yaml",
    }


def test_sglang_env_disabled_when_inference_phase_disabled() -> None:
    args = SimpleNamespace(
        dumper_enable=False,
        dumper_inference=["non_intrusive_mode=all"],
        dumper_source_patcher_config_inference=None,
    )

    assert dumper_utils.get_sglang_env(args) == {}


def _make_args(dump_dir: Path, *, enable: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        dumper_enable=enable,
        dumper_dir=str(dump_dir),
        dumper_fwd_bwd=[],
        dumper_fwd_only=[],
    )


class TestDumperMegatronUtilConfigure:
    """Per-rollout dump directory layout and the rollout-0 parent-wipe gate."""

    @pytest.fixture()
    def parallel_state(self):
        """Single-rank, DP-rank-0 parallel state so cleanup and output run locally."""
        state = SimpleNamespace(
            effective_dp=SimpleNamespace(rank=0),
            indep_dp=SimpleNamespace(rank=0, group=None),
        )
        with patch("miles.utils.dumper_utils.get_parallel_state", return_value=state):
            yield state

    def _configure(self, args: SimpleNamespace, *, phase: DumperPhase, rollout_id: int) -> bool:
        with (
            patch("miles.utils.dumper_utils.dumper") as mock_dumper,
            patch("miles.utils.dumper_utils.dist") as mock_dist,
        ):
            mock_dist.is_initialized.return_value = False
            enabled = DumperMegatronUtil._configure(args, phase=phase, rollout_id=rollout_id)
            return enabled, mock_dumper

    def test_disabled_phase_returns_false(self, tmp_path: Path, parallel_state) -> None:
        """A phase whose override has enable=false short-circuits without configuring the dumper."""
        args = _make_args(tmp_path, enable=False)
        enabled, mock_dumper = self._configure(args, phase=DumperPhase.FWD_BWD, rollout_id=0)
        assert enabled is False
        mock_dumper.configure.assert_not_called()

    def test_exp_name_includes_phase_and_rollout_id(self, tmp_path: Path, parallel_state) -> None:
        """exp_name is '{phase}/rollout_{rollout_id}' so each rollout dumps to its own subdirectory."""
        args = _make_args(tmp_path)
        enabled, mock_dumper = self._configure(args, phase=DumperPhase.FWD_BWD, rollout_id=3)
        assert enabled is True
        config_kwargs = mock_dumper.configure.call_args.kwargs
        assert config_kwargs["exp_name"] == "fwd_bwd/rollout_3"

    def test_rollout_zero_configure_wipes_phase_parent_dir(self, tmp_path: Path, parallel_state) -> None:
        """Configuring at rollout 0 removes the whole phase parent dir (stale rollouts from a prior run)."""
        args = _make_args(tmp_path)
        phase_parent = tmp_path / "fwd_bwd"
        stale_rollout = phase_parent / "rollout_9"
        stale_rollout.mkdir(parents=True)
        (stale_rollout / "stale.pt").write_text("stale")

        self._configure(args, phase=DumperPhase.FWD_BWD, rollout_id=0)

        assert not phase_parent.exists()

    def test_second_configure_does_not_wipe_parent_only_own_subdir(self, tmp_path: Path, parallel_state) -> None:
        """The second rollout keeps sibling rollout dirs and only cleans/recreates its own subdir."""
        args = _make_args(tmp_path)
        self._configure(args, phase=DumperPhase.FWD_BWD, rollout_id=0)

        phase_parent = tmp_path / "fwd_bwd"
        rollout_0 = phase_parent / "rollout_0"
        rollout_0.mkdir(parents=True)
        (rollout_0 / "keep.pt").write_text("rollout_0 data")
        rollout_1 = phase_parent / "rollout_1"
        rollout_1.mkdir(parents=True)
        (rollout_1 / "stale.pt").write_text("old rollout_1 data")

        self._configure(args, phase=DumperPhase.FWD_BWD, rollout_id=1)

        assert (rollout_0 / "keep.pt").exists()
        assert not rollout_1.exists()

    def test_respawned_process_at_nonzero_rollout_preserves_phase_parent_dir(
        self, tmp_path: Path, parallel_state
    ) -> None:
        """A fresh process configuring mid-run (rollout > 0, e.g. after a respawn) keeps earlier rollout dumps."""
        args = _make_args(tmp_path)
        phase_parent = tmp_path / "fwd_bwd"
        rollout_0 = phase_parent / "rollout_0"
        rollout_0.mkdir(parents=True)
        (rollout_0 / "keep.pt").write_text("rollout_0 data")

        self._configure(args, phase=DumperPhase.FWD_BWD, rollout_id=2)

        assert (rollout_0 / "keep.pt").exists()

    def test_rollout_zero_wipe_only_touches_own_phase_dir(self, tmp_path: Path, parallel_state) -> None:
        """A rollout-0 wipe of one phase leaves the other phase's dumps intact."""
        args = _make_args(tmp_path)
        other_phase_file = tmp_path / "fwd_bwd" / "rollout_0" / "keep.pt"
        other_phase_file.parent.mkdir(parents=True)
        other_phase_file.write_text("fwd_bwd data")

        self._configure(args, phase=DumperPhase.FWD_ONLY, rollout_id=0)

        assert other_phase_file.exists()


class TestBarrierAfterDumpDirCleanup:
    def test_cross_cell_barrier_abort_does_not_raise(self) -> None:
        """A peer death aborts the cross-cell PG mid-barrier; the survivor continues instead of erroring."""
        group = MagicMock()
        group.barrier.side_effect = RuntimeError("NCCL communicator was aborted on rank 1")
        state = SimpleNamespace(indep_dp=SimpleNamespace(rank=1, size=2, group=group, debug_info={"quorum": 0}))

        with (
            patch("miles.utils.dumper_utils.get_parallel_state", return_value=state),
            patch("miles.utils.dumper_utils.dist") as mock_dist,
        ):
            mock_dist.is_initialized.return_value = False
            dumper_utils._barrier_after_dump_dir_cleanup()

        group.barrier.assert_called_once()


class TestCleanupDumpDir:
    """Best-effort dump-dir wipe gated on global rank 0 + indep-DP (cell) rank 0."""

    def test_rank0_cell0_existing_dir_is_removed(self, tmp_path: Path) -> None:
        """Global rank 0 of cell 0 deletes the dump dir when it exists."""
        dump_dir = tmp_path / "fwd_bwd"
        dump_dir.mkdir()
        (dump_dir / "stale.pt").write_text("stale")

        with patch("miles.utils.dumper_utils._get_rank", return_value=0):
            dumper_utils._cleanup_dump_dir(dump_dir, indep_dp_rank=0)

        assert not dump_dir.exists()

    def test_rmtree_oserror_is_swallowed_and_warns(self, tmp_path: Path, caplog) -> None:
        """A failing rmtree (e.g. NFS stale handle) is logged as a warning and does not propagate."""
        dump_dir = tmp_path / "fwd_bwd"
        dump_dir.mkdir()

        with (
            patch("miles.utils.dumper_utils._get_rank", return_value=0),
            patch("miles.utils.dumper_utils.shutil.rmtree", side_effect=OSError("Directory not empty")),
            caplog.at_level(logging.WARNING, logger="miles.utils.dumper_utils"),
        ):
            dumper_utils._cleanup_dump_dir(dump_dir, indep_dp_rank=0)

        assert any("dump dir cleanup failed" in record.message for record in caplog.records)
        assert dump_dir.exists()

    def test_nonzero_rank_does_not_delete(self, tmp_path: Path) -> None:
        """A non-zero global rank never calls rmtree, so the dir survives for rank 0 to own."""
        dump_dir = tmp_path / "fwd_bwd"
        dump_dir.mkdir()

        with (
            patch("miles.utils.dumper_utils._get_rank", return_value=1),
            patch("miles.utils.dumper_utils.shutil.rmtree") as mock_rmtree,
        ):
            dumper_utils._cleanup_dump_dir(dump_dir, indep_dp_rank=0)

        mock_rmtree.assert_not_called()
        assert dump_dir.exists()

    def test_nonzero_indep_dp_rank_does_not_delete(self, tmp_path: Path) -> None:
        """Only cell 0 (indep_dp_rank 0) deletes; other cells leave the shared dir intact."""
        dump_dir = tmp_path / "fwd_bwd"
        dump_dir.mkdir()

        with (
            patch("miles.utils.dumper_utils._get_rank", return_value=0),
            patch("miles.utils.dumper_utils.shutil.rmtree") as mock_rmtree,
        ):
            dumper_utils._cleanup_dump_dir(dump_dir, indep_dp_rank=1)

        mock_rmtree.assert_not_called()
        assert dump_dir.exists()

    def test_missing_dir_is_noop(self, tmp_path: Path) -> None:
        """When the dump dir does not exist, cleanup is a silent no-op (no rmtree, no error)."""
        dump_dir = tmp_path / "does_not_exist"

        with (
            patch("miles.utils.dumper_utils._get_rank", return_value=0),
            patch("miles.utils.dumper_utils.shutil.rmtree") as mock_rmtree,
        ):
            dumper_utils._cleanup_dump_dir(dump_dir, indep_dp_rank=0)

        mock_rmtree.assert_not_called()
        assert not dump_dir.exists()
