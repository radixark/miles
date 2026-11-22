import sys
from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.utils.tracking_utils.mlflow_utils import _init_mlflow_primary


class TestPrimaryTrackingIdentity:
    def test_primary_tracking_uses_rank_zero_without_a_worker_rank(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MLflow primary tracking works with launcher configuration that has no worker rank."""
        started: list[dict[str, object]] = []
        logged: list[dict[str, str]] = []
        run = SimpleNamespace(info=SimpleNamespace(run_id="run-id"))
        sdk = SimpleNamespace(
            start_run=lambda **kwargs: started.append(kwargs) or run,
            log_params=logged.append,
        )
        monkeypatch.setitem(sys.modules, "mlflow", sdk)
        monkeypatch.setenv("SLURM_JOB_ID", "job-7")
        args = Namespace(mlflow_run_name=None, wandb_group="group")

        _init_mlflow_primary(args, experiment_name="experiment")

        assert started == [{"run_name": "group", "tags": {"slurm_job_id": "job-7", "rank": "0"}}]
        assert args.mlflow_run_id == "run-id"
        assert logged[0]["env_vars.SLURM_JOB_ID"] == "job-7"
