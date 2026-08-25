import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.fast.utils.env_report.conftest import make_args

from miles.utils.tracking_utils import wandb_utils
from miles.utils.tracking_utils.wandb_utils import _compute_config_for_logging


class TestPrimaryWandbInit:
    def test_primary_wandb_init_resumes_a_preassigned_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A replacement primary resumes the explicitly assigned WandB run."""
        init_calls: list[dict[str, object]] = []
        args = make_args(
            rank=0,
            use_wandb=True,
            wandb_dir=None,
            wandb_group="group",
            wandb_host=None,
            wandb_key=None,
            wandb_mode=None,
            wandb_project="project",
            wandb_random_suffix=False,
            wandb_run_id="assigned-run",
            wandb_team=None,
        )
        monkeypatch.setattr(wandb_utils.wandb, "init", lambda **kwargs: init_calls.append(kwargs))
        monkeypatch.setattr(wandb_utils.wandb, "define_metric", lambda *args, **kwargs: None)
        monkeypatch.setattr(wandb_utils.wandb, "run", SimpleNamespace(id="assigned-run"), raising=False)

        wandb_utils.init_wandb_primary(args)

        assert init_calls[0]["id"] == "assigned-run"
        assert init_calls[0]["resume"] == "allow"


class TestTheLoggedConfig:
    def test_carries_the_record_of_the_launch_that_started_this_run(self, tmp_path: Path) -> None:
        """The launcher's record is what an algorithm engineer looks for first, and wandb is where they look."""
        record = tmp_path / "launch-1.json"
        record.write_text(json.dumps({"run_id": "260101-000000-000", "worker_argv": ["--lr", "1"]}))

        config = _compute_config_for_logging(make_args(env_report=str(record)))

        assert config["launcher_env_report"]["run_id"] == "260101-000000-000"
        assert config["launcher_env_report"]["worker_argv"] == ["--lr", "1"]

    def test_names_no_launcher_record_when_no_launcher_wrote_one(self) -> None:
        """A run started by hand has no record, and wandb must not grow an empty key for it."""
        assert "launcher_env_report" not in _compute_config_for_logging(make_args())

    def test_survives_a_record_the_shared_disk_lost(self, tmp_path: Path) -> None:
        """Losing the record must cost the run its audit trail, not its tracking."""
        config = _compute_config_for_logging(make_args(env_report=str(tmp_path / "gone.json")))

        assert "launcher_env_report" not in config
