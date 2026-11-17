import argparse

import pytest
from pydantic import ConfigDict

from miles.backends.sglang_utils.arguments import collect_eval_sglang_overrides
from miles.utils.args.schema import BaseConfig
from miles.utils.args.utils import config_values
from miles.utils.env_report.collector import _dump_args
from miles.utils.tracking_utils import mlflow_utils, wandb_utils


class _LoggedConfig(BaseConfig):
    model_config = ConfigDict(extra="allow")
    env_report: str | None = None


class TestConfigValuesConsumers:
    @pytest.mark.parametrize("typed", [False, True])
    def test_tracking_and_environment_reports_retain_fields_outside_pydantic_storage(
        self, monkeypatch: pytest.MonkeyPatch, typed: bool
    ) -> None:
        """Namespace and typed configurations preserve custom values through real report consumers."""
        values = {"env_report": None, "nested": {"lr": 0.01}, "eval_sglang_dp_size": 3, "custom_value": 7}
        config = _LoggedConfig(**values) if typed else argparse.Namespace(**values)
        monkeypatch.setenv("SLURM_JOB_ID", "job-123")

        wandb_config = wandb_utils._compute_config_for_logging(config)
        mlflow_config = mlflow_utils._compute_config_for_logging(config)
        report = _dump_args(config)
        eval_overrides = collect_eval_sglang_overrides(config)
        extracted = config_values(config)
        extracted.pop("custom_value")
        wandb_config["nested"]["lr"] = 0.9

        assert wandb_config["custom_value"] == 7
        assert wandb_config["env_vars"] == {"SLURM_JOB_ID": "job-123"}
        assert mlflow_config["custom_value"] == "7"
        assert mlflow_config["nested.lr"] == "0.01"
        assert report.values["custom_value"] == 7
        assert report.values["nested"] == {"lr": 0.01}
        assert eval_overrides == {"dp_size": 3}
        assert config_values(config) == values


class TestSecondaryTypedTracking:
    def test_secondary_wandb_submits_custom_typed_fields_to_the_sdk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Secondary tracking submits Pydantic extras rather than only the declared storage dictionary."""
        from tests.fast.fixtures.sglang_config_fixtures import make_sglang_config

        calls: list[dict[str, object]] = []
        config = _LoggedConfig(
            wandb_run_id="run",
            wandb_mode="offline",
            wandb_key=None,
            wandb_host=None,
            wandb_dir=None,
            wandb_team=None,
            wandb_project="project",
            custom_value=7,
            sglang=make_sglang_config(enable_metrics=False),
        )
        monkeypatch.setattr(wandb_utils.wandb, "init", lambda **kwargs: calls.append(kwargs))
        monkeypatch.setattr(wandb_utils.wandb, "define_metric", lambda *args, **kwargs: None)

        wandb_utils.init_wandb_secondary(config)

        assert calls[0]["config"]["custom_value"] == 7
        assert calls[0]["config"]["wandb_run_id"] == "run"
        assert config.custom_value == 7
