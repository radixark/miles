from unittest.mock import Mock

import pytest

from miles.utils.args.configs.wandb import WandbConfig
from miles.utils.tracking_utils import wandb_utils


class TestTypedSecondaryTrackingDisabled:
    def test_missing_run_id_skips_all_secondary_sdk_setup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A typed configuration without a primary run ID must not contact the tracking SDK."""
        init, login, define_metric = Mock(), Mock(), Mock()
        monkeypatch.setattr(wandb_utils.wandb, "init", init)
        monkeypatch.setattr(wandb_utils.wandb, "login", login)
        monkeypatch.setattr(wandb_utils.wandb, "define_metric", define_metric)
        config = WandbConfig(wandb_run_id=None, wandb_key="unused-key", wandb_mode="offline")

        wandb_utils.init_wandb_secondary(config)

        init.assert_not_called()
        login.assert_not_called()
        define_metric.assert_not_called()
