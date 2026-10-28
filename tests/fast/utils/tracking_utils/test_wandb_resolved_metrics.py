import pytest
from tests.fast.fixtures.sglang_config_fixtures import make_sglang_config
from tests.fast.utils.tracking_utils.test_wandb_utils import _args

from miles.utils.tracking_utils import wandb_utils


class TestResolvedWandbMetrics:
    @pytest.mark.parametrize("enabled", [False, True])
    def test_secondary_metrics_endpoint_follows_resolved_sglang_settings(
        self, monkeypatch: pytest.MonkeyPatch, enabled: bool
    ) -> None:
        """The secondary WandB SDK receives an engine metrics endpoint only when resolved metrics are enabled."""
        calls: list[dict[str, object]] = []
        args = _args(wandb_run_id="run", sglang=make_sglang_config(enable_metrics=enabled))
        monkeypatch.setattr(wandb_utils.wandb, "init", lambda **kwargs: calls.append(kwargs))
        monkeypatch.setattr(wandb_utils.wandb, "define_metric", lambda *args, **kwargs: None)

        wandb_utils.init_wandb_secondary(args, router_addr="http://router:30000")

        endpoints = calls[0]["settings"].x_stats_open_metrics_endpoints or {}
        assert endpoints == ({"sgl_engine": "http://router:30000/engine_metrics"} if enabled else {})
