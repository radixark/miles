from functools import partial
from unittest.mock import Mock

import prometheus_client
import pytest
from tests.fast.utils.observability_fixtures import _TrackingConfig

from miles.utils.tracking_utils.prometheus_utils import _PrometheusCollector


class TestTypedCollectorRunName:
    @pytest.mark.parametrize(
        ("explicit", "group", "expected"),
        [
            ("prometheus-run", "wandb-run", "prometheus-run"),
            (None, "wandb-run", "wandb-run"),
            (None, None, "miles_training"),
        ],
    )
    def test_real_collector_publishes_the_selected_run_label(
        self, monkeypatch: pytest.MonkeyPatch, explicit: str | None, group: str | None, expected: str
    ) -> None:
        """Real gauges expose explicit, W&B fallback, or default run names from typed configuration."""
        registry = prometheus_client.CollectorRegistry()
        start_server = Mock()
        monkeypatch.setattr(prometheus_client, "start_http_server", start_server)
        monkeypatch.setattr(prometheus_client, "Gauge", partial(prometheus_client.Gauge, registry=registry))
        monkeypatch.setattr("ray._private.services.get_node_ip_address", lambda: "127.0.0.1")
        config = _TrackingConfig(prometheus_run_name=explicit, wandb_group=group, prometheus_port=19321)

        collector = _PrometheusCollector(args=config)
        collector.update({"train/loss": 1.5})
        collector.update({"train/loss": 2.5})

        assert registry.get_sample_value("miles_metric_train_loss", labels={"run_name": expected}) == 2.5
        start_server.assert_called_once_with(19321)
