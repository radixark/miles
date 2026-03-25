from __future__ import annotations

import socket
import time
from argparse import Namespace
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import ray

import miles.utils.prometheus_utils as prometheus_mod
from miles.utils.prometheus_utils import _PrometheusCollector, get_prometheus, init_prometheus


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _fetch_metrics(port: int) -> str:
    return httpx.get(f"http://localhost:{port}/metrics", timeout=5).text


def _make_args(port: int, run_name: str = "test-run") -> Namespace:
    return Namespace(
        prometheus_port=port,
        prometheus_run_name=run_name,
        wandb_group=None,
    )


def _wait_for_server(port: int, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            _fetch_metrics(port)
            return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError(f"Prometheus HTTP server on port {port} not ready after {timeout}s")


@pytest.fixture(scope="module")
def ray_context() -> Any:
    ray.init(num_cpus=2)
    yield
    ray.shutdown()


@pytest.fixture(scope="module")
def prometheus_server(ray_context: Any) -> int:
    port = _find_free_port()
    init_prometheus(_make_args(port=port), start_server=True)
    _wait_for_server(port)
    return port


def test_init_creates_named_actor_and_ping(prometheus_server: int) -> None:
    """Named actor exists and responds to ping."""
    actor = ray.get_actor("miles_prometheus_collector")
    assert actor is not None
    assert ray.get(actor.ping.remote()) is True


def test_init_start_server_false_discovers_actor(prometheus_server: int) -> None:
    """start_server=False finds the existing named actor."""
    original = prometheus_mod._collector_handle
    prometheus_mod._collector_handle = None

    init_prometheus(_make_args(port=0, run_name="ignored"), start_server=False)
    assert get_prometheus() is not None

    prometheus_mod._collector_handle = original


def test_set_gauge_and_update_visible_on_http(prometheus_server: int) -> None:
    """set_gauge, update, overwrite, and special-char sanitization all visible via HTTP."""
    handle = get_prometheus()

    # Step 1: basic set_gauge
    ray.get(handle.set_gauge.remote("test_sg", 42.0))

    # Step 2: update with prefix
    ray.get(handle.update.remote({"loss": 0.5, "mfu": 0.3}))

    # Step 3: overwrite
    ray.get(handle.set_gauge.remote("test_sg", 99.0))

    # Step 4: special char sanitization
    ray.get(handle.update.remote({"train/loss": 1.0, "grad-norm": 2.0, "lr@step": 3.0}))

    # Step 5: non-numeric skipped
    ray.get(handle.update.remote({"good_val": 7.0, "bad_val": "hello"}))

    body = _fetch_metrics(prometheus_server)

    assert 'test_sg{run_name="test-run"} 99.0' in body
    assert 'miles_metric_loss{run_name="test-run"} 0.5' in body
    assert 'miles_metric_mfu{run_name="test-run"} 0.3' in body
    assert 'miles_metric_train_loss{run_name="test-run"} 1.0' in body
    assert 'miles_metric_grad_norm{run_name="test-run"} 2.0' in body
    assert 'miles_metric_lr_at_step{run_name="test-run"} 3.0' in body
    assert "miles_metric_good_val" in body
    assert "miles_metric_bad_val" not in body


def test_extra_labels_merged_and_independent(prometheus_server: int) -> None:
    """extra_labels are merged with run_name; different label values coexist."""
    handle = get_prometheus()

    ray.get(handle.set_gauge.remote("test_cell", 1.0, extra_labels={"cell_id": "c0"}))
    ray.get(handle.set_gauge.remote("test_cell", 0.0, extra_labels={"cell_id": "c1"}))

    body = _fetch_metrics(prometheus_server)

    assert 'test_cell{cell_id="c0",run_name="test-run"} 1.0' in body
    assert 'test_cell{cell_id="c1",run_name="test-run"} 0.0' in body


class TestRunNameResolution:
    def test_prefers_prometheus_run_name(self) -> None:
        with patch("prometheus_client.start_http_server"), patch(
            "miles.utils.prometheus_utils.get_current_node_ip", return_value="127.0.0.1"
        ):
            collector = _PrometheusCollector(
                Namespace(prometheus_port=0, prometheus_run_name="prom", wandb_group="wandb")
            )
        assert collector._run_name == "prom"

    def test_falls_back_to_wandb_group(self) -> None:
        with patch("prometheus_client.start_http_server"), patch(
            "miles.utils.prometheus_utils.get_current_node_ip", return_value="127.0.0.1"
        ):
            collector = _PrometheusCollector(
                Namespace(prometheus_port=0, prometheus_run_name=None, wandb_group="wandb")
            )
        assert collector._run_name == "wandb"

    def test_defaults_to_miles_training(self) -> None:
        with patch("prometheus_client.start_http_server"), patch(
            "miles.utils.prometheus_utils.get_current_node_ip", return_value="127.0.0.1"
        ):
            collector = _PrometheusCollector(Namespace(prometheus_port=0, prometheus_run_name=None, wandb_group=None))
        assert collector._run_name == "miles_training"
