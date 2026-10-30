import argparse
import logging

import pytest

from miles.dashboard.args import collector_config_from_args, validate_dashboard_args
from miles.dashboard.sglang_scraper import DEFAULT_METRIC_WHITELIST


@pytest.fixture(autouse=True)
def no_ambient_settings(monkeypatch):
    for name in (
        "FLUSH_INTERVAL",
        "GPU_SAMPLE_INTERVAL",
        "SGLANG_SCRAPE_INTERVAL",
        "SGLANG_METRICS",
        "FORWARD_PROMETHEUS",
    ):
        monkeypatch.delenv(f"MILES_DASHBOARD_{name}", raising=False)


def full_args(**overrides):
    args = argparse.Namespace()
    args.dump_details = "/tmp/dump"
    args.cluster_backend = "ray"
    args.use_rollout_entropy = True
    args.wandb_group = "my-run"
    args.colocate = True
    for key, value in overrides.items():
        setattr(args, key, value)
    return vars(args)


def test_defaults():
    config = collector_config_from_args(full_args(), start_ts=0.0)
    assert config.flush_interval_seconds == 5.0
    assert config.gpu_sample_interval_seconds == 1.0
    assert config.scrape_interval_seconds == 2.0
    assert config.forward_prometheus is False


def test_validate_requires_dump_details():
    args = full_args(dump_details=None)
    with pytest.raises(ValueError, match="dump-details"):
        validate_dashboard_args(args)


def test_validate_warns_without_entropy(caplog):
    with caplog.at_level(logging.WARNING):
        validate_dashboard_args(full_args(use_rollout_entropy=False))
    assert any("use-rollout-entropy" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        validate_dashboard_args(full_args())
    assert caplog.records == []


def test_validate_refuses_a_kubernetes_run():
    with pytest.raises(ValueError, match="kubernetes"):
        validate_dashboard_args(full_args(cluster_backend="kubernetes"))


def test_collector_config_from_args():
    config = collector_config_from_args(full_args(), start_ts=123.0)
    assert config.dashboard_dir == "/tmp/dump/dashboard"
    assert config.run_name == "my-run"
    assert config.start_ts == 123.0
    assert config.metric_whitelist == DEFAULT_METRIC_WHITELIST
    assert config.args_snapshot["colocate"] is True
    assert "hf_checkpoint" not in config.args_snapshot  # absent arguments are skipped


def test_collector_config_whitelist_override_and_run_name_fallback(monkeypatch):
    args = full_args(wandb_group=None)
    monkeypatch.setenv("MILES_DASHBOARD_SGLANG_METRICS", "sglang_gen_throughput,sglang_token_usage")
    config = collector_config_from_args(args, start_ts=0.0)
    assert config.metric_whitelist == ("sglang_gen_throughput", "sglang_token_usage")
    assert config.run_name == "miles-run"
