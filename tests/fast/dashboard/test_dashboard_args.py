import argparse
import logging

import pytest

from miles.dashboard.args import add_dashboard_arguments, collector_config_from_args, validate_dashboard_args
from miles.dashboard.sglang_scraper import DEFAULT_METRIC_WHITELIST


def parse(argv):
    parser = argparse.ArgumentParser()
    add_dashboard_arguments(parser)
    return parser.parse_args(argv)


def full_args(**overrides):
    args = parse(["--use-miles-dashboard"])
    args.dump_details = "/tmp/dump"
    args.use_rollout_entropy = True
    args.wandb_group = "my-run"
    args.colocate = True
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_defaults():
    args = parse([])
    assert args.use_miles_dashboard is False
    assert args.dashboard_flush_interval == 5.0
    assert args.dashboard_gpu_sample_interval == 1.0
    assert args.dashboard_sglang_scrape_mode == "auto"
    assert args.dashboard_sglang_metrics is None
    assert args.dashboard_forward_prometheus is False


def test_validate_requires_dump_details():
    args = full_args(dump_details=None)
    with pytest.raises(AssertionError, match="dump-details"):
        validate_dashboard_args(args)


def test_validate_warns_without_entropy(caplog):
    with caplog.at_level(logging.WARNING):
        validate_dashboard_args(full_args(use_rollout_entropy=False))
    assert any("use-rollout-entropy" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        validate_dashboard_args(full_args())
    assert caplog.records == []


def test_validate_disabled_checks_nothing():
    validate_dashboard_args(parse([]))  # no dump_details attribute needed


def test_collector_config_from_args():
    config = collector_config_from_args(full_args(), start_ts=123.0)
    assert config.dashboard_dir == "/tmp/dump/dashboard"
    assert config.run_name == "my-run"
    assert config.start_ts == 123.0
    assert config.metric_whitelist == DEFAULT_METRIC_WHITELIST
    assert config.args_snapshot["colocate"] is True
    assert "hf_checkpoint" not in config.args_snapshot  # absent attrs are skipped


def test_collector_config_whitelist_override_and_run_name_fallback():
    args = full_args(wandb_group=None)
    args.dashboard_sglang_metrics = "sglang_gen_throughput,sglang_token_usage"
    config = collector_config_from_args(args, start_ts=0.0)
    assert config.metric_whitelist == ("sglang_gen_throughput", "sglang_token_usage")
    assert config.run_name == "miles-run"
