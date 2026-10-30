"""Turning the miles dashboard on and configuring it, from environment variables and the run's arguments."""

from __future__ import annotations

import logging
import os
from typing import Any

from miles.dashboard.collector import CollectorConfig
from miles.dashboard.sglang_scraper import DEFAULT_METRIC_WHITELIST

logger = logging.getLogger(__name__)

ENABLE_ENV = "MILES_DASHBOARD"

# curated subset of args persisted into meta.json for the dashboard header
_SNAPSHOT_KEYS = (
    "wandb_group",
    "wandb_team",
    "wandb_project",
    "wandb_run_id",
    "wandb_host",
    "colocate",
    "num_gpus_per_node",
    "actor_num_nodes",
    "actor_num_gpus_per_node",
    "rollout_num_gpus",
    "rollout_num_gpus_per_engine",
    "rollout_batch_size",
    "n_samples_per_prompt",
    "rollout_max_response_len",
    "hf_checkpoint",
    "sglang_max_running_requests",
    "sglang_mem_fraction_static",
    "sglang_load_balance_method",
    "router",
    "router_dp_aware",
    "router_policy",
    "sglang_router_policy",
    "router_assignment_mode",
)


def validate_dashboard_args(args) -> None:
    if args["dump_details"] is None:
        raise ValueError(
            f"{ENABLE_ENV}=1 writes telemetry under {{dump-details}}/dashboard/ and the "
            "trajectory views read the rollout/train dumps, so --dump-details is required"
        )
    if args["cluster_backend"] == "kubernetes":
        raise ValueError(f"{ENABLE_ENV}=1 creates a Ray actor, which --cluster-backend kubernetes has no cluster for")
    if not args["use_rollout_entropy"]:
        logger.warning(
            f"{ENABLE_ENV}=1 without --use-rollout-entropy: per-token entropy "
            "will be missing from the dashboard token view"
        )


def collector_config_from_args(args: dict[str, Any], *, start_ts: float) -> CollectorConfig:
    """``args`` is every argument of the run, as the observers' ``start_run`` reports them."""
    if (metrics := os.environ.get("MILES_DASHBOARD_SGLANG_METRICS")) is not None:
        whitelist = tuple(m for m in metrics.split(",") if m)
        assert whitelist, f"empty MILES_DASHBOARD_SGLANG_METRICS: {metrics!r}"
    else:
        whitelist = DEFAULT_METRIC_WHITELIST
    return CollectorConfig(
        dashboard_dir=f"{args['dump_details']}/dashboard",
        run_name=args["wandb_group"] or "miles-run",
        start_ts=start_ts,
        args_snapshot={key: args[key] for key in _SNAPSHOT_KEYS if key in args},
        flush_interval_seconds=float(os.environ.get("MILES_DASHBOARD_FLUSH_INTERVAL", 5.0)),
        gpu_sample_interval_seconds=float(os.environ.get("MILES_DASHBOARD_GPU_SAMPLE_INTERVAL", 1.0)),
        scrape_interval_seconds=float(os.environ.get("MILES_DASHBOARD_SGLANG_SCRAPE_INTERVAL", 2.0)),
        metric_whitelist=whitelist,
        forward_prometheus=os.environ.get("MILES_DASHBOARD_FORWARD_PROMETHEUS") == "1",
    )
