"""CLI arguments and configuration plumbing for the miles dashboard."""

from __future__ import annotations

import logging

from miles.dashboard.collector import CollectorConfig
from miles.dashboard.sglang_scraper import DEFAULT_METRIC_WHITELIST

logger = logging.getLogger(__name__)

# curated subset of args persisted into meta.json for the dashboard header
_SNAPSHOT_KEYS = (
    "wandb_group",
    "colocate",
    "num_gpus_per_node",
    "actor_num_nodes",
    "actor_num_gpus_per_node",
    "rollout_num_gpus",
    "rollout_num_gpus_per_engine",
    "rollout_batch_size",
    "n_samples_per_prompt",
    "hf_checkpoint",
)


def add_dashboard_arguments(parser) -> None:
    group = parser.add_argument_group("miles dashboard")
    group.add_argument(
        "--use-miles-dashboard",
        action="store_true",
        default=False,
        help="Collect dashboard telemetry (phases, GPU util, engine metrics) under {dump-details}/dashboard/. "
        "Requires --dump-details. View with `python -m miles.dashboard.serve`.",
    )
    group.add_argument("--dashboard-flush-interval", type=float, default=5.0, help="collector disk flush cadence (s)")
    group.add_argument("--dashboard-gpu-sample-interval", type=float, default=1.0, help="NVML sampling cadence (s)")
    group.add_argument("--dashboard-sglang-scrape-interval", type=float, default=2.0, help="engine scrape cadence (s)")
    group.add_argument(
        "--dashboard-sglang-scrape-mode",
        type=str,
        choices=["auto", "router", "direct"],
        default="auto",
        help="auto scrapes {router}/engine_metrics, or each engine's /metrics under --use-miles-router",
    )
    group.add_argument(
        "--dashboard-sglang-metrics",
        type=str,
        default=None,
        help="comma-separated override of the scraped sglang metric whitelist",
    )
    group.add_argument(
        "--dashboard-forward-prometheus",
        action="store_true",
        default=False,
        help="also push dashboard gauges to the --use-prometheus collector for external Grafana",
    )


def validate_dashboard_args(args) -> None:
    if not args.use_miles_dashboard:
        return
    assert args.dump_details is not None, (
        "--use-miles-dashboard writes telemetry under {dump-details}/dashboard/ and the "
        "trajectory views read the rollout/train dumps, so --dump-details is required"
    )
    if not args.use_rollout_entropy:
        logger.warning(
            "--use-miles-dashboard without --use-rollout-entropy: per-token entropy "
            "will be missing from the dashboard token view"
        )


def collector_config_from_args(args, *, start_ts: float) -> CollectorConfig:
    if args.dashboard_sglang_metrics is not None:
        whitelist = tuple(m for m in args.dashboard_sglang_metrics.split(",") if m)
        assert whitelist, f"empty --dashboard-sglang-metrics: {args.dashboard_sglang_metrics!r}"
    else:
        whitelist = DEFAULT_METRIC_WHITELIST
    snapshot = {key: getattr(args, key) for key in _SNAPSHOT_KEYS if hasattr(args, key)}
    return CollectorConfig(
        dashboard_dir=f"{args.dump_details}/dashboard",
        run_name=args.wandb_group or "miles-run",
        start_ts=start_ts,
        args_snapshot=snapshot,
        flush_interval_seconds=args.dashboard_flush_interval,
        gpu_sample_interval_seconds=args.dashboard_gpu_sample_interval,
        scrape_interval_seconds=args.dashboard_sglang_scrape_interval,
        scrape_mode=args.dashboard_sglang_scrape_mode,
        metric_whitelist=whitelist,
        forward_prometheus=args.dashboard_forward_prometheus,
    )
