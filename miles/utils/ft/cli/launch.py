from __future__ import annotations

import json
import logging
import shlex
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import ray
import typer

from miles.utils.ft.adapters.config import FtControllerConfig
from miles.utils.ft.adapters.types import ft_controller_actor_name
from miles.utils.ft.controller.detectors.chain import DetectorChainConfig
from miles.utils.ft.factories.controller import launch_ft_controller_actor
from miles.utils.logging_utils import configure_logger

logger = logging.getLogger(__name__)


def launch(
    ctx: typer.Context,
    ft_id: Annotated[
        str, typer.Option(help="FT instance ID for multi-instance isolation (auto-generated if empty)")
    ] = "",
    k8s_label_prefix: Annotated[
        str,
        typer.Option(
            help="K8s label prefix for label isolation (empty = no prefix)", envvar="MILES_FT_K8S_LABEL_PREFIX"
        ),
    ] = "",
    tick_interval: Annotated[float, typer.Option(help="Controller main loop interval (seconds)")] = 30.0,
    platform: Annotated[str, typer.Option(help="Platform mode: 'stub' or 'k8s-ray'")] = "k8s-ray",
    ray_address: Annotated[str, typer.Option(help="Ray dashboard address (k8s-ray mode)")] = "http://127.0.0.1:8265",
    metric_store_backend: Annotated[str, typer.Option(help="Metric store backend: 'mini' or 'prometheus'")] = "mini",
    prometheus_url: Annotated[
        str, typer.Option(help="Prometheus server URL (prometheus mode)")
    ] = "http://prometheus:9090",
    controller_exporter_port: Annotated[
        int, typer.Option(help="Controller Prometheus exporter HTTP port (0 = auto)")
    ] = 0,
    runtime_env_json: Annotated[str, typer.Option(help="Runtime env JSON for the training Ray job")] = "{}",
    notify_webhook_url: Annotated[
        str, typer.Option(help="Webhook URL for notifications (empty = no webhook notifications)")
    ] = "",
    notify_platform: Annotated[str, typer.Option(help="Notification platform: 'lark', 'slack', or 'discord'")] = "",
    rollout_num_cells: Annotated[int, typer.Option(help="Number of rollout cells")] = 0,
    scrape_interval_seconds: Annotated[
        float, typer.Option(help="Metric scrape interval in seconds (mini backend)")
    ] = 10.0,
    mini_prometheus_retention_minutes: Annotated[
        float, typer.Option(help="MiniPrometheus data retention in minutes (mini backend)")
    ] = 60.0,
    recovery_cooldown_window_minutes: Annotated[
        float, typer.Option(help="Recovery cooldown sliding window in minutes")
    ] = 30.0,
    recovery_cooldown_max_count: Annotated[
        int, typer.Option(help="Max recoveries allowed within cooldown window")
    ] = 3,
    registration_grace_ticks: Annotated[
        int, typer.Option(help="Grace ticks before requiring node registration")
    ] = 5,
    max_simultaneous_bad_nodes: Annotated[
        int, typer.Option(help="Max simultaneous bad nodes before aborting")
    ] = 3,
    monitoring_success_iterations: Annotated[
        int, typer.Option(help="Consecutive healthy iterations to confirm recovery")
    ] = 10,
    monitoring_timeout_seconds: Annotated[
        int, typer.Option(help="Timeout for post-recovery monitoring phase")
    ] = 600,
    recovery_timeout_seconds: Annotated[
        int, typer.Option(help="Timeout for the entire recovery process")
    ] = 3600,
    rollout_alive_threshold_seconds: Annotated[
        float | None, typer.Option(help="Rollout alive threshold in seconds (None = default)")
    ] = None,
    rollout_monitoring_alive_duration_seconds: Annotated[
        float | None, typer.Option(help="Rollout monitoring alive duration in seconds (None = default)")
    ] = None,
    detector_config_json: Annotated[
        str, typer.Option(help="Detector config JSON string or @filepath (default: all detectors with defaults)")
    ] = "",
) -> None:
    """FT Controller entry point.

    Creates an FtController as a named Ray Actor, submits the training
    command (passed after --) as a Ray job, and blocks until the
    controller loop finishes.

    Usage: python -m miles.utils.ft launch [OPTIONS] -- COMMAND...
    """
    configure_logger()

    entrypoint = shlex.join(ctx.args)
    runtime_env = json.loads(runtime_env_json) if runtime_env_json else {}

    ft_id = ft_id or uuid4().hex[:8]

    detector_config = _parse_detector_config(detector_config_json) if detector_config_json else DetectorChainConfig()

    config = FtControllerConfig(
        ft_id=ft_id,
        k8s_label_prefix=k8s_label_prefix,
        platform=platform,
        ray_address=ray_address,
        entrypoint=entrypoint,
        runtime_env=runtime_env,
        metric_store_backend=metric_store_backend,
        prometheus_url=prometheus_url,
        controller_exporter_port=controller_exporter_port,
        tick_interval=tick_interval,
        rollout_num_cells=rollout_num_cells,
        notify_webhook_url=notify_webhook_url,
        notify_platform=notify_platform,
        scrape_interval_seconds=scrape_interval_seconds,
        mini_prometheus_retention_minutes=mini_prometheus_retention_minutes,
        recovery_cooldown_window_minutes=recovery_cooldown_window_minutes,
        recovery_cooldown_max_count=recovery_cooldown_max_count,
        registration_grace_ticks=registration_grace_ticks,
        max_simultaneous_bad_nodes=max_simultaneous_bad_nodes,
        monitoring_success_iterations=monitoring_success_iterations,
        monitoring_timeout_seconds=monitoring_timeout_seconds,
        recovery_timeout_seconds=recovery_timeout_seconds,
        rollout_alive_threshold_seconds=rollout_alive_threshold_seconds,
        rollout_monitoring_alive_duration_seconds=rollout_monitoring_alive_duration_seconds,
        detector_config=detector_config,
    )

    actor_name = ft_controller_actor_name(ft_id)
    logger.info(
        "launcher_started actor_name=%s config=%s entrypoint=%s",
        actor_name,
        str(config),
        entrypoint,
    )

    actor = launch_ft_controller_actor(config=config, actor_name=actor_name)
    ray.get(actor.submit_and_run.remote())


def _parse_detector_config(raw: str) -> DetectorChainConfig:
    """Parse detector config from a JSON string or @filepath reference."""
    if raw.startswith("@"):
        raw = Path(raw[1:]).read_text()
    return DetectorChainConfig.model_validate_json(raw)
