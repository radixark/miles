from __future__ import annotations

import json
import logging
import shlex
from typing import Annotated
from uuid import uuid4

import ray
import typer

from miles.utils.ft.controller.detectors.chain import DetectorChainConfig
from miles.utils.ft.controller.detectors.hang import HangDetectorConfig
from miles.utils.ft.controller.detectors.mfu_decline import MfuDeclineDetectorConfig
from miles.utils.ft.controller.detectors.network import NetworkAlertDetectorConfig
from miles.utils.ft.platform.controller_actor import FtControllerActor
from miles.utils.ft.platform.controller_factory import FtControllerConfig
from miles.utils.ft.protocols.platform import ft_controller_actor_name
from miles.utils.logging_utils import configure_logger

logger = logging.getLogger(__name__)

app = typer.Typer()

_DEFAULT_HANG = HangDetectorConfig()
_DEFAULT_NETWORK = NetworkAlertDetectorConfig()
_DEFAULT_MFU = MfuDeclineDetectorConfig()


@app.command(
    context_settings={"allow_extra_args": True, "allow_interspersed_args": False},
)
def main(
    ctx: typer.Context,
    ft_id: Annotated[
        str, typer.Option(help="FT instance ID for multi-instance isolation (auto-generated if empty)")
    ] = "",
    k8s_label_prefix: Annotated[
        str, typer.Option(help="K8s label prefix for label isolation (empty = no prefix)", envvar="MILES_FT_K8S_LABEL_PREFIX")
    ] = "",
    tick_interval: Annotated[
        float, typer.Option(help="Controller main loop interval (seconds)")
    ] = 30.0,
    platform: Annotated[
        str, typer.Option(help="Platform mode: 'stub' or 'k8s-ray'")
    ] = "k8s-ray",
    ray_address: Annotated[
        str, typer.Option(help="Ray dashboard address (k8s-ray mode)")
    ] = "http://127.0.0.1:8265",
    metric_store_backend: Annotated[
        str, typer.Option(help="Metric store backend: 'mini' or 'prometheus'")
    ] = "mini",
    prometheus_url: Annotated[
        str, typer.Option(help="Prometheus server URL (prometheus mode)")
    ] = "http://prometheus:9090",
    controller_exporter_port: Annotated[
        int, typer.Option(help="Controller Prometheus exporter HTTP port (0 = auto)")
    ] = 0,
    runtime_env_json: Annotated[
        str, typer.Option(help="Runtime env JSON for the training Ray job")
    ] = "{}",
    hang_training_timeout_minutes: Annotated[
        int, typer.Option(help="Hang detector: training timeout (minutes)")
    ] = _DEFAULT_HANG.training_timeout_minutes,
    hang_checkpoint_saving_timeout_minutes: Annotated[
        int, typer.Option(help="Hang detector: checkpoint saving timeout (minutes)")
    ] = _DEFAULT_HANG.checkpoint_saving_timeout_minutes,
    network_alert_window_minutes: Annotated[
        float, typer.Option(help="Network detector: alert window (minutes)")
    ] = _DEFAULT_NETWORK.alert_window_minutes,
    network_alert_threshold: Annotated[
        int, typer.Option(help="Network detector: alert count threshold")
    ] = _DEFAULT_NETWORK.alert_threshold,
    mfu_baseline: Annotated[
        float, typer.Option(help="MFU detector: explicit MFU baseline (0 = auto-detect from history)")
    ] = 0.0,
    mfu_threshold_ratio: Annotated[
        float, typer.Option(help="MFU detector: decline threshold as ratio of baseline")
    ] = _DEFAULT_MFU.mfu_threshold_ratio,
    mfu_consecutive_steps: Annotated[
        int, typer.Option(help="MFU detector: consecutive steps to average")
    ] = _DEFAULT_MFU.consecutive_steps,
    mfu_temperature_delta_threshold: Annotated[
        float, typer.Option(help="MFU detector: temperature delta threshold (celsius)")
    ] = _DEFAULT_MFU.temperature_delta_threshold,
    mfu_decline_timeout_minutes: Annotated[
        float, typer.Option(help="MFU detector: decline timeout before NOTIFY_HUMAN (minutes)")
    ] = _DEFAULT_MFU.decline_timeout_minutes,
    mfu_baseline_steps: Annotated[
        int, typer.Option(help="MFU detector: steps used for baseline computation")
    ] = _DEFAULT_MFU.baseline_steps,
    mfu_absolute_minimum: Annotated[
        float, typer.Option(help="MFU detector: absolute MFU floor (0 = disabled)")
    ] = _DEFAULT_MFU.mfu_absolute_minimum,
) -> None:
    """FT Controller entry point.

    Creates an FtController as a named Ray Actor, submits the training
    command (passed after --) as a Ray job, and blocks until the
    controller loop finishes.

    Usage: python -m miles.utils.ft.launcher [OPTIONS] -- COMMAND...
    """
    configure_logger()

    entrypoint = shlex.join(ctx.args)
    runtime_env = json.loads(runtime_env_json) if runtime_env_json else {}

    ft_id = ft_id or uuid4().hex[:8]

    detector_config = DetectorChainConfig(
        hang=HangDetectorConfig(
            training_timeout_minutes=hang_training_timeout_minutes,
            checkpoint_saving_timeout_minutes=hang_checkpoint_saving_timeout_minutes,
        ),
        network=NetworkAlertDetectorConfig(
            alert_window_minutes=network_alert_window_minutes,
            alert_threshold=network_alert_threshold,
        ),
        mfu=MfuDeclineDetectorConfig(
            mfu_baseline=mfu_baseline if mfu_baseline > 0 else None,
            mfu_threshold_ratio=mfu_threshold_ratio,
            consecutive_steps=mfu_consecutive_steps,
            temperature_delta_threshold=mfu_temperature_delta_threshold,
            decline_timeout_minutes=mfu_decline_timeout_minutes,
            baseline_steps=mfu_baseline_steps,
            mfu_absolute_minimum=mfu_absolute_minimum,
        ),
    )

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
        detector_config=detector_config,
    )

    actor_name = ft_controller_actor_name(ft_id)
    logger.info(
        "launcher_started actor_name=%s config=%s entrypoint=%s",
        actor_name,
        str(config),
        entrypoint,
    )

    actor = FtControllerActor.options(name=actor_name).remote(config=config)
    ray.get(actor.submit_and_run.remote())


if __name__ == "__main__":
    app()
