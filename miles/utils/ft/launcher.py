from __future__ import annotations

import asyncio
import logging
from typing import Annotated

import typer

from miles.utils.ft.models import FT_CONTROLLER_ACTOR_NAME
from miles.utils.ft.platform.controller_actor import FtControllerActor
from miles.utils.ft.platform.controller_factory import (
    FtControllerConfig,
    build_ft_controller,
)

logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def main(
    tick_interval: Annotated[
        float, typer.Option(help="Controller main loop interval (seconds)")
    ] = 30.0,
    platform: Annotated[
        str, typer.Option(help="Platform mode: 'stub' or 'k8s-ray'")
    ] = "stub",
    ray_address: Annotated[
        str, typer.Option(help="Ray dashboard address (k8s-ray mode)")
    ] = "http://127.0.0.1:8265",
    entrypoint: Annotated[
        str, typer.Option(help="Training job entrypoint command (k8s-ray mode)")
    ] = "",
    metric_store_backend: Annotated[
        str, typer.Option(help="Metric store backend: 'mini' or 'prometheus'")
    ] = "mini",
    prometheus_url: Annotated[
        str, typer.Option(help="Prometheus server URL (prometheus mode)")
    ] = "http://prometheus:9090",
    controller_exporter_port: Annotated[
        int, typer.Option(help="Controller Prometheus exporter HTTP port")
    ] = 9400,
    as_ray_actor: Annotated[
        bool, typer.Option(help="Create a detached Ray Actor instead of running inline")
    ] = False,
) -> None:
    """FT Controller entry point.

    When --as-ray-actor is set (production mode), creates a detached named
    Ray Actor and returns immediately. The actor runs the controller loop
    in the background. FtMegatronAgent finds it via ray.get_actor("ft_controller").

    When --as-ray-actor is not set (dev/test mode), builds and runs the
    controller inline with asyncio.run().
    """
    config = FtControllerConfig(
        platform=platform,
        ray_address=ray_address,
        entrypoint=entrypoint,
        metric_store_backend=metric_store_backend,
        prometheus_url=prometheus_url,
        controller_exporter_port=controller_exporter_port,
        tick_interval=tick_interval,
    )

    if as_ray_actor:
        actor = FtControllerActor.options(
            name=FT_CONTROLLER_ACTOR_NAME,
            lifetime="detached",
        ).remote(config=config)
        actor.run.remote()
        logger.info(
            "ft_controller actor created and started "
            "platform=%s backend=%s exporter_port=%d",
            config.platform, config.metric_store_backend, config.controller_exporter_port,
        )
        return

    controller = build_ft_controller(config=config)
    logger.info(
        "launcher_started_inline platform=%s backend=%s exporter_port=%d",
        config.platform, config.metric_store_backend, config.controller_exporter_port,
    )
    asyncio.run(controller.run())


if __name__ == "__main__":
    app()
