import asyncio
from typing import Annotated

import typer

from miles.utils.ft.controller.controller import FtController
from miles.utils.ft.controller.mini_prometheus import MiniPrometheus, MiniPrometheusConfig
from miles.utils.ft.controller.mini_wandb import MiniWandb
from miles.utils.ft.platform.stubs import StubNodeManager, StubTrainingJob

app = typer.Typer()


def _build_k8s_ray_components(
    ray_address: str,
    entrypoint: str,
) -> tuple:
    """Lazily import and construct K8s + Ray platform components."""
    from miles.utils.ft.platform.k8s_node_manager import K8sNodeManager
    from miles.utils.ft.platform.ray_training_job import RayTrainingJob
    from ray.job_submission import JobSubmissionClient

    node_manager = K8sNodeManager()
    training_job = RayTrainingJob(
        client=JobSubmissionClient(address=ray_address),
        entrypoint=entrypoint,
    )
    return node_manager, training_job


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
) -> None:
    """FT Controller entry point."""
    if platform == "stub":
        node_manager = StubNodeManager()
        training_job = StubTrainingJob()
    elif platform == "k8s-ray":
        node_manager, training_job = _build_k8s_ray_components(
            ray_address=ray_address,
            entrypoint=entrypoint,
        )
    else:
        raise typer.BadParameter(f"Unknown platform: {platform}")

    metric_store = MiniPrometheus(config=MiniPrometheusConfig())
    mini_wandb = MiniWandb()

    controller = FtController(
        node_manager=node_manager,
        training_job=training_job,
        metric_store=metric_store,
        mini_wandb=mini_wandb,
        tick_interval=tick_interval,
    )

    asyncio.run(controller.run())


if __name__ == "__main__":
    app()
