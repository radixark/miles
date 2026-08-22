import asyncio
import logging

# TODO: this entrypoint reaches into miles.ray; fix the layering later
from miles.ray.specs.entrypoint import compute_specs
from miles.ray.specs.train import compute_trainer_ids, create_trainer_controller_handle
from miles.ray.wiring import get_backend_capability, launch_worker_manager
from miles.utils.arguments import parse_args
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.ft_utils.api_server.server import start_api_server
from miles.utils.ft_utils.mini_ft_controller import maybe_start_mini_ft_controller
from miles.utils.logging_utils import configure_logger
from miles.utils.workers.types import DeployComponent

logger = logging.getLogger(__name__)


def main() -> None:
    asyncio.run(_serve_deployed_workers(parse_args()))


async def _serve_deployed_workers(args) -> None:
    configure_logger(args, source=SimpleProcessIdentity(component="main"))
    component = DeployComponent(args.deploy_component)
    assert not component.deploys_orchestration_script(), (
        f"this entrypoint installs the workers of a deployment that carries no orchestration script, and "
        f"--deploy-component {component.value} carries one: launch it through its driver script instead"
    )

    _worker_manager = launch_worker_manager(args)
    logger.info(f"Deployed the {component.value} workers of this run: {[spec.name for spec in compute_specs(args)]}")
    _maybe_serve_own_fault_tolerance(args, component=component)
    logger.info(
        "This deployment carries no orchestration script, so it has no training to finish and stays up until it is "
        "torn down"
    )

    await asyncio.Event().wait()


def _maybe_serve_own_fault_tolerance(args, *, component: DeployComponent) -> None:
    if not args.api_server_port or not component.selects(DeployComponent.TRAINER):
        return

    capability = get_backend_capability(args)
    start_api_server(
        args=args,
        trainer_models={
            trainer_id: create_trainer_controller_handle(args, capability=capability, trainer_id=trainer_id)
            for trainer_id in compute_trainer_ids(args)
        },
        inference_controller=None,
        port=args.api_server_port,
        ft_components=args.ft_components,
        cell_operations=capability.cell_operations(),
    )
    maybe_start_mini_ft_controller(args)
    logger.info(f"Serving the fault tolerance of this deployment's own cells on port {args.api_server_port}")


if __name__ == "__main__":
    main()
