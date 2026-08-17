import asyncio
import logging

# TODO: this entrypoint reaches into miles.ray; fix the layering later
from miles.ray.specs.entrypoint import compute_specs
from miles.ray.wiring import launch_worker_manager
from miles.utils.arguments import parse_args
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
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
    logger.info(
        "This deployment carries no orchestration script, so it has no training to finish and stays up until it is "
        "torn down"
    )

    await asyncio.Event().wait()


if __name__ == "__main__":
    main()
