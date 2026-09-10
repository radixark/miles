import asyncio
import json
import logging
import os
import re
from pathlib import Path

from tests.utils.soak.action import run_command
from tests.utils.soak.state import EventLog, SoakTeardownEvent

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.test_utils.kubectl_reads import compute_release_selector
from miles.utils.workers.types import ClusterBackend

logger = logging.getLogger(__name__)


def teardown_run(*, config: ExecuteTrainConfig, event_log: EventLog, evidence_dir: Path) -> None:
    asyncio.run(_teardown_run(config=config, event_log=event_log, evidence_dir=evidence_dir))


async def _teardown_run(*, config: ExecuteTrainConfig, event_log: EventLog, evidence_dir: Path) -> None:
    if config.cluster_backend is ClusterBackend.RAY:
        assert config.ray_submission_id, "Soak cleanup requires an owned Ray submission ID"
        resource = f"ray-job:{config.ray_submission_id}"
    else:
        assert config.namespace, "Soak cleanup requires an explicit namespace"
        release = ReleaseName(
            run_id=config.run_id,
            deploy_component=config.deploy_component,
            deploy_instance_id=config.deploy_instance_id,
        ).serialize()
        resource = f"helm:{config.namespace}/{release}"
    try:
        async with asyncio.timeout(300):
            if config.cluster_backend is ClusterBackend.RAY:
                await run_command(
                    [
                        "ray",
                        "job",
                        "stop",
                        *([] if "RAY_ADDRESS" in os.environ else ["--address", "http://127.0.0.1:8265"]),
                        config.ray_submission_id,
                    ],
                    timeout_seconds=300,
                    output_path=evidence_dir / "ray-job-stop.log",
                )
            else:
                await _remove_release(release=release, namespace=config.namespace)
    except BaseException as error:
        event_log.note_teardown(SoakTeardownEvent(resource=resource, returned=False, error=repr(error)))
        logger.error("Soak resource cleanup failed: %s", resource, exc_info=True)
        raise
    event_log.note_teardown(SoakTeardownEvent(resource=resource, returned=True))


async def _remove_release(*, release: str, namespace: str) -> None:
    await run_command(
        ["helm", "uninstall", release, "--namespace", namespace, "--ignore-not-found", "--wait", "--timeout", "240s"],
        timeout_seconds=250,
    )
    while True:
        async with asyncio.TaskGroup() as reads:
            manifests = reads.create_task(
                run_command(
                    [
                        "helm",
                        "list",
                        "--namespace",
                        namespace,
                        "--all",
                        "--filter",
                        f"^{re.escape(release)}$",
                        "--output",
                        "json",
                    ],
                    timeout_seconds=30,
                )
            )
            pods = reads.create_task(
                run_command(
                    [
                        "kubectl",
                        "get",
                        "pods",
                        "--namespace",
                        namespace,
                        "--selector",
                        compute_release_selector(release=release),
                        "--output",
                        "json",
                    ],
                    timeout_seconds=30,
                )
            )
        if not json.loads(manifests.result().stdout) and not json.loads(pods.result().stdout)["items"]:
            return
        await asyncio.sleep(2)
