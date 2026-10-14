import asyncio
import logging
import os
from pathlib import Path

from tests.utils.deploy.hot_restart.release import remove_release_and_wait
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.events import SoakTeardownEvent
from tests.utils.soak.core.utils import compute_release_of_config

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.common import run_process
from miles.utils.workers.types import ClusterBackend

logger = logging.getLogger(__name__)

_TEARDOWN_TIMEOUT_SECONDS: float = 300.0


async def teardown_run(*, config: ExecuteTrainConfig, event_log: EventLog, evidence_dir: Path) -> None:
    stops_ray_job = config.cluster_backend is ClusterBackend.RAY
    if stops_ray_job:
        assert config.ray_submission_id, "Soak cleanup requires an owned Ray submission ID"
        resource = f"ray-job:{config.ray_submission_id}"
    else:
        assert config.namespace, "Soak cleanup requires an explicit namespace"
        release = compute_release_of_config(config)
        resource = f"helm:{config.namespace}/{release}"

    try:
        async with asyncio.timeout(_TEARDOWN_TIMEOUT_SECONDS):
            if stops_ray_job:
                await _stop_ray_job(submission_id=config.ray_submission_id, evidence_dir=evidence_dir)
            else:
                await _remove_release(release=release, namespace=config.namespace)
    except BaseException as error:
        event_log.append(SoakTeardownEvent(resource=resource, returned=False, error=repr(error)))
        logger.error("Soak resource cleanup failed: %s", resource, exc_info=True)
        raise

    event_log.append(SoakTeardownEvent(resource=resource, returned=True))


async def _stop_ray_job(*, submission_id: str, evidence_dir: Path) -> None:
    result = await asyncio.to_thread(
        run_process,
        [
            "ray",
            "job",
            "stop",
            *([] if "RAY_ADDRESS" in os.environ else ["--address", "http://127.0.0.1:8265"]),
            submission_id,
        ],
        capture_output=True,
        check=True,
        timeout=_TEARDOWN_TIMEOUT_SECONDS,
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "ray-job-stop.log").write_text(result.stdout)


async def _remove_release(*, release: str, namespace: str) -> None:
    await asyncio.to_thread(remove_release_and_wait, release=release, namespace=namespace)
