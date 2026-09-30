import logging
import os
import signal
from collections.abc import Awaitable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from tests.utils.dirs import get_test_data_dir, get_test_model_dir
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.events import LaunchOutcome

from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.workers.types import ClusterBackend

logger = logging.getLogger(__name__)

REPLACED_LAUNCH_EXIT_CODE: int = 128 + signal.SIGTERM

API_SERVER_PORT: int = 18080
API_SERVER_ARGS: str = f"--api-server-port {API_SERVER_PORT} --api-server-host 0.0.0.0 "
MODEL_DIR: str = get_test_model_dir()
DATA_DIR: str = get_test_data_dir()


@contextmanager
def recording_error(errors: dict[str, str], key: str) -> Iterator[None]:
    raise NotImplementedError


def compute_base_url(config: command_utils.ExecuteTrainConfig) -> str:
    return f"http://{config.create_backend().api_server_host(config)}:{API_SERVER_PORT}"


def compute_release_of_config(config: command_utils.ExecuteTrainConfig) -> str:
    return ReleaseName(
        run_id=config.run_id,
        deploy_component=config.deploy_component,
        deploy_instance_id=config.deploy_instance_id,
    ).serialize()


def create_soak_config(config: command_utils.ExecuteTrainConfig) -> command_utils.ExecuteTrainConfig:
    if config.cluster_backend is not ClusterBackend.RAY:
        return config
    return replace(config, ray_submission_id=f"miles-soak-{uuid4().hex}")


_DUMPS_ROOT_ENV = "MILES_TEST_DUMPS_ROOT"
_DEFAULT_DUMPS_ROOT = Path("/node_public/dumps")


def get_dumps_root() -> Path:
    root = Path(os.environ.get(_DUMPS_ROOT_ENV) or _DEFAULT_DUMPS_ROOT)
    if not root.is_absolute():
        raise ValueError("The shared dumps root must be an absolute path")
    return root


def resolve_dump_dir(test_name: str, *, run_id: str) -> str:
    dump_dir = get_dumps_root() / run_id / test_name
    os.makedirs(dump_dir, exist_ok=True)
    return str(dump_dir)


def assert_fresh_dump_dir(dump_dir: Path) -> None:
    if dump_dir.exists() and any(dump_dir.iterdir()):
        raise ValueError(f"Soak dump directory contains existing artifacts: {dump_dir}; choose a new run_id")
    dump_dir.mkdir(parents=True, exist_ok=True)


def evidence_directory(dump_dir: Path) -> Path:
    return dump_dir.with_name(f"{dump_dir.name}-soak") / uuid4().hex


async def note_launch_outcome(
    *, event_log: EventLog, request_id: str | None, launching: Awaitable[None]
) -> LaunchOutcome:
    raise NotImplementedError
