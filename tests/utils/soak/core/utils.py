import logging
import signal
from collections.abc import Awaitable, Iterator
from contextlib import contextmanager
from pathlib import Path

from tests.utils.dirs import get_test_data_dir, get_test_model_dir
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.events import LaunchOutcome

from miles.utils.external_utils import command_utils

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
    raise NotImplementedError


def compute_release_of_config(config: command_utils.ExecuteTrainConfig) -> str:
    raise NotImplementedError


def create_soak_config(config: command_utils.ExecuteTrainConfig) -> command_utils.ExecuteTrainConfig:
    raise NotImplementedError


_DUMPS_ROOT_ENV = "MILES_TEST_DUMPS_ROOT"
_DEFAULT_DUMPS_ROOT = Path("/node_public/dumps")


def get_dumps_root() -> Path:
    raise NotImplementedError


def resolve_dump_dir(test_name: str, *, run_id: str) -> str:
    raise NotImplementedError


def assert_fresh_dump_dir(dump_dir: Path) -> None:
    raise NotImplementedError


def evidence_directory(dump_dir: Path) -> Path:
    raise NotImplementedError


async def note_launch_outcome(
    *, event_log: EventLog, request_id: str | None, launching: Awaitable[None]
) -> LaunchOutcome:
    raise NotImplementedError
