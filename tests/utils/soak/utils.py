import os
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from tests.e2e.common_dirs import get_test_data_dir, get_test_model_dir
from tests.utils.soak.entrypoint import API_SERVER_PORT

from miles.utils.external_utils import command_utils
from miles.utils.workers.types import ClusterBackend

MODEL_DIR: str = get_test_model_dir()
DATA_DIR: str = get_test_data_dir()


def create_soak_config(config: command_utils.ExecuteTrainConfig) -> command_utils.ExecuteTrainConfig:
    if config.cluster_backend is not ClusterBackend.RAY:
        return config
    return replace(config, ray_submission_id=f"miles-soak-{uuid4().hex}")


def get_api_server_args(config: command_utils.ExecuteTrainConfig | None = None) -> str:
    resolved = config if config is not None else command_utils.default_config()
    if resolved.cluster_backend is not ClusterBackend.KUBERNETES:
        return f"--api-server-port {API_SERVER_PORT} --api-server-host 0.0.0.0 "
    return f"--api-server-port {API_SERVER_PORT} --api-server-host 0.0.0.0 --fault-witness-enable "


DEFAULT_TRAIN_SCRIPT: str = "train.py"
FULLY_ASYNC_TRAIN_SCRIPT: str = "train_async.py"


def get_train_script(*, fully_async: bool) -> str:
    return FULLY_ASYNC_TRAIN_SCRIPT if fully_async else DEFAULT_TRAIN_SCRIPT


def get_fully_async_args(*, fully_async: bool) -> str:
    if not fully_async:
        return ""
    return "--fully-async --pause-generation-mode in_place "


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


def evidence_directory(dump_dir: Path, *, session_id: str | None = None) -> Path:
    return dump_dir.with_name(f"{dump_dir.name}-soak") / (session_id if session_id is not None else uuid4().hex)
