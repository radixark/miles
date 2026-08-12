import base64
import datetime
import json
import logging
import os
import platform
import random
import shlex
import socket
import subprocess
from pathlib import Path

from miles.utils.file_arg_utils import PSEUDO_FILE_PREFIX

logger = logging.getLogger(__name__)

repo_base_dir = Path(os.path.abspath(__file__)).resolve().parents[4]


def _pythonpath_with_sources(megatron_path: str, *additional_pythonpaths: str | None) -> str:
    entries = [str(repo_base_dir), megatron_path]
    for pythonpath in (*additional_pythonpaths, os.environ.get("PYTHONPATH")):
        if pythonpath:
            entries.extend(pythonpath.split(os.pathsep))
    return os.pathsep.join(dict.fromkeys(entries))


def _parse_extra_env_vars(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return {kv[0]: kv[1] for item in text.split(" ") if item.strip() != "" if (kv := item.split("=")) or True}


def get_default_wandb_args(test_file: str, run_name_prefix: str | None = None, run_id: str | None = None):
    if not os.environ.get("WANDB_API_KEY"):
        logger.info("Skip wandb configuration since WANDB_API_KEY is not found")
        return ""

    test_file = Path(test_file)
    test_name = test_file.stem
    if len(test_name) < 6:
        test_name = f"{test_file.parent.name}_{test_name}"

    wandb_run_name = run_id or create_run_id()
    if (x := os.environ.get("GITHUB_COMMIT_NAME")) is not None:
        wandb_run_name += f"_{x}"
    if (x := run_name_prefix) is not None:
        wandb_run_name = f"{x}_{wandb_run_name}"

    # Use the actual key value from environment to avoid shell expansion issues
    wandb_key = os.environ.get("WANDB_API_KEY")
    return (
        "--use-wandb "
        f"--wandb-project miles-{test_name} "
        f"--wandb-group {wandb_run_name} "
        f"--wandb-key '{wandb_key}' "
        "--disable-wandb-random-suffix "
    )


def create_run_id() -> str:
    return datetime.datetime.utcnow().strftime("%y%m%d-%H%M%S") + f"-{random.Random().randint(0, 999):03d}"


_warned_bool_env_var_keys = set()


# copied from SGLang
def get_bool_env_var(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default)
    value = value.lower()

    truthy_values = ("true", "1")
    falsy_values = ("false", "0")

    if (value not in truthy_values) and (value not in falsy_values):
        if value not in _warned_bool_env_var_keys:
            logger.warning(f"get_bool_env_var({name}) see non-understandable value={value} and treat as false")
        _warned_bool_env_var_keys.add(value)

    return value in truthy_values


def get_env_enable_infinite_run():
    return get_bool_env_var("MILES_TEST_ENABLE_INFINITE_RUN", "false")


class ArgvManipulator:
    @staticmethod
    def values_of(argv: list[str], flag: str) -> list[str]:
        values: list[str] = []
        for index, token in enumerate(argv):
            if token == flag:
                assert index + 1 < len(argv), f"{flag} is the last argument, so it names no value"
                values.append(argv[index + 1])
            elif token.startswith(f"{flag}="):
                values.append(token.split("=", maxsplit=1)[1])
        return values

    @staticmethod
    def declares(argv: list[str], flag: str) -> bool:
        return any(token == flag or token.startswith(f"{flag}=") for token in argv)

    @staticmethod
    def with_flag(argv: list[str], flag: str, value: str) -> list[str]:
        if ArgvManipulator.declares(argv, flag):
            return list(argv)
        return [*argv, flag, value]

    @staticmethod
    def replacing_value(argv: list[str], flag: str, value: str) -> list[str]:
        assert flag in argv, f"{flag} is not among the arguments, so there is no value of it to replace"
        rewritten = list(argv)
        rewritten[rewritten.index(flag) + 1] = value
        return rewritten


MOONCAKE_MASTER_PORT = 50051
MOONCAKE_MASTER_METRICS_PORT = 0
MOONCAKE_MASTER_LOG_PATH = Path("/tmp/mooncake_master.log")


def get_mooncake_object_store_args(master_port: int = MOONCAKE_MASTER_PORT) -> str:
    init_kwargs = {
        "protocol": "tcp",
        "master_server_address": f"127.0.0.1:{master_port}",
        "global_segment_size": "2gb",
        "local_buffer_size": "2gb",
    }
    return "--object-store-backend mooncake " f"--mooncake-store-init-kwargs {shlex.quote(json.dumps(init_kwargs))} "


def _is_tcp_server_ready(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def encode_pseudo_file(text: str) -> str:
    return PSEUDO_FILE_PREFIX + base64.b64encode(text.encode()).decode()


NUM_GPUS_OF_HARDWARE = {
    "H100": 8,
    "H200": 8,
    "B200": 8,
    "B300": 8,
    "GB200": 4,
    "GB300": 4,
    "MI350X": 8,
    "MI355X": 8,
}

GENERATION_HARDWARE = {
    "H100": "Hopper",
    "H200": "Hopper",
    "B200": "Blackwell",
    "B300": "Blackwell",
    "GB200": "Blackwell",
    "GB300": "Blackwell",
}


def detect_hardware() -> str:
    """Which NUM_GPUS_OF_HARDWARE entry this node is. Call it where the answer is used: prepare steps run GPU-free."""
    import torch

    assert torch.cuda.is_available(), "no visible GPU to detect the hardware from, pass --hardware explicitly"
    name = torch.cuda.get_device_name()
    if torch.version.hip is not None:
        detected = next((hardware for hardware in ("MI350X", "MI355X") if hardware in name), None)
    else:
        grace = platform.machine() == "aarch64"
        match torch.cuda.get_device_capability():
            case (9, 0):
                detected = "H200" if torch.cuda.get_device_properties(0).total_memory > 100 * 1024**3 else "H100"
            case (10, 0):
                detected = "GB200" if grace else "B200"
            case (10, 3):
                detected = "GB300" if grace else "B300"
            case _:
                detected = None
    assert detected is not None, f"cannot tell which hardware {name!r} is, pass --hardware explicitly"
    return detected


def run_process(
    argv: list[str], *, capture_output: bool, check: bool, input: str | None = None
) -> subprocess.CompletedProcess[str]:
    logger.info(f"EXEC: {shlex.join(argv)}")
    return subprocess.run(argv, check=check, capture_output=capture_output, text=True, input=input)
