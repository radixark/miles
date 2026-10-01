import argparse
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from miles.utils.arguments import parse_args
from miles.utils.test_utils.snapshot import snapshot_values


@dataclass(frozen=True)
class ResultScenario:
    backend: str
    arguments: tuple[str, ...] = ()
    legacy: bool = False
    error: type[Exception] | None = None
    message: str = ""
    custom: bool = False


def capture_result(*, scenario: ResultScenario, directory: Path) -> dict[str, Any]:
    arguments = [
        "--train-backend",
        scenario.backend,
        "--rollout-batch-size",
        "2",
        "--num-rollout",
        "1",
        "--actor-num-gpus-per-node",
        "1",
        "--micro-batch-size",
        "1",
        "--run-uuid",
        "0123456789abcdef",
    ]
    if scenario.backend == "megatron":
        arguments.extend(["--num-layers", "1", "--hidden-size", "128", "--num-attention-heads", "2"])
    arguments.extend(token.replace("$FIXTURES", str(directory)) for token in scenario.arguments)

    with _environment(arguments=arguments, legacy=scenario.legacy):
        try:
            args = parse_args(add_custom_arguments=_custom_arguments if scenario.custom else None)
        except (AssertionError, ValueError, NotImplementedError, FileNotFoundError) as error:
            if scenario.error is None or type(error) is not scenario.error or scenario.message not in str(error):
                raise
            result = {"error": {"type": type(error).__name__, "message": str(error)}}
        else:
            if scenario.error is not None:
                raise AssertionError(f"Expected {scenario.error.__name__}: {scenario.message}")
            result = {"config": snapshot_values(args)}

    return _normalize(result, directory=directory)


def _custom_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--snapshot-custom", type=int, default=17)
    return parser


def _normalize(value: Any, *, directory: Path) -> Any:
    if isinstance(value, dict):
        return {key: _normalize(item, directory=directory) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize(item, directory=directory) for item in value]
    if isinstance(value, str):
        return value.replace(str(directory), "$FIXTURES")
    return value


@contextmanager
def _environment(*, arguments: list[str], legacy: bool) -> Iterator[None]:
    original_argv = sys.argv
    values = {
        "MILES_USE_LEGACY_ROLLOUT_V1": str(int(legacy)),
        "PROMETHEUS_PORT": "9090",
        "MILES_SCRIPT_ENV_REPORT": "",
        "MILES_BACKEND": None,
        "MILES_CI_GATE_RECORD_DIR": None,
        "DEPRECATED_MEGATRON_COMPATIBLE": "0",
        "RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
    }
    previous = {name: os.environ.get(name) for name in values}
    sys.argv = ["argument-result-snapshot", *arguments]
    try:
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        sys.argv = original_argv
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
