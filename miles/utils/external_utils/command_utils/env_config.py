from __future__ import annotations

import dataclasses
import hashlib
import itertools
import os
import re
import sys
import typing
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.typer_utils import SCRIPT_ENV_VAR_PREFIX
from miles.utils.workers.types import ClusterBackend

RUN_ID_PREFIX = "e2e"
RUN_ID_STEM_BUDGET = 12
RUN_ID_DIGEST_BYTES = 2

_TRUTHY = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSY = frozenset({"0", "false", "f", "no", "n", "off"})

_UNSAFE_RUN_ID_CHARS = re.compile(r"[^a-z0-9]+")

_launch_indices = itertools.count(1)


def default_config(config_class: type = ExecuteTrainConfig) -> ExecuteTrainConfig:
    config = config_from_env(config_class)
    if not config.run_id and config.cluster_backend == ClusterBackend.KUBERNETES.value:
        config.run_id = derive_run_id(entry_script=sys.argv[0], launch_index=next(_launch_indices))
    return config


def config_from_env(
    config_class: type = ExecuteTrainConfig, environ: Mapping[str, str] | None = None
) -> ExecuteTrainConfig:
    environ = os.environ if environ is None else environ
    hints = typing.get_type_hints(config_class)
    overrides = {
        field.name: parse_value(hints[field.name], value)
        for field in dataclasses.fields(config_class)
        if (value := environ.get(env_var_name(field.name))) is not None
    }
    return config_class(**overrides)


def env_var_name(field_name: str) -> str:
    return f"{SCRIPT_ENV_VAR_PREFIX}{field_name.upper()}"


def parse_value(annotation: Any, value: str) -> Any:
    if annotation is bool:
        return parse_bool(value)
    if annotation is int:
        return int(value)
    if annotation is str:
        return value
    if typing.get_origin(annotation) is tuple:
        return tuple(value.split())
    raise AssertionError(
        f"{ExecuteTrainConfig.__name__} grew a {annotation} field, and a launch script's environment carries "
        f"strings only; teach this function how to read one"
    )


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    assert lowered in _TRUTHY or lowered in _FALSY, (
        f"{value!r} is neither true nor false; write one of {sorted(_TRUTHY | _FALSY)} so the launcher and the "
        f"command line agree on what it means"
    )
    return lowered in _TRUTHY


def derive_run_id(*, entry_script: str, launch_index: int) -> str:
    path = Path(entry_script)
    stem = _UNSAFE_RUN_ID_CHARS.sub("-", path.stem.lower())[:RUN_ID_STEM_BUDGET].strip("-")
    digest = hashlib.blake2b(f"{path.parent.name}/{path.name}".encode(), digest_size=RUN_ID_DIGEST_BYTES).hexdigest()
    return "-".join(part for part in (RUN_ID_PREFIX, stem, digest, str(launch_index)) if part)
