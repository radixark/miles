from __future__ import annotations

import dataclasses
import enum
import logging
from argparse import Namespace
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from sglang.srt.debug_utils.dumper import _DumperConfig, _FrozenConfig, dumper

logger = logging.getLogger(__name__)


class DumperPhase(enum.Enum):
    INFERENCE = "inference"
    FWD_ONLY = "fwd_only"
    FWD_BWD = "fwd_bwd"


def parse_dumper_config(pairs: list[str] | None) -> dict[str, Any]:
    """Parse key=value pairs into a dict suitable for dumper.configure().

    Values are coerced based on the target field's type in _DumperConfig
    (bool fields: 'true'/'1' -> True; int fields: parsed as int; str fields: kept as-is).
    Keys are validated against _DumperConfig fields.
    """
    if not pairs:
        return {}

    valid_fields = {f.name: f for f in dataclasses.fields(_DumperConfig)}
    config: dict[str, Any] = {}

    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"Invalid dumper config pair (missing '='): {pair!r}")
        if key not in valid_fields:
            raise ValueError(f"Unknown dumper config key {key!r}. Valid keys: {sorted(valid_fields)}")

        config[key] = _FrozenConfig._parse_env_value(value, valid_fields[key].default)

    return config


def is_phase_enabled(args: Namespace, phase: DumperPhase) -> bool:
    config = _get_phase_config(args, phase)
    return bool(config.get("enable", False))


def get_dumper_dir(args: Namespace, phase: DumperPhase) -> Path:
    return Path(args.dumper_dir) / phase.value


def get_dumper_env_for_sglang(args: Namespace, engine_rank: int) -> dict[str, str]:
    """Build DUMPER_* env vars dict for the SGLang server subprocess.

    Returns an empty dict if the inference phase is not enabled.
    Each engine gets its own subdirectory via DUMPER_EXP_NAME=engine_{rank}.
    """
    config = _get_phase_config(args, DumperPhase.INFERENCE)
    if not config.get("enable", False):
        return {}

    dumper_dir = get_dumper_dir(args, DumperPhase.INFERENCE)
    env: dict[str, str] = {
        "DUMPER_ENABLE": "1",
        "DUMPER_DIR": str(dumper_dir),
        "DUMPER_EXP_NAME": f"engine_{engine_rank}",
        "DUMPER_SERVER_PORT": "reuse",
    }

    skip_keys = {"enable", "dir", "server_port", "exp_name"}
    for field in dataclasses.fields(_DumperConfig):
        if field.name in skip_keys:
            continue
        if field.name in config:
            env_name = _DumperConfig._env_name(field.name)
            raw_value = config[field.name]
            if raw_value is True:
                env[env_name] = "1"
            elif raw_value is False:
                env[env_name] = "0"
            else:
                env[env_name] = str(raw_value)

    logger.info(f"Built DUMPER_* env vars for engine_rank={engine_rank}: dir={dumper_dir}")
    return env


def configure_dumper_for_phase(args: Namespace, phase: DumperPhase) -> bool:
    """Configure the dumper singleton for a Megatron phase. Returns True if enabled."""
    config = _get_phase_config(args, phase)
    if not config.get("enable", False):
        return False

    defaults = {f.name: f.default for f in dataclasses.fields(_DumperConfig)}
    configure_kwargs = {**defaults}
    configure_kwargs.update({k: v for k, v in config.items() if k != "enable"})
    configure_kwargs["enable"] = True
    configure_kwargs["dir"] = str(get_dumper_dir(args, phase))
    if "enable_http_server" not in config:
        configure_kwargs["enable_http_server"] = False
    if "exp_name" not in config:
        configure_kwargs["exp_name"] = phase.value

    dumper.reset()
    dumper.configure(**configure_kwargs)
    return True


@contextmanager
def dumper_phase_scope(
    args: Namespace,
    phase: DumperPhase,
    model: torch.nn.Module,
) -> Generator[bool, None, None]:
    """Context manager wrapping the full dumper lifecycle for a Megatron phase.

    On entry: configure + step() (if enabled).
    On exit: dump_model (if enabled).
    """
    enabled = configure_dumper_for_phase(args, phase)

    if enabled:
        dumper.step()

    try:
        yield enabled
    finally:
        if enabled:
            dumper.dump_model(model)
            dumper.configure(enable=False)


def _get_phase_config(args: Namespace, phase: DumperPhase) -> dict[str, Any]:
    raw = getattr(args, f"dumper_{phase.value}", None)
    config = parse_dumper_config(raw) if isinstance(raw, list) else (raw or {})

    if "enable" not in config and getattr(args, "dumper_enable", False):
        config["enable"] = True

    return config
