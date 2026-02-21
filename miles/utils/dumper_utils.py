from __future__ import annotations

import dataclasses
import logging
from argparse import Namespace
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import torch
from sglang.srt.debug_utils.dumper import _DumperConfig, dumper

logger = logging.getLogger(__name__)

DumperPhase = Literal["sglang_inference", "megatron_forward_only", "megatron_forward_backward"]

PHASE_SGLANG_INFERENCE: DumperPhase = "sglang_inference"
PHASE_MEGATRON_FORWARD_ONLY: DumperPhase = "megatron_forward_only"
PHASE_MEGATRON_FORWARD_BACKWARD: DumperPhase = "megatron_forward_backward"

_PHASE_ATTR_MAP: dict[DumperPhase, str] = {
    PHASE_SGLANG_INFERENCE: "dumper_sglang",
    PHASE_MEGATRON_FORWARD_ONLY: "dumper_fwd_only",
    PHASE_MEGATRON_FORWARD_BACKWARD: "dumper_fwd_bwd",
}


def _get_valid_dumper_keys() -> set[str]:
    return {f.name for f in dataclasses.fields(_DumperConfig)}


def parse_dumper_config(pairs: list[str] | None) -> dict[str, Any]:
    """Parse key=value pairs into a dict suitable for dumper.configure().

    Values are auto-converted: 'true'/'1' -> True, 'false'/'0' -> False,
    numeric strings -> int. Keys are validated against _DumperConfig fields.
    """
    if not pairs:
        return {}

    valid_keys = _get_valid_dumper_keys()
    config: dict[str, Any] = {}

    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"Invalid dumper config pair (missing '='): {pair!r}")
        if key not in valid_keys:
            raise ValueError(f"Unknown dumper config key {key!r}. Valid keys: {sorted(valid_keys)}")

        if value.lower() in ("true", "1"):
            config[key] = True
        elif value.lower() in ("false", "0"):
            config[key] = False
        else:
            try:
                config[key] = int(value)
            except ValueError:
                config[key] = value

    return config


def is_phase_enabled(args: Namespace, phase: DumperPhase) -> bool:
    config = _get_phase_config(args, phase)
    return bool(config.get("enable", False))


def get_dumper_dir(args: Namespace, phase: DumperPhase) -> Path:
    return Path(args.dumper_dir) / phase


def get_dumper_env_for_sglang(args: Namespace, engine_rank: int) -> dict[str, str]:
    """Build DUMPER_* env vars dict for the SGLang server subprocess.

    Returns an empty dict if the sglang_inference phase is not enabled.
    Each engine gets its own subdirectory via DUMPER_EXP_NAME=engine_{rank}.
    """
    config = _get_phase_config(args, PHASE_SGLANG_INFERENCE)
    if not config.get("enable", False):
        return {}

    dumper_dir = get_dumper_dir(args, PHASE_SGLANG_INFERENCE)
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

    configure_kwargs = {k: v for k, v in config.items() if k != "enable"}
    configure_kwargs["enable"] = True
    configure_kwargs["dir"] = str(get_dumper_dir(args, phase))
    configure_kwargs.setdefault("enable_http_server", False)
    configure_kwargs.setdefault("exp_name", phase)

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


def _get_phase_config(args: Namespace, phase: DumperPhase) -> dict[str, Any]:
    if phase not in _PHASE_ATTR_MAP:
        raise ValueError(f"Unknown dumper phase {phase!r}. Valid: {list(_PHASE_ATTR_MAP)}")

    raw = getattr(args, _PHASE_ATTR_MAP[phase], None)
    config = parse_dumper_config(raw) if isinstance(raw, list) else (raw or {})

    if "enable" not in config and getattr(args, "dumper_enable", False):
        config["enable"] = True

    return config
