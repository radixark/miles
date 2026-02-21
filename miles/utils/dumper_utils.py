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
from sglang.srt.debug_utils.dumper import _DumperConfig, dumper

logger = logging.getLogger(__name__)


class DumperPhase(enum.Enum):
    INFERENCE = "inference"
    FWD_ONLY = "fwd_only"
    FWD_BWD = "fwd_bwd"


def is_phase_enabled(args: Namespace, phase: DumperPhase) -> bool:
    overrides = _get_phase_overrides(args, phase)
    return bool(overrides.get("enable", False))


def get_dumper_dir(args: Namespace, phase: DumperPhase) -> Path:
    return Path(args.dumper_dir) / phase.value


def get_dumper_env_for_sglang(args: Namespace, engine_rank: int) -> dict[str, str]:
    """Build DUMPER_* env vars dict for the SGLang server subprocess.

    Returns an empty dict if the inference phase is not enabled.
    Each engine gets its own subdirectory via DUMPER_EXP_NAME=engine_{rank}.
    Only explicitly-set user overrides are exported — unset fields let the
    SGLang server use its own defaults.
    """
    overrides = _get_phase_overrides(args, DumperPhase.INFERENCE)
    if not overrides.get("enable", False):
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
        if field.name in overrides:
            env_name = _DumperConfig._env_name(field.name)
            raw_value = overrides[field.name]
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
    overrides = _get_phase_overrides(args, phase)
    if not overrides.get("enable", False):
        return False

    overrides["dir"] = str(get_dumper_dir(args, phase))
    overrides.setdefault("enable_http_server", False)
    overrides.setdefault("exp_name", phase.value)

    full_config = _DumperConfig(**overrides)
    dumper.reset()
    dumper.configure(**dataclasses.asdict(full_config))
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


def _get_phase_overrides(args: Namespace, phase: DumperPhase) -> dict[str, Any]:
    """Return only the explicitly-set dumper field overrides for *phase*."""
    raw = getattr(args, f"dumper_{phase.value}", None)
    overrides = _DumperConfig._kv_pairs_to_dict(raw) if isinstance(raw, list) else {}

    if "enable" not in overrides and getattr(args, "dumper_enable", False):
        overrides["enable"] = True

    return overrides
