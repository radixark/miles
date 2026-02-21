from __future__ import annotations

import asyncio
import dataclasses
import enum
import logging
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch
from sglang.srt.debug_utils.dumper import _DumperConfig, dumper

logger = logging.getLogger(__name__)


class DumperPhase(enum.Enum):
    INFERENCE = "inference"
    FWD_ONLY = "fwd_only"
    FWD_BWD = "fwd_bwd"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def is_phase_enabled(args: Namespace, phase: DumperPhase) -> bool:
    overrides = _get_phase_overrides(args, phase)
    return bool(overrides.get("enable", False))


def get_dumper_dir(args: Namespace, phase: DumperPhase) -> Path:
    return Path(args.dumper_dir) / phase.value


def _get_phase_overrides(args: Namespace, phase: DumperPhase) -> dict[str, Any]:
    raw = getattr(args, f"dumper_{phase.value}", None)
    overrides = _DumperConfig._kv_pairs_to_dict(raw) if isinstance(raw, list) else {}

    if "enable" not in overrides and getattr(args, "dumper_enable", False):
        overrides["enable"] = True

    return overrides


# ---------------------------------------------------------------------------
# SGLang inference — env vars (actor creation) + HTTP (runtime control)
# ---------------------------------------------------------------------------

def get_dumper_env_for_sglang(args: Namespace) -> dict[str, str]:
    """Return env vars to set when creating SGLang Ray actors.

    Only registers the HTTP endpoint (DUMPER_SERVER_PORT=reuse) so we can
    configure the dumper at runtime via HTTP.  The dumper starts disabled;
    actual enable/configure happens via ``configure_dumper_for_sglang``.
    """
    if not is_phase_enabled(args, DumperPhase.INFERENCE):
        return {}
    return {"DUMPER_SERVER_PORT": "reuse"}


async def configure_dumper_for_sglang(args: Namespace, worker_urls: list[str]) -> None:
    """Configure and enable the dumper on all SGLang engines via HTTP.

    Each engine gets its own subdirectory via exp_name=engine_{i}.
    """
    if not is_phase_enabled(args, DumperPhase.INFERENCE):
        return

    overrides = _get_phase_overrides(args, DumperPhase.INFERENCE)
    dumper_dir = str(get_dumper_dir(args, DumperPhase.INFERENCE))

    from miles.utils.http_utils import post

    coros = []
    for i, url in enumerate(worker_urls):
        body: dict[str, Any] = {
            "enable": True,
            "dir": dumper_dir,
            "exp_name": f"engine_{i}",
        }
        skip_keys = {"enable", "dir", "exp_name"}
        for key, value in overrides.items():
            if key not in skip_keys:
                body[key] = value

        coros.append(post(f"{url}/dumper/configure", body))

    await asyncio.gather(*coros)
    logger.info("Configured dumper on %d SGLang engines (dir=%s)", len(worker_urls), dumper_dir)


async def reset_dumper_for_sglang(worker_urls: list[str]) -> None:
    """Reset the dumper on all SGLang engines via HTTP."""
    from miles.utils.http_utils import post

    await asyncio.gather(*[post(f"{url}/dumper/reset", {}) for url in worker_urls])


# ---------------------------------------------------------------------------
# Megatron phases — in-process dumper singleton
# ---------------------------------------------------------------------------

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


def finalize_dumper_phase(model: torch.nn.Module) -> None:
    """Finalize a dumper phase: dump model weights/grads, advance step, then disable."""
    dumper.dump_model(model)
    dumper.step()
    dumper.configure(enable=False)
