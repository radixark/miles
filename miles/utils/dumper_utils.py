from __future__ import annotations

import asyncio
import dataclasses
import enum
import logging
from argparse import Namespace
from collections.abc import Callable
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


def get_dumper_dir(args: Namespace) -> Path:
    return Path(args.dumper_dir)


def get_dumper_env_for_sglang(args: Namespace) -> dict[str, str]:
    if not is_phase_enabled(args, DumperPhase.INFERENCE):
        return {}
    return {"DUMPER_SERVER_PORT": "reuse"}


async def configure_dumper_for_sglang(args: Namespace) -> None:
    if not is_phase_enabled(args, DumperPhase.INFERENCE):
        return

    from miles.rollout.inference_rollout.inference_rollout_train import get_worker_urls
    from miles.utils.http_utils import post

    worker_urls = await get_worker_urls(args)
    overrides = _get_phase_overrides(args, DumperPhase.INFERENCE)
    dumper_dir = str(get_dumper_dir(args))

    coros = []
    for i, url in enumerate(worker_urls):
        body = {
            **overrides,
            "enable": True,
            "dir": dumper_dir,
            "exp_name": f"engine_{i}",
            "cleanup_previous": True,
        }
        coros.append(post(f"{url}/dumper/configure", body))

    await asyncio.gather(*coros)
    logger.info("Configured dumper on %d SGLang engines (dir=%s)", len(worker_urls), dumper_dir)


def configure_dumper_for_phase(args: Namespace, phase: DumperPhase) -> bool:
    overrides = _get_phase_overrides(args, phase)
    if not overrides.get("enable", False):
        return False

    overrides["dir"] = str(get_dumper_dir(args))
    overrides.setdefault("enable_http_server", False)
    overrides.setdefault("exp_name", phase.value)
    overrides.setdefault("cleanup_previous", True)

    full_config = _DumperConfig(**overrides)
    dumper.reset()
    dumper.configure(**dataclasses.asdict(full_config))
    return True


def finalize_dumper_phase(model: torch.nn.Module) -> None:
    dumper.dump_model(model)
    dumper.step()
    dumper.configure(enable=False)


def wrap_forward_step_with_dumper_stepping(forward_step_func: Callable) -> Callable:
    """Wrap a Megatron ``forward_step`` so that ``dumper.step()`` is called
    between microbatches.

    The wrapper calls ``dumper.step()`` before every invocation of
    *forward_step_func* **except** the first one.  This ensures that each
    microbatch's forward (and the subsequent loss computation callback) lands
    in its own dumper step, while avoiding an off-by-one empty step at the
    beginning.
    """
    is_first_call = True

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        nonlocal is_first_call
        if not is_first_call:
            dumper.step()
        is_first_call = False
        return forward_step_func(*args, **kwargs)

    return _wrapped


def _get_phase_overrides(args: Namespace, phase: DumperPhase) -> dict[str, Any]:
    raw = getattr(args, f"dumper_{phase.value}", None)
    overrides = _DumperConfig._kv_pairs_to_dict(raw) if isinstance(raw, list) else {}

    if "enable" not in overrides and getattr(args, "dumper_enable", False):
        overrides["enable"] = True

    return overrides
