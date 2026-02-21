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
from sglang.srt.debug_utils.dumper import DumperConfig, dumper

logger = logging.getLogger(__name__)


class DumperPhase(enum.Enum):
    INFERENCE = "inference"
    FWD_ONLY = "fwd_only"
    FWD_BWD = "fwd_bwd"


def get_sglang_env(args: Namespace) -> dict[str, str]:
    if not _is_phase_enabled(args, DumperPhase.INFERENCE):
        return {}

    return {"DUMPER_SERVER_PORT": "reuse"}


async def configure_sglang(args: Namespace) -> None:
    if not _is_phase_enabled(args, DumperPhase.INFERENCE):
        return

    from miles.rollout.inference_rollout.inference_rollout_train import get_worker_urls
    from miles.utils.http_utils import post

    worker_urls = await get_worker_urls(args)
    overrides = _get_phase_overrides(args, DumperPhase.INFERENCE)
    coros = []
    for i, url in enumerate(worker_urls):
        body = {
            "enable": True,
            "dir": str(_get_dir(args)),
            "exp_name": f"engine_{i}",
            "cleanup_previous": True,
            **overrides,
        }
        coros.append(post(f"{url}/dumper/configure", body))

    await asyncio.gather(*coros)
    logger.info("Configured dumper on %d SGLang engines", len(worker_urls))


class DumperMegatronUtil:
    def __init__(self, args: Namespace, phase: DumperPhase) -> None:
        self.enabled = self._configure(args, phase)

    def wrap_forward_step(self, forward_step_func: Callable) -> Callable:
        if not self.enabled:
            return forward_step_func

        return _wrap_forward_step_with_stepping(forward_step_func)

    def finalize(self, model: torch.nn.Module) -> None:
        if not self.enabled:
            return

        dumper.dump_model(model)
        dumper.step()
        dumper.configure(enable=False)

    @staticmethod
    def _configure(args: Namespace, phase: DumperPhase) -> bool:
        overrides = _get_phase_overrides(args, phase)
        if not overrides.get("enable", False):
            return False

        merged = {
            "dir": str(_get_dir(args)),
            "exp_name": phase.value,
            "cleanup_previous": True,
            **overrides,
        }

        full_config = DumperConfig(**merged)
        dumper.reset()
        dumper.configure(**dataclasses.asdict(full_config))
        return True


def _wrap_forward_step_with_stepping(forward_step_func: Callable) -> Callable:
    is_first_call = True

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        nonlocal is_first_call
        if not is_first_call:
            dumper.step()
        is_first_call = False
        return forward_step_func(*args, **kwargs)

    return _wrapped


def _get_phase_overrides(args: Namespace, phase: DumperPhase) -> dict[str, Any]:
    raw = getattr(args, f"dumper_{phase.value}")
    overrides = DumperConfig._kv_pairs_to_dict(raw) if isinstance(raw, list) else {}

    if "enable" not in overrides and args.dumper_enable:
        overrides["enable"] = True

    return overrides


def _is_phase_enabled(args: Namespace, phase: DumperPhase) -> bool:
    overrides = _get_phase_overrides(args, phase)
    return bool(overrides.get("enable", False))


def _get_dir(args: Namespace) -> Path:
    return Path(args.dumper_dir)

