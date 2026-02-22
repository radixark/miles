from __future__ import annotations

import asyncio
import dataclasses
import enum
import logging
import shutil
from argparse import Namespace
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from sglang.srt.debug_utils.dumper import DumperConfig, _get_rank, dumper

logger = logging.getLogger(__name__)


class DumperPhase(enum.Enum):
    INFERENCE = "inference"
    FWD_ONLY = "fwd_only"
    FWD_BWD = "fwd_bwd"


# ------------------------------- SGLang -------------------------------------


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
    overrides = _get_phase_override_configs(args, DumperPhase.INFERENCE)
    coros = []
    for i, url in enumerate(worker_urls):
        body = {
            "enable": True,
            "dir": str(_get_dir(args)),
            "exp_name": f"engine_{i}",
            **overrides,
        }
        coros.append(post(f"{url}/dumper/configure", body))

    await asyncio.gather(*coros)
    logger.info("Configured dumper on %d SGLang engines", len(worker_urls))


# ------------------------------- Megatron -------------------------------------


class DumperMegatronUtil:
    def __init__(self, args: Namespace, model: Sequence[torch.nn.Module], phase: DumperPhase) -> None:
        self.enabled = self._configure(args, phase)
        if self.enabled:
            dumper.register_non_intrusive_dumper(self._extract_model(model))

    def wrap_forward_step(self, forward_step_func: Callable) -> Callable:
        if not self.enabled:
            return forward_step_func

        return _wrap_forward_step_with_stepping(forward_step_func)

    def finalize(self, model: Sequence[torch.nn.Module]) -> None:
        if not self.enabled:
            return

        dumper.dump_model(self._extract_model(model))
        dumper.step()
        dumper.configure(enable=False)

    @staticmethod
    def _extract_model(model: Sequence[torch.nn.Module]) -> torch.nn.Module:
        assert (
            len(model) == 1
        ), f"Dumper does not yet support virtual pipeline parallelism (got {len(model)} model chunks)"
        return model[0]

    @staticmethod
    def _configure(args: Namespace, phase: DumperPhase) -> bool:
        overrides = _get_phase_override_configs(args, phase)
        if not overrides.get("enable"):
            return False

        merged = {
            "dir": str(_get_dir(args)),
            "exp_name": phase.value,
            **overrides,
        }

        full_config = DumperConfig(**merged)
        dumper.reset()
        _cleanup_dump_dir(Path(merged["dir"]) / merged["exp_name"])
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


# ------------------------------- Common -------------------------------------


def _cleanup_dump_dir(dump_dir: Path) -> None:
    if _get_rank() == 0 and dump_dir.is_dir():
        shutil.rmtree(dump_dir)
    if dist.is_initialized():
        dist.barrier()


def _get_phase_override_configs(args: Namespace, phase: DumperPhase) -> dict[str, Any]:
    raw = getattr(args, f"dumper_{phase.value}")
    return {"enable": args.dumper_enable, **DumperConfig._kv_pairs_to_dict(raw)}


def _is_phase_enabled(args: Namespace, phase: DumperPhase) -> bool:
    return _get_phase_override_configs(args, phase).get("enable", False)


def _get_dir(args: Namespace) -> Path:
    return Path(args.dumper_dir)
