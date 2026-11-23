"""The trainer and every rollout engine of a TOP run must share one GPU architecture.

FA4, DeepGEMM and the JIT build kernels per architecture, so the same contract on two architectures runs two
programs. The trainer records its device when it pins SGLang; each engine reports its own per DP rank in
``/server_info``, and every (re)connect compares them.
"""

import asyncio
from collections.abc import Sequence
from typing import Any

import torch.distributed as dist
from sglang.srt.true_on_policy.device import DEVICE_STATE_KEY, device_report, sm_name

from miles.utils import async_utils
from miles.utils.distributed_utils import get_gloo_group

_trainer_device: dict[str, Any] | None = None


def record_trainer_device() -> None:
    global _trainer_device
    _trainer_device = device_report()


def refuse_mixed_gpu_architectures(*, rollout_engines: Sequence, engine_cell_ids: Sequence[str]) -> None:
    """Rank 0 reads each engine's server info; every rank raises its error, so none waits in a later collective."""
    distributed = dist.is_initialized()
    error = [None]
    if not distributed or dist.get_rank() == 0:
        try:
            server_infos = async_utils.run(_gather_server_infos(rollout_engines))
            mismatch = gpu_architecture_mismatch(
                trainer=_trainer_device, engine_cell_ids=engine_cell_ids, server_infos=server_infos
            )
            error[0] = None if mismatch is None else RuntimeError(mismatch)
        except Exception as exc:
            error[0] = exc
    if distributed:
        dist.broadcast_object_list(error, src=0, group=get_gloo_group())
    if error[0] is not None:
        raise error[0]


async def _gather_server_infos(rollout_engines: Sequence) -> list[dict]:
    return await asyncio.gather(*(engine.get_server_info() for engine in rollout_engines))


def gpu_architecture_mismatch(
    *, trainer: dict[str, Any] | None, engine_cell_ids: Sequence[str], server_infos: Sequence[dict]
) -> str | None:
    if trainer is None:
        return "[top] the trainer connects to rollout engines before it pinned SGLang, so it has no GPU to compare"
    mismatches = []
    for cell_id, server_info in zip(engine_cell_ids, server_infos, strict=True):
        for dp_rank, state in enumerate(server_info["internal_states"]):
            engine = state.get(DEVICE_STATE_KEY)
            where = cell_id if len(server_info["internal_states"]) == 1 else f"{cell_id} (DP rank {dp_rank})"
            if engine is None:
                mismatches.append(
                    f"[top] engine {where} reports no GPU architecture; it runs without the contract or an "
                    "SGLang that predates the TOP device report"
                )
            elif tuple(engine["capability"]) != tuple(trainer["capability"]):
                mismatches.append(
                    f"[top] engine {where} runs {sm_name(engine['capability'])} ({engine['name']}), the trainer "
                    f"{sm_name(trainer['capability'])} ({trainer['name']}); FA4, DeepGEMM and the JIT build "
                    "different kernels per architecture, so run them on one GPU type"
                )
    return "\n".join(mismatches) or None
