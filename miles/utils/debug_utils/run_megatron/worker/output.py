"""Compute and save output info (logprobs, etc.) from forward-pass results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import mpu

from miles.utils.debug_utils.run_megatron.worker.logprob import compute_logprob_info


def compute_and_save_output_info(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    position_ids: torch.Tensor,
    output_dir: Path,
) -> None:
    """Compute output info and save one JSON file per rank."""
    rank = _get_rank()
    payload = _compute_output_info(
        logits=logits,
        labels=labels,
        position_ids=position_ids,
        rank=rank,
    )

    if payload is None:
        print(
            f"[output] rank={rank}: skipping — logits shape {logits.shape} "
            f"does not look like vocab logits (critic model?)",
            flush=True,
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"rank_{rank}.json"
    output_path.write_text(json.dumps(payload, indent=2))

    print(f"[output] rank={rank}: saved to {output_path}", flush=True)


def _compute_output_info(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    position_ids: torch.Tensor,
    rank: int,
) -> dict[str, Any] | None:
    """Assemble the full output info payload for one rank.

    Returns None if there is nothing to compute (e.g. critic model logits).
    """
    logprob_info = compute_logprob_info(
        logits=logits,
        labels=labels,
        position_ids=position_ids,
    )
    if logprob_info is None:
        return None

    return {
        "rank": rank,
        "tp_size": mpu.get_tensor_model_parallel_world_size() if dist.is_initialized() else 1,
        "cp_size": mpu.get_context_parallel_world_size() if dist.is_initialized() else 1,
        "pp_size": mpu.get_pipeline_model_parallel_world_size() if dist.is_initialized() else 1,
        "logprob_entries": logprob_info,
    }


def _get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0
