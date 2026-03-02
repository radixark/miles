"""Compute and save output info (logprobs, etc.) from forward-pass results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import mpu


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
    logprob_entries = _compute_logprob_entries(
        logits=logits,
        labels=labels,
        position_ids=position_ids,
    )
    if logprob_entries is None:
        return None

    return {
        "rank": rank,
        "tp_size": mpu.get_tensor_model_parallel_world_size() if dist.is_initialized() else 1,
        "cp_size": mpu.get_context_parallel_world_size() if dist.is_initialized() else 1,
        "pp_size": mpu.get_pipeline_model_parallel_world_size() if dist.is_initialized() else 1,
        "logprob_entries": logprob_entries,
    }


def _compute_logprob_entries(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    position_ids: torch.Tensor,
) -> list[list[dict[str, Any]]] | None:
    """Compute per-token logprob entries.

    Returns ``entries[batch][seq]`` dicts, or None if logits don't look like
    vocab logits (e.g. critic model).

    Args:
        logits: [batch_size, local_seq_len, vocab_size] — must already be gathered across TP.
        labels: [batch_size, local_seq_len], -100 = ignore.
        position_ids: [batch_size, local_seq_len], global positions.
    """
    if logits.ndim < 3 or logits.size(-1) == 1:
        return None

    batch_size, local_seq_len, _ = logits.shape
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    all_entries: list[list[dict[str, Any]]] = []
    for b in range(batch_size):
        batch_entries: list[dict[str, Any]] = []
        for s in range(local_seq_len):
            label_id: int = labels[b, s].item()
            is_valid = label_id != -100
            lp: float = log_probs[b, s, label_id].item() if is_valid else 0.0

            batch_entries.append({
                "global_position": position_ids[b, s].item(),
                "token_id": label_id if is_valid else -1,
                "logprob": lp,
                "is_valid": is_valid,
            })
        all_entries.append(batch_entries)

    return all_entries


def _get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0
