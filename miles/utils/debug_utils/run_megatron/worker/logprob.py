"""Compute and save per-token log-probabilities from forward-pass logits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import mpu


@dataclass
class _LogprobEntry:
    global_position: int
    token_id: int
    logprob: float
    is_valid: bool


def compute_and_save_logprobs(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    position_ids: torch.Tensor,
    output_dir: Path,
) -> None:
    """Compute per-token logprobs and save one JSON file per rank.

    Args:
        logits: [batch_size, local_seq_len, vocab_size] — must already be gathered across TP.
        labels: [batch_size, local_seq_len], -100 = ignore.
        position_ids: [batch_size, local_seq_len], global positions.
        output_dir: directory to write ``rank_{rank}.json`` files into.
    """
    if logits.ndim < 3 or logits.size(-1) == 1:
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(
            f"[logprob] rank={rank}: skipping logprob — logits shape {logits.shape} "
            f"does not look like vocab logits (critic model?)",
            flush=True,
        )
        return

    rank = dist.get_rank() if dist.is_initialized() else 0
    output_dir.mkdir(parents=True, exist_ok=True)

    log_probs: torch.Tensor = torch.log_softmax(logits.float(), dim=-1)

    batch_size, local_seq_len, vocab_size = logits.shape
    all_entries: list[list[dict[str, Any]]] = []
    total_valid = 0
    sum_logprob = 0.0
    min_logprob = float("inf")
    max_logprob = float("-inf")

    for b in range(batch_size):
        batch_entries: list[dict[str, Any]] = []
        for s in range(local_seq_len):
            label_id: int = labels[b, s].item()
            is_valid = label_id != -100
            if is_valid:
                lp: float = log_probs[b, s, label_id].item()
                total_valid += 1
                sum_logprob += lp
                min_logprob = min(min_logprob, lp)
                max_logprob = max(max_logprob, lp)
            else:
                lp = 0.0

            batch_entries.append(
                _entry_to_dict(_LogprobEntry(
                    global_position=position_ids[b, s].item(),
                    token_id=label_id if is_valid else -1,
                    logprob=lp,
                    is_valid=is_valid,
                ))
            )
        all_entries.append(batch_entries)

    tp_size = mpu.get_tensor_model_parallel_world_size() if dist.is_initialized() else 1
    cp_size = mpu.get_context_parallel_world_size() if dist.is_initialized() else 1
    pp_size = mpu.get_pipeline_model_parallel_world_size() if dist.is_initialized() else 1

    payload: dict[str, Any] = {
        "rank": rank,
        "tp_size": tp_size,
        "cp_size": cp_size,
        "pp_size": pp_size,
        "seq_length": local_seq_len,
        "vocab_size": vocab_size,
        "batch_size": batch_size,
        "entries": all_entries,
        "summary": {
            "mean_logprob": sum_logprob / total_valid if total_valid > 0 else 0.0,
            "min_logprob": min_logprob if total_valid > 0 else 0.0,
            "max_logprob": max_logprob if total_valid > 0 else 0.0,
            "num_valid": total_valid,
        },
    }

    output_path = output_dir / f"rank_{rank}.json"
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"[logprob] rank={rank}: saved {total_valid} valid entries to {output_path}", flush=True)


def _entry_to_dict(entry: _LogprobEntry) -> dict[str, Any]:
    return {
        "global_position": entry.global_position,
        "token_id": entry.token_id,
        "logprob": entry.logprob,
        "is_valid": entry.is_valid,
    }
