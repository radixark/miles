"""Compute and save per-token log-probabilities from forward-pass logits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import mpu


def _get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


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
    rank = _get_rank()

    if logits.ndim < 3 or logits.size(-1) == 1:
        print(
            f"[logprob] rank={rank}: skipping logprob — logits shape {logits.shape} "
            f"does not look like vocab logits (critic model?)",
            flush=True,
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size, local_seq_len, vocab_size = logits.shape
    log_probs = torch.log_softmax(logits.float(), dim=-1)

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

            batch_entries.append({
                "global_position": position_ids[b, s].item(),
                "token_id": label_id if is_valid else -1,
                "logprob": lp,
                "is_valid": is_valid,
            })
        all_entries.append(batch_entries)

    payload: dict[str, Any] = {
        "rank": rank,
        "tp_size": mpu.get_tensor_model_parallel_world_size() if dist.is_initialized() else 1,
        "cp_size": mpu.get_context_parallel_world_size() if dist.is_initialized() else 1,
        "pp_size": mpu.get_pipeline_model_parallel_world_size() if dist.is_initialized() else 1,
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
