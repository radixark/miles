"""Pure computation of per-token log-probability info from forward-pass logits."""

from __future__ import annotations

from typing import Any

import torch


def compute_logprob_info(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    position_ids: torch.Tensor,
) -> dict[str, Any] | None:
    """Compute per-token logprob entries and summary statistics.

    Returns None if logits don't look like vocab logits (e.g. critic model).

    Args:
        logits: [batch_size, local_seq_len, vocab_size] — must already be gathered across TP.
        labels: [batch_size, local_seq_len], -100 = ignore.
        position_ids: [batch_size, local_seq_len], global positions.
    """
    if logits.ndim < 3 or logits.size(-1) == 1:
        return None

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

    return {
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
