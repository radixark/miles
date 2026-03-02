"""Pure computation of per-token log-probability info from forward-pass logits."""

from __future__ import annotations

from typing import Any

import torch


def compute_logprob_info(
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
