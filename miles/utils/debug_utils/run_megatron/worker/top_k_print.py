"""Top-K prediction printing for debugging forward pass outputs."""

import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def print_top_k(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    top_k: int,
    tokenizer_path: Path,
) -> None:
    """Load tokenizer and print top-k predictions across all ranks."""
    from transformers import AutoTokenizer

    tokenizer: Any = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    pad_token_id: int | None = tokenizer.pad_token_id or tokenizer.eos_token_id

    _print_top_predictions_all_ranks(
        logits=logits,
        input_ids=input_ids,
        top_k=top_k,
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
    )


def _print_top_predictions_all_ranks(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    top_k: int,
    tokenizer: object,
    pad_token_id: int | None = None,
) -> None:
    """Print top-k predictions from all ranks sequentially (rank 0 first, then rank 1, etc.)."""
    rank, world_size = _get_dist_info()

    if rank == 0:
        print(f"\n{'=' * 80}")
        print(f"Top-{top_k} Predictions (all ranks)")
        print(f"World size: {world_size}")
        print(f"{'=' * 80}")

    for r in range(world_size):
        _maybe_barrier()
        if rank == r:
            _print_top_predictions_for_rank(
                logits=logits,
                input_ids=input_ids,
                top_k=top_k,
                tokenizer=tokenizer,
                rank=rank,
                pad_token_id=pad_token_id,
            )
            sys.stdout.flush()

    _maybe_barrier()


def _print_top_predictions_for_rank(
        *,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        top_k: int,
        tokenizer: object,
        rank: int,
        pad_token_id: int | None = None,
) -> None:
    """Print top-k predictions for this rank, one line per position."""
    batch_size: int = logits.shape[0]
    seq_length: int = logits.shape[1]

    print(f"\n--- Rank {rank} (seq_len={seq_length}) ---")
    for b in range(batch_size):
        if batch_size > 1:
            print(f"  Batch {b}:")
        for pos in range(seq_length):
            if pad_token_id is not None and input_ids[b, pos].item() == pad_token_id:
                continue

            input_token: int = input_ids[b, pos].item()
            probs: torch.Tensor = torch.softmax(logits[b, pos], dim=-1)
            top_probs: torch.Tensor
            top_indices: torch.Tensor
            top_probs, top_indices = torch.topk(probs, top_k)

            input_str: str = _decode_token(tokenizer, token_id=input_token)
            preds: str = ", ".join(
                f"{_decode_token(tokenizer, token_id=idx.item())}({prob.item():.3f})"
                for prob, idx in zip(top_probs, top_indices, strict=True)
            )
            print(f"pos[{pos:3d}] {input_str!r:12s} -> {preds}")


def _decode_token(tokenizer: object, *, token_id: int) -> str:
    return tokenizer.decode([token_id]) if tokenizer else f"t{token_id}"


def _get_dist_info() -> tuple[int, int]:
    """Return (rank, world_size), defaulting to (0, 1) if not initialized."""
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _maybe_barrier() -> None:
    if dist.is_initialized():
        dist.barrier()
