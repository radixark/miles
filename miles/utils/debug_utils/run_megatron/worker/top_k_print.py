"""Top-K prediction printing for debugging forward pass outputs."""

import sys

import torch
import torch.distributed as dist


def print_top_predictions_for_rank(
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

            input_str: str = tokenizer.decode([input_token]) if tokenizer else f"t{input_token}"
            preds: str = ", ".join(
                f"{tokenizer.decode([idx.item()]) if tokenizer else f't{idx.item()}'}({prob.item():.3f})"
                for prob, idx in zip(top_probs, top_indices, strict=True)
            )
            print(f"pos[{pos:3d}] {input_str!r:12s} -> {preds}")


def print_top_predictions_all_ranks(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    top_k: int,
    tokenizer: object,
    pad_token_id: int | None = None,
) -> None:
    """Print top-k predictions from all ranks sequentially (rank 0 first, then rank 1, etc.)."""
    rank: int = dist.get_rank() if dist.is_initialized() else 0
    world_size: int = dist.get_world_size() if dist.is_initialized() else 1

    if rank == 0:
        print(f"\n{'=' * 80}")
        print(f"Top-{top_k} Predictions (all ranks)")
        print(f"World size: {world_size}")
        print(f"{'=' * 80}")

    for r in range(world_size):
        if dist.is_initialized():
            dist.barrier()
        if rank == r:
            print_top_predictions_for_rank(
                logits=logits,
                input_ids=input_ids,
                top_k=top_k,
                tokenizer=tokenizer,
                rank=rank,
                pad_token_id=pad_token_id,
            )
            sys.stdout.flush()

    if dist.is_initialized():
        dist.barrier()
