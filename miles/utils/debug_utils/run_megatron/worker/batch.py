"""Input batch preparation and loss function for standalone Megatron forward/backward."""

from types import SimpleNamespace
from typing import Any

import torch


def prepare_batch(
    *,
    token_ids: list[int],
    batch_size: int,
    cp_rank: int = 0,
    cp_size: int = 1,
) -> dict[str, torch.Tensor]:
    """Build the batch dict for Megatron forward from pre-tokenized token IDs.

    Returns a dict containing:
    - input_ids: [batch_size, local_seq_len]
    - position_ids: [batch_size, local_seq_len]
    - attention_mask: [batch_size, 1, local_seq_len, local_seq_len] causal mask
    - labels: [batch_size, local_seq_len] (CP-aware next-token labels)
    - global_input_ids: [batch_size, full_seq_len] (unsliced, for reference)
    """
    seq_length: int = len(token_ids)

    token_tensor: torch.Tensor = torch.tensor(token_ids, dtype=torch.long, device="cuda")
    position_tensor: torch.Tensor = torch.arange(seq_length, dtype=torch.long, device="cuda")

    # Keep global copy before CP slicing (needed for CP-aware labels)
    global_tokens: torch.Tensor = token_tensor.clone()

    if cp_size > 1:
        from miles.backends.training_utils.cp_utils import slice_with_cp

        cp_kwargs: dict[str, object] = dict(
            pad_value=0,
            parallel_state=SimpleNamespace(cp_rank=cp_rank, cp_size=cp_size),
            qkv_format="bshd",
            max_seq_len=seq_length,
        )
        token_tensor = slice_with_cp(token_tensor, **cp_kwargs)
        position_tensor = slice_with_cp(position_tensor, **cp_kwargs)

    local_seq_len: int = token_tensor.shape[0]

    input_ids: torch.Tensor = token_tensor.unsqueeze(0).expand(batch_size, -1)
    position_ids: torch.Tensor = position_tensor.unsqueeze(0).expand(batch_size, -1)
    global_input_ids: torch.Tensor = global_tokens.unsqueeze(0).expand(batch_size, -1)

    labels: torch.Tensor = _build_labels(
        input_ids=input_ids,
        position_ids=position_ids,
        global_input_ids=global_input_ids,
        cp_size=cp_size,
    )

    attention_mask: torch.Tensor = torch.tril(
        torch.ones(batch_size, 1, local_seq_len, local_seq_len, dtype=torch.bool, device="cuda")
    )

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "global_input_ids": global_input_ids,
    }


def _build_labels(
    *,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    global_input_ids: torch.Tensor,
    cp_size: int,
) -> torch.Tensor:
    """Build next-token prediction labels, handling CP zigzag slicing.

    With CP>1, each rank sees non-contiguous positions (zigzag pattern).
    We use position_ids to gather the correct next-token from global_input_ids.
    Positions at the end of the global sequence get label -100 (ignored by loss).
    """
    if cp_size > 1:
        global_seq_len: int = global_input_ids.shape[1]
        valid_mask: torch.Tensor = position_ids < global_seq_len - 1
        next_pos: torch.Tensor = torch.where(valid_mask, position_ids + 1, torch.zeros_like(position_ids))
        labels: torch.Tensor = global_input_ids.gather(1, next_pos)
        labels = labels.masked_fill(~valid_mask, -100)
        return labels
    else:
        batch_size: int = input_ids.shape[0]
        return torch.cat(
            [input_ids[:, 1:], torch.full((batch_size, 1), -100, device=input_ids.device, dtype=input_ids.dtype)],
            dim=1,
        )


def loss_func(
    labels: torch.Tensor,
    output_tensor: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Cross-entropy loss for forward-backward pipeline schedule.

    Uses ignore_index=-100 to handle CP-aware label masking.
    """
    logits: torch.Tensor = output_tensor.float()
    vocab_size: int = logits.size(-1)

    loss: torch.Tensor = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )
    return loss, {"loss": loss.detach()}
