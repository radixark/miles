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
    """Build the batch dict for Megatron forward from pre-tokenized token IDs."""
    seq_length: int = len(token_ids)

    token_tensor: torch.Tensor = torch.tensor(token_ids, dtype=torch.long, device="cuda")
    position_tensor: torch.Tensor = torch.arange(seq_length, dtype=torch.long, device="cuda")

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

    input_ids: torch.Tensor = token_tensor.unsqueeze(0).expand(batch_size, -1)
    position_ids: torch.Tensor = position_tensor.unsqueeze(0).expand(batch_size, -1)
    labels: torch.Tensor = input_ids.clone()

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "labels": labels,
    }


def loss_func(
    labels: torch.Tensor,
    output_tensor: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Simple cross-entropy loss for forward-backward pipeline schedule."""
    logits: torch.Tensor = output_tensor.float()
    shift_logits: torch.Tensor = logits[..., :-1, :].contiguous()
    shift_labels: torch.Tensor = labels[..., 1:].contiguous()

    loss: torch.Tensor = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    return loss, {"loss": loss.detach()}
