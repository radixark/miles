"""Input batch preparation and loss function for standalone Megatron forward/backward."""

from typing import Any

import torch


def prepare_batch(
    *,
    token_ids: list[int],
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Build the batch dict for Megatron forward from pre-tokenized token IDs."""
    seq_length: int = len(token_ids)

    input_ids: torch.Tensor = torch.tensor(
        [token_ids] * batch_size,
        dtype=torch.long,
        device="cuda",
    )
    position_ids: torch.Tensor = (
        torch.arange(seq_length, dtype=torch.long, device="cuda").unsqueeze(0).expand(batch_size, -1)
    )
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
