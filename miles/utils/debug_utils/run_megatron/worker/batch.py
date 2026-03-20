"""Input batch preparation and loss function for standalone Megatron forward/backward."""

from typing import Any

import torch


def prepare_batch(
    *,
    token_ids: list[int],
    batch_size: int,
    device: str | torch.device = "cuda",
) -> dict[str, torch.Tensor]:
    """Build the batch dict for Megatron forward from pre-tokenized token IDs.

    Returns a dict containing:
    - input_ids: [batch_size, seq_len]
    - position_ids: [batch_size, seq_len]
    - attention_mask: None (let flash attention handle causal masking)
    - labels: [batch_size, seq_len] (next-token labels)
    """
    seq_length: int = len(token_ids)

    token_tensor: torch.Tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
    position_tensor: torch.Tensor = torch.arange(seq_length, dtype=torch.long, device=device)

    input_ids: torch.Tensor = token_tensor.unsqueeze(0).expand(batch_size, -1)
    position_ids: torch.Tensor = position_tensor.unsqueeze(0).expand(batch_size, -1)

    labels: torch.Tensor = torch.cat(
        [input_ids[:, 1:], torch.full((batch_size, 1), -100, device=input_ids.device, dtype=input_ids.dtype)],
        dim=1,
    )

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": None,
        "labels": labels,
    }


def loss_func(
    labels: torch.Tensor,
    output_tensor: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Cross-entropy loss for forward-backward pipeline schedule.

    Uses ignore_index=-100 to handle label masking.
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
