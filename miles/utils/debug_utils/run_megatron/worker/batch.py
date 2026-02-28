"""Input batch preparation and loss function for standalone Megatron forward/backward."""

import argparse
from typing import Any

import torch


def prepare_batch(
    args: argparse.Namespace,
    prompt_text: str,
) -> dict[str, torch.Tensor]:
    """Tokenize prompt and build the batch dict for Megatron forward."""
    from megatron.training.global_vars import get_tokenizer

    tokenizer = get_tokenizer()
    token_ids: list[int] = tokenizer.tokenize(prompt_text)

    seq_length: int = args.seq_length
    batch_size: int = args.micro_batch_size

    if len(token_ids) > seq_length:
        token_ids = token_ids[:seq_length]
    elif len(token_ids) < seq_length:
        pad_id: int = tokenizer.pad if hasattr(tokenizer, "pad") and tokenizer.pad is not None else tokenizer.eod
        token_ids = token_ids + [pad_id] * (seq_length - len(token_ids))

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
