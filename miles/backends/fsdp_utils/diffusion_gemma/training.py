"""Fixed-data DiffusionGemma batches and globally token-normalized SFT losses."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DiffusionBatch:
    input_ids: torch.Tensor
    decoder_input_ids: torch.Tensor
    targets: torch.Tensor
    valid_mask: torch.Tensor
    position_ids: torch.Tensor
    decoder_position_ids: torch.Tensor
    encoder_attention_mask: dict[str, torch.Tensor]
    decoder_attention_mask: dict[str, torch.Tensor]
    self_conditioning_mask: torch.Tensor

    def model_kwargs(self) -> dict:
        return {
            name: getattr(self, name)
            for name in (
                "input_ids",
                "decoder_input_ids",
                "position_ids",
                "decoder_position_ids",
                "encoder_attention_mask",
                "decoder_attention_mask",
                "self_conditioning_mask",
            )
        }


def prepare_batch(
    *,
    tokens: list[torch.Tensor],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    canvas_length: int,
    sliding_window: int,
    eos_token_id: int,
    pad_token_id: int,
    vocab_size: int,
    noise_epsilon: float,
    self_conditioning_probability: float,
    seed: int,
) -> DiffusionBatch:
    """Select one response block per row; fill its terminal partial block with EOS.

    Whole-response decoder attention is block diagonal, so evaluating only the
    sampled canvas yields the same selected-token objective with less memory.
    """
    if not tokens or canvas_length < 1 or sliding_window < 1:
        raise ValueError("nonempty batches, positive canvas length and sliding window are required")
    if not 0 < noise_epsilon <= 1 or not 0 <= self_conditioning_probability <= 1:
        raise ValueError("invalid corruption epsilon or self-conditioning probability")
    device = tokens[0].device
    generator = torch.Generator(device=device).manual_seed(seed)
    clean_rows, targets, starts = [], [], []
    for token_ids, response_length, mask in zip(tokens, response_lengths, loss_masks, strict=True):
        if token_ids.ndim != 1 or not 0 < response_length < token_ids.numel():
            raise ValueError("each sample needs a nonempty prompt and response")
        if mask.numel() != response_length or not bool((mask == 1).all()):
            raise ValueError("DiffusionGemma SFT requires one contiguous, fully supervised final response")
        padding = (-response_length) % canvas_length
        clean = F.pad(token_ids, (0, padding), value=eos_token_id)
        blocks = (response_length + padding) // canvas_length
        block = int(torch.randint(blocks, (), device=device, generator=generator))
        start = token_ids.numel() - response_length + block * canvas_length
        clean_rows.append(clean)
        targets.append(clean[start : start + canvas_length])
        starts.append(start)
    lengths = torch.tensor([row.numel() for row in clean_rows], device=device)
    input_ids = torch.nn.utils.rnn.pad_sequence(clean_rows, batch_first=True, padding_value=pad_token_id)
    target_ids = torch.stack(targets)
    positions = torch.arange(input_ids.shape[1], device=device)
    valid = positions[None, :] < lengths[:, None]
    block_starts = torch.tensor(starts, device=device)
    encoder_masks, decoder_masks = _attention_masks(
        positions=positions,
        valid=valid,
        block_starts=block_starts,
        canvas_length=canvas_length,
        sliding_window=sliding_window,
    )
    timestep = noise_epsilon + (1 - noise_epsilon) * torch.rand(len(tokens), 1, device=device, generator=generator)
    corrupt = torch.rand(target_ids.shape, device=device, generator=generator) < timestep
    replacements = torch.randint(vocab_size, target_ids.shape, device=device, generator=generator)
    noisy = torch.where(corrupt, replacements, target_ids)
    conditioning = torch.rand(len(tokens), device=device, generator=generator) < self_conditioning_probability
    return DiffusionBatch(
        input_ids=input_ids,
        decoder_input_ids=noisy,
        targets=target_ids,
        valid_mask=valid,
        position_ids=positions[None, :].expand(len(tokens), -1),
        decoder_position_ids=block_starts[:, None] + torch.arange(canvas_length, device=device)[None, :],
        encoder_attention_mask=encoder_masks,
        decoder_attention_mask=decoder_masks,
        self_conditioning_mask=conditioning,
    )


def _attention_masks(*, positions, valid, block_starts, canvas_length, sliding_window):
    # True means attend. No encoder key from the sampled block or any later block
    # is visible; encoder representations themselves are strictly causal.
    causal = positions[:, None] >= positions[None, :]
    encoder_full = causal[None, :, :] & valid[:, None, :]
    encoder_local = encoder_full & (positions[:, None] - positions[None, :] < sliding_window)[None, :, :]
    clean_full = (positions[None, :] < block_starts[:, None]) & valid
    clean_local = clean_full & (positions[None, :] >= block_starts[:, None] - sliding_window + 1)
    canvas = torch.ones(len(valid), canvas_length, canvas_length, device=positions.device, dtype=torch.bool)
    return (
        {"full_attention": encoder_full[:, None], "sliding_attention": encoder_local[:, None]},
        {
            kind: torch.cat((clean[:, None, :].expand(-1, canvas_length, -1), canvas), dim=-1)[:, None]
            for kind, clean in (("full_attention", clean_full), ("sliding_attention", clean_local))
        },
    )


def loss_sums(*, encoder_logits, decoder_logits, batch: DiffusionBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """Unnormalized same-position canvas CE and independently shifted encoder CE.

    Score every canvas token, including retained clean tokens and EOS fill.
    There is no inverse-time weighting or corrupted-token-only denominator.
    """
    diffusion = F.cross_entropy(decoder_logits.flatten(0, 1).float(), batch.targets.flatten(), reduction="sum")
    valid_ar = batch.valid_mask[:, :-1] & batch.valid_mask[:, 1:]
    encoder = F.cross_entropy(
        encoder_logits[:, :-1][valid_ar].float(),
        batch.input_ids[:, 1:][valid_ar],
        reduction="sum",
    )
    return diffusion, encoder


def token_counts(batch: DiffusionBatch) -> torch.Tensor:
    return torch.stack(
        (
            torch.tensor(batch.targets.numel(), device=batch.input_ids.device),
            (batch.valid_mask[:, :-1] & batch.valid_mask[:, 1:]).sum(),
        )
    )


def normalize_loss(*, sums, token_counts, dp_size: int, encoder_loss_weight: float) -> torch.Tensor:
    """FSDP averages gradients: multiply global-denominator local sums by DP size."""
    return dp_size * (sums[0] / token_counts[0] + encoder_loss_weight * sums[1] / token_counts[1])
