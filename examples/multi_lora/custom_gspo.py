"""Sequence-level GSPO callback for Tinker's client-side custom-loss path.

Keep the usual target-token Datum objects. Capture the sampling-policy logprobs,
response-only masks, and one detached group advantage per sequence locally:

    loss_fn = make_gspo_loss(old_logprobs, response_masks, sequence_advantages)
    future = training.forward_backward_custom(datums, loss_fn)
    future.result()

This uses the public Tinker callback contract and also works with a compatible
Miles gateway. Do not transmit the captured masks as unsupported Datum fields.
For native Miles, use --advantage-estimator gspo with the same clipping values
and per-sequence loss normalization. This callback does not add KL regularization.

https://tinker-docs.thinkingmachines.ai/tinker/losses/custom/
"""

from collections.abc import Sequence

import torch


def make_gspo_loss(
    old_logprobs: Sequence[torch.Tensor],
    response_masks: Sequence[torch.Tensor],
    sequence_advantages: torch.Tensor,
    *,
    clip_low: float = 1e-4,
    clip_high: float = 1e-4,
):
    """Freeze the rollout inputs and return a forward_backward_custom callback."""
    if not old_logprobs or len(old_logprobs) != len(response_masks):
        raise ValueError("Supply one old-logprob vector and response mask per sequence")
    if not 0 < clip_low < 1 or not 0 < clip_high < 1:
        raise ValueError("GSPO clipping widths must be between zero and one")
    advantages = sequence_advantages.detach().clone()
    if advantages.shape != (len(old_logprobs),) or not torch.isfinite(advantages).all():
        raise ValueError("Supply one finite advantage per sequence")
    old = tuple(lp.detach().clone() for lp in old_logprobs)
    masks = tuple(mask.detach().bool().clone() for mask in response_masks)
    for lp, mask in zip(old, masks, strict=True):
        if lp.ndim != 1 or mask.shape != lp.shape or not mask.any() or not torch.isfinite(lp[mask]).all():
            raise ValueError("Each response mask must select finite old logprobs from a matching 1-D vector")

    def loss_fn(data, logprobs: list[torch.Tensor]):
        if len(data) != len(old) or len(logprobs) != len(old):
            raise ValueError("The returned logprobs must match the captured rollout batch")
        terms, ratios = [], []
        for lp, previous, mask, advantage in zip(logprobs, old, masks, advantages, strict=True):
            mask = mask.to(device=lp.device)
            if lp.shape != previous.shape or not torch.isfinite(lp[mask]).all():
                raise ValueError("The model must return matching finite response logprobs")
            delta = (lp[mask] - previous.to(lp)[mask]).mean()
            ratio = delta.exp()
            if not torch.isfinite(ratio):
                raise ValueError("The sequence importance ratio overflowed")
            advantage = advantage.to(lp)
            terms.append(-torch.minimum(ratio * advantage, ratio.clamp(1 - clip_low, 1 + clip_high) * advantage))
            ratios.append(ratio.detach())
        loss = torch.stack(terms).mean()
        ratio_values = torch.stack(ratios)
        return loss, {
            "gspo_loss": loss.detach().item(),
            "sequence_ratio_mean": ratio_values.mean().item(),
            "sequence_ratio_outside_clip_fraction": ((ratio_values < 1 - clip_low) | (ratio_values > 1 + clip_high))
            .float()
            .mean()
            .item(),
        }

    return loss_fn
