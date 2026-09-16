"""OPSD's per-entry clipping of the shared forward-KL contributions."""

import math

import torch

from miles.backends.training_utils.loss_hub.opd import iter_forward_kl_terms


def loss_function(args, batch, logits, sum_of_sample_mean):
    clip = args.opsd_kl_clip
    if not math.isfinite(clip) or clip <= 0:
        raise ValueError("opsd_kl_clip must be finite and positive.")
    losses, coverage, clipped = [], [], []
    for terms, probabilities in iter_forward_kl_terms(args, batch, logits):
        losses.append(terms.clamp(max=clip).sum(dim=-1))
        coverage.append(probabilities.sum(dim=-1))
        clipped.append((terms > clip).float().mean(dim=-1))
    kl = sum_of_sample_mean(torch.cat(losses))
    loss = args.opd_kl_coef * kl
    return loss, {
        "loss": loss.detach(),
        "opd_forward_kl": kl.detach(),
        "opd_teacher_coverage": sum_of_sample_mean(torch.cat(coverage)).detach(),
        "opd_kl_clipfrac": sum_of_sample_mean(torch.cat(clipped)).detach(),
    }
