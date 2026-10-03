"""Categorical value predictions and HL-Gauss targets for PPO critics."""

from argparse import Namespace

import torch
import torch.nn.functional as F


def value_bin_centers(args: Namespace, device: torch.device) -> torch.Tensor:
    width = (args.critic_value_max - args.critic_value_min) / args.critic_value_bins
    return args.critic_value_min + (torch.arange(args.critic_value_bins, device=device) + 0.5) * width


def decode_values(logits: torch.Tensor, args: Namespace) -> torch.Tensor:
    if getattr(args, "critic_value_bins", 1) == 1:
        assert logits.size(-1) == 1, f"{logits.shape}"
        return logits.squeeze(-1).float()
    assert logits.size(-1) == args.critic_value_bins, f"{logits.shape}"
    centers = value_bin_centers(args, logits.device)
    return (logits.float().softmax(dim=-1) * centers).sum(dim=-1)


def hl_gauss_loss(logits: torch.Tensor, returns: torch.Tensor, args: Namespace) -> torch.Tensor:
    """Cross entropy against Gaussian mass integrated over each value bin."""
    width = (args.critic_value_max - args.critic_value_min) / args.critic_value_bins
    boundaries = (
        args.critic_value_min
        + torch.arange(1, args.critic_value_bins, device=logits.device, dtype=torch.float32) * width
    )
    cdf = torch.special.ndtr((boundaries - returns.float().unsqueeze(-1)) / args.critic_value_sigma)
    cumulative = torch.cat((torch.zeros_like(cdf[..., :1]), cdf, torch.ones_like(cdf[..., :1])), dim=-1)
    target = (cumulative[..., 1:] - cumulative[..., :-1]).clamp_min(0)
    return -(target * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1)
