"""Importance sampling for score centering (arXiv:2609.20807, Appendix A.2/A.3).

Each scheme applies an importance-sampling weight function f to the ratio
r = p / q. It supplies the terms of Eq. 12, whose head coefficient is
q * f(p / q) - alpha * p with tail scale alpha = rho * f(1 / rho), and the
sampled-token weight of the loss:

- ``head_mass(p, q, log_ratio)``: q * f(p / q) on the head candidates;
- ``tail_scale(rho, dtype)``: alpha = rho * f(1 / rho), where rho is the
  sampler-to-trainer tail-mass ratio;
- ``sample_weight(log_ratio)``: f(r) for the sampled token.

No term exponentiates an unbounded ratio.
"""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NoImportanceSampling:
    """Vanilla score centering: f(r) = 1."""

    def head_mass(self, p: torch.Tensor, q: torch.Tensor, log_ratio: torch.Tensor) -> torch.Tensor:
        return q

    def tail_scale(self, rho: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return rho

    def sample_weight(self, log_ratio: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(log_ratio)


@dataclass(frozen=True)
class TruncatedImportanceSampling:
    """Truncated importance sampling (TIS): f(r) = min(r, clip)."""

    clip: float

    def head_mass(self, p: torch.Tensor, q: torch.Tensor, log_ratio: torch.Tensor) -> torch.Tensor:
        # q * min(p/q, c), including q=0, without 0 * inf.
        return torch.minimum(p, self.clip * q)

    def tail_scale(self, rho: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return (self.clip * rho).clamp_max(1)

    def sample_weight(self, log_ratio: torch.Tensor) -> torch.Tensor:
        if torch.isnan(log_ratio).any():
            raise ValueError("Score-centering importance log-ratio contains NaN")
        return log_ratio.clamp(max=math.log(self.clip)).exp()


@dataclass(frozen=True)
class MaskedImportanceSampling:
    """Masked importance sampling (MIS): f(r) = r for low <= r <= high, else 0."""

    low: float
    high: float

    def _inside(self, log_ratio: torch.Tensor) -> torch.Tensor:
        """Whether low <= r <= high, tested on log(r)."""
        return (log_ratio >= math.log(self.low)) & (log_ratio <= math.log(self.high))

    def head_mass(self, p: torch.Tensor, q: torch.Tensor, log_ratio: torch.Tensor) -> torch.Tensor:
        return torch.where(self._inside(log_ratio), p, 0.0)

    def tail_scale(self, rho: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return ((rho >= 1 / self.high) & (rho <= 1 / self.low)).to(dtype)

    def sample_weight(self, log_ratio: torch.Tensor) -> torch.Tensor:
        if torch.isnan(log_ratio).any():
            raise ValueError("Score-centering importance log-ratio contains NaN")
        return torch.where(self._inside(log_ratio), log_ratio.clamp(max=math.log(self.high)).exp(), 0.0)


def importance_sampling(
    mode: str, *, tis_clip: float, mis_low: float, mis_high: float
) -> NoImportanceSampling | TruncatedImportanceSampling | MaskedImportanceSampling:
    """Validate the parameters of ``mode`` and return its importance sampling."""
    if mode == "tis" and (not math.isfinite(tis_clip) or tis_clip <= 0):
        raise ValueError("Score-centering TIS clip must be positive and finite")
    if mode == "mis" and (not math.isfinite(mis_low) or not math.isfinite(mis_high) or not 0 < mis_low <= mis_high):
        raise ValueError("Score-centering MIS bounds must be finite with 0 < low <= high")
    if mode == "none":
        return NoImportanceSampling()
    if mode == "tis":
        return TruncatedImportanceSampling(clip=tis_clip)
    if mode == "mis":
        return MaskedImportanceSampling(low=mis_low, high=mis_high)
    raise ValueError(f"Unknown score-centering importance weighting: {mode}")


def importance_weights(
    log_ratio: torch.Tensor,
    mode: str,
    *,
    tis_clip: float = 2.0,
    mis_low: float = 0.5,
    mis_high: float = 5.0,
) -> torch.Tensor:
    """Evaluate token-level weights without exponentiating unbounded ratios."""
    sampling = importance_sampling(mode, tis_clip=tis_clip, mis_low=mis_low, mis_high=mis_high)
    return sampling.sample_weight(log_ratio)
