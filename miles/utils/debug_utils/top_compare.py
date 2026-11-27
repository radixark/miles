"""Tensor comparisons of TOP's backward gates: whole-tensor metrics and their localized maxima (the worst token row,
the worst expert), shared by the cross-layout check and the VJP verdict."""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence

import torch

# A row (or expert) is judged relative to max(its own reference norm, this fraction of the tensor's RMS row norm), so a
# row whose reference gradient is nearly zero (a prompt token far from every scored token) does not read as a 100%
# error on rounding alone.
LOCAL_NORM_FLOOR = 1e-2

# |sum_t g_t| / sum_t |g_t| below this: the gradient is a near-cancelling token sum, whose error is judged relative to
# sum_t |g_t| (an absolute scale); relative to its nearly cancelled net, rounding alone moves it by multiples.
CANCELLATION_RATIO = 0.1

# Per-expert parameters: TOP's (``...mlp.experts.linear_fc1.weight7``) and DeepSeek-V4's reference
# (``...ffn.experts.7.w1.weight``); HF's fused experts stack every expert along dim 0.
_PER_EXPERT_RES = (
    re.compile(r"^(?P<group>.*\.mlp\.experts)\.linear_fc[12]\.weight(?P<expert>\d+)$"),
    re.compile(r"^(?P<group>.*\.experts)\.(?P<expert>\d+)\.w[123]\.weight$"),
)
_STACKED_EXPERTS_RE = re.compile(r"^(?P<group>.*\.mlp\.experts)\.(?:gate_up_proj|down_proj)$")


def compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    """``a`` against the reference ``b``: relative L2 distance, cosine, norm ratio |a| / |b| and max abs difference."""
    a, b = a.double().flatten(), b.double().flatten()
    a_norm, b_norm = float(a.norm()), float(b.norm())
    return dict(
        rel_l2=float((a - b).norm()) / max(b_norm, 1e-300),
        cosine=float(a @ b) / max(a_norm * b_norm, 1e-300),
        norm_ratio=a_norm / max(b_norm, 1e-300),
        max_abs=float((a - b).abs().max()) if a.numel() else 0.0,
    )


def worst(comparisons: Sequence[dict]) -> dict[str, float]:
    """The comparison with the largest relative L2 (the worst microbatch)."""
    return max(comparisons, key=lambda c: c["rel_l2"])


def _local_errors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Each row's distance relative to its own reference norm, floored at LOCAL_NORM_FLOOR of the RMS row norm."""
    a, b = a.double(), b.double()
    norms = b.norm(dim=-1)
    floor = LOCAL_NORM_FLOOR * float(norms.square().mean().sqrt()) if norms.numel() else 0.0
    return (a - b).norm(dim=-1) / norms.clamp_min(max(floor, 1e-300))


def worst_row(a: torch.Tensor, b: torch.Tensor) -> dict:
    """The token row (last dimension = hidden) of ``a`` farthest from ``b``'s, each relative to its own row norm."""
    errors = _local_errors(a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1]))
    if not errors.numel():
        return dict(row=None, rel_l2=0.0)
    row = int(errors.argmax())
    return dict(row=row, rel_l2=float(errors[row]))


def expert_vectors(grads: Mapping[str, torch.Tensor]) -> dict[str, dict[int, torch.Tensor]]:
    """{expert group: {expert: every parameter gradient of that expert, flattened and concatenated by name}}."""
    parts: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for name in sorted(grads):
        if (match := _STACKED_EXPERTS_RE.match(name)) is not None:
            for expert, grad in enumerate(grads[name].unbind(0)):
                parts[match["group"]][expert].append(grad.flatten())
            continue
        for pattern in _PER_EXPERT_RES:
            if (match := pattern.match(name)) is not None:
                parts[match["group"]][int(match["expert"])].append(grads[name].flatten())
                break
    return {group: {e: torch.cat(p) for e, p in experts.items()} for group, experts in parts.items()}


def worst_experts(top: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor]) -> dict[str, dict]:
    """Per expert group, the expert whose gradient is farthest from the reference's, relative to its own norm."""
    top_experts, reference_experts = expert_vectors(top), expert_vectors(reference)
    out = {}
    for group, experts in reference_experts.items():
        ids = sorted(experts)
        errors = _local_errors(
            torch.stack([top_experts[group][e] for e in ids]), torch.stack([experts[e] for e in ids])
        )
        index = int(errors.argmax())
        out[group] = dict(expert=ids[index], rel_l2=float(errors[index]))
    return out


def summary(comparisons: Sequence[dict]) -> dict:
    """Median and worst relative L2 and the lowest cosine over the gradients that are not zero on both sides."""
    # A zero gradient against a nonzero reference has norm ratio 0 and relative L2 1; zero on both sides has both 0.
    nonzero = [c for c in comparisons if c["norm_ratio"] > 0 or c["rel_l2"] > 0]
    return dict(
        rel_l2_median=statistics.median(c["rel_l2"] for c in nonzero) if nonzero else None,
        rel_l2_max=max((c["rel_l2"] for c in nonzero), default=None),
        cosine_min=min((c["cosine"] for c in nonzero), default=None),
    )


def vs_token_sum(a: torch.Tensor, b: torch.Tensor, token_sum: float) -> float:
    """|a - b| relative to sum_t |g_t| of the reference ``b``, a token sum."""
    return float((a.double() - b.double()).norm()) / max(token_sum, 1e-300)
