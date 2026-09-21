"""Compute domain weights once on the finalized rollout, before DP partitioning."""

import math
from collections import defaultdict

import torch

from miles.utils.types import Sample


def parse_domain_targets(entries):
    targets = {}
    for entry in entries or []:
        name, sep, value = entry.partition("=")
        if not sep or not name or name in targets:
            raise ValueError(f"Invalid or duplicate OPD domain target {entry!r}; expected NAME=WEIGHT")
        weight = float(value)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("OPD domain targets must be finite and positive")
        targets[name] = weight
    total = sum(targets.values())
    return {name: value / total for name, value in targets.items()}


def set_domain_weights(args, samples: list[Sample]):
    mode = getattr(args, "opd_domain_balance", "none")
    for sample in samples:
        sample.opd_loss_weights = torch.ones(sample.response_length, dtype=torch.float32)
    if mode == "none":
        return
    targets = parse_domain_targets(args.opd_domain_targets)
    mass, gap_sums, token_counts = defaultdict(float), defaultdict(float), defaultdict(float)
    domains = []
    for sample in samples:
        domain = sample.metadata.get("domain")
        if domain not in targets:
            raise ValueError(f"Unknown OPD domain {domain!r}; configured targets: {sorted(targets)}")
        domains.append(domain)
        mask = torch.tensor(sample.loss_mask if sample.loss_mask is not None else [1] * sample.response_length)
        if sample.remove_sample:
            mask.zero_()
        count = mask.sum().item()
        mass[domain] += count if args.calculate_per_token_loss else float(count > 0)
        token_counts[domain] += count
        if mode == "gap":
            old = sample.opd_candidate_old_log_probs
            teacher = sample.opd_candidate_teacher_log_probs
            gap = (old.softmax(-1) * (teacher - old)).sum(-1).abs()
            gap_sums[domain] += (gap * mask).sum().item()
    if any(mass[domain] == 0 for domain in targets):
        raise ValueError("Every configured OPD domain must have active samples in each rollout")
    total_mass = sum(mass.values())
    fractions = {d: mass[d] / total_mass for d in targets}
    weights = {d: targets[d] / fractions[d] for d in targets}
    if mode == "gap":
        gaps = {d: gap_sums[d] / token_counts[d] for d in targets}
        mean_gap = sum(gaps.values()) / len(gaps)
        if mean_gap > 0:
            weights = {
                d: w * min(20.0, max(0.05, (gaps[d] / mean_gap) ** args.opd_gap_alpha)) for d, w in weights.items()
            }
    scale = sum(fractions[d] * weights[d] for d in weights)
    for sample, domain in zip(samples, domains, strict=True):
        sample.opd_loss_weights.fill_(weights[domain] / scale)
        sample.train_metadata = {**(sample.train_metadata or {}), "opd_domain": domain}
