"""Cross-layout backward check of TOP's gates: every phase of a gate against its ``cp1`` phase, per captured tensor.

The phases train the same step-1 rollouts from the same weights (``top_capture.check_same_rollouts`` first), so their
captured gradients differ only by reduction order. Each packed sequence is found by its token ids in every phase (CP
ranks hold consecutive slices of the packed stream), and per captured layer the input gradient and output cotangent
are compared over every sequence, the router and index selections bitwise; the parameter gradients are compared after
Megatron's reduce. A bug in one tensor, in one expert or in a few rows (CP chunk boundaries) shows here although the
step's scalar gradient norm, which the gates still compare for the uncaptured layers, barely moves.

Activation gradients carry each phase's loss normalization (its DP and CP averaging) until the reduce, so they are
compared after one scale per phase, fitted over every captured activation gradient and reported; a scale that differs
between layers or rows still fails, and parameter gradients are compared unscaled.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence

import torch

from miles.utils.debug_utils import top_capture
from miles.utils.debug_utils.top_compare import CANCELLATION_RATIO, compare, vs_token_sum, worst_experts, worst_row
from miles.utils.debug_utils.vendor_reference import read_config_json

# Bars of a phase against cp1: twice the largest value measured on known-good captures of each family's e2e precision
# test (a same-layout rerun is bitwise, so it is not the floor). Measured: relative L2 0.77%,
# 1 - cosine 2.6e-5, |norm ratio - 1| 0.66%, worst row 2.8% (Qwen3.5's CP4), worst expert 0.54%.
LAYOUT_REL_L2_BAR = 0.016
LAYOUT_COSINE_MIN = 0.99994
# |norm ratio - 1|
LAYOUT_NORM_RATIO_BAND = 0.014
LAYOUT_ROW_BAR = 0.057
LAYOUT_EXPERT_BAR = 0.011
# A near-cancelling parameter (the cp1 verdict's token_sums): its difference relative to sum_t |g_t|. Twice the largest
# measured, 1.33e-4 (DeepSeek-V4's attention mHC bias, PP2 against CP1, at 4 prompts x 2 samples: fewer tokens leave a
# near-cancelling sum noisier against its terms than at 4 x 4, which measured <= 4e-5).
LAYOUT_TOKEN_SUM_BAR = 2.7e-4

_GRADS = ("input_grad", "output_grad")
_SELECTIONS = ("routing", "indices")


def layout_verdict(
    *,
    model_path: str,
    dump_dirs: Mapping[str, str],
    layers: Sequence[int],
    reference: str = "cp1",
    stage_layers: Mapping[str, Sequence[int]] | None = None,
    rollout: int = 0,
    token_sums: Mapping[str, float] | None = None,
) -> dict:
    """Every phase in ``dump_dirs`` against ``reference``; ``stage_layers`` names each PP phase's layers per stage.
    ``token_sums`` (the cp1 verdict's, {"decoder.layers.N....": sum_t |g_t|}) names the parameter gradients that are
    token sums: one that nearly cancels is judged relative to sum_t |g_t|, as reduction order moves its net by
    multiples of its own size."""
    config = read_config_json(model_path)
    config = getattr(
        config, "text_config", config
    )  # config-access-exempt: HF configs may wrap the text model's config in text_config
    num_experts = getattr(config, "num_experts", None) or getattr(
        config, "n_routed_experts", 0
    )  # config-access-exempt: TransformerConfig versions and model families differ in optional num_experts
    samples = top_capture.rollout_samples(dump_dirs[reference], rollout)
    captured = {
        phase: _phase(
            dump_dir,
            num_layers=config.num_hidden_layers,
            num_experts=num_experts,
            layers=layers,
            samples=samples,
            stage_layers=(stage_layers or {}).get(phase),
            rollout=rollout,
        )
        for phase, dump_dir in dump_dirs.items()
    }
    base = captured[reference]
    verdict = {
        phase: _compare_phase(captured[phase], base, layers, token_sums or {})
        for phase in dump_dirs
        if phase != reference
    }
    verdict = dict(phases=verdict, failures=[f"{p}: {f}" for p, r in verdict.items() for f in r.pop("failures")])
    print(json.dumps(verdict, indent=1))
    return verdict


def _phase(dump_dir, *, num_layers, num_experts, layers, samples, stage_layers, rollout) -> dict:
    """{"params": {TOP name: reduced grad}, "sequences": {sample index: {layer: {kind: rows}}}}"""
    ranks = top_capture.load_captures(
        dump_dir, num_layers=num_layers, layers=layers, rollout=rollout, stage_layers=stage_layers
    )
    return dict(
        params=top_capture.global_param_grads(ranks, num_experts),
        sequences=_sequences(top_capture.keyed_microbatches(ranks), samples, layers),
    )


def _sequences(views: Mapping[tuple, dict], samples: Sequence[dict], layers: Sequence[int]) -> dict[int, dict]:
    """{sample index: {layer: {kind: that sequence's rows}}}: CP ranks' views joined into the packed stream of each
    (DP rank, step), and every sample found in it by its token ids, starting at a sequence boundary; padding matches
    none."""
    by_tokens = defaultdict(list)
    for position, sample in enumerate(samples):
        by_tokens[tuple(sample["tokens"])].append(position)
    lengths = sorted({len(tokens) for tokens in by_tokens})
    streams = defaultdict(list)
    for (dp_rank, cp_rank, step), view in views.items():
        streams[dp_rank, step].append((cp_rank, view))
    out = {}
    for parts in streams.values():
        parts = [view for _, view in sorted(parts, key=lambda part: part[0])]
        tokens = torch.cat([view["input_ids"] for view in parts])
        rows = {
            number: {
                kind: torch.cat([view[number][kind] for view in parts])
                for kind in (*_GRADS, *_SELECTIONS)
                if kind in parts[0][number]
            }
            for number in layers
        }
        for lo in _starts(parts, len(tokens)):
            for length in lengths:
                matches = by_tokens.get(tuple(tokens[lo : lo + length].tolist()))
                if matches:
                    out[matches.pop(0)] = {
                        number: {kind: value[lo : lo + length] for kind, value in kinds.items()}
                        for number, kinds in rows.items()
                    }
                    break
    return out


def _starts(parts: Sequence[dict], total: int) -> list[int]:
    """Where a sequence can start in the joined stream. Megatron's all-gather CP gives every rank the stream's bounds;
    GDN TOP's gives each rank its own rows' bounds (a sequence cut by a rank boundary starts on one rank and ends on
    the next)."""
    bounds = [view["cu_seqlens"].tolist() for view in parts]
    if all(b == bounds[0] for b in bounds) and bounds[0][-1] == total:
        return bounds[0][:-1]
    starts, offset = set(), 0
    for view, local in zip(parts, bounds, strict=True):
        starts |= {offset + b for b in local[:-1]}
        offset += len(view["input_ids"])
    return sorted(starts)


def _activation_scale(phase: dict, base: dict, common: Sequence[int], layers: Sequence[int]) -> float:
    """The least-squares scale of this phase's activation gradients onto the reference phase's."""
    dot = norm = 0.0
    for index in common:
        for number in layers:
            for kind in _GRADS:
                a, b = (
                    phase["sequences"][index][number][kind].double(),
                    base["sequences"][index][number][kind].double(),
                )
                dot, norm = dot + float((a * b).sum()), norm + float(a.square().sum())
    return dot / norm if norm else 1.0


def _compare_phase(phase: dict, base: dict, layers: Sequence[int], token_sums: Mapping[str, float]) -> dict:
    failures = []
    missing = sorted(set(base["sequences"]) ^ set(phase["sequences"]))
    if missing:
        failures.append(f"samples {missing} captured in one phase only")
    common = sorted(set(base["sequences"]) & set(phase["sequences"]))
    scale = _activation_scale(phase, base, common, layers)
    record = dict(activation_grad_scale=scale, layers={}, params={})
    for number in layers:
        record["layers"][number] = {}
        for kind in _GRADS:
            a = torch.cat([phase["sequences"][i][number][kind] for i in common]).double() * scale
            b = torch.cat([base["sequences"][i][number][kind] for i in common]).double()
            comparison = dict(compare(a, b), worst_row=worst_row(a, b))
            record["layers"][number][kind] = comparison
            if _fails(comparison) or comparison["worst_row"]["rel_l2"] > LAYOUT_ROW_BAR:
                failures.append(f"layer {number} {kind}")
        for kind in _SELECTIONS:
            differing = [i for i in common if not _same_selection(phase, base, i, number, kind)]
            if differing:
                failures.append(f"layer {number} {kind} differs on samples {differing}")
    if set(phase["params"]) != set(base["params"]):
        failures.append(f"parameter sets differ: {sorted(set(phase['params']) ^ set(base['params']))}")
    for name in sorted(set(phase["params"]) & set(base["params"])):
        a, b = phase["params"][name], base["params"][name]
        record["params"][name] = comparison = compare(a, b)
        token_sum = token_sums.get(name.partition("module.module.")[2] or name)
        if token_sum is not None and float(b.double().norm()) < CANCELLATION_RATIO * token_sum:
            comparison["judged_vs_token_sum"] = vs_token_sum(a, b, token_sum)
        if _fails(comparison, scalar=b.numel() == 1):
            failures.append(name)
    record["experts"] = worst_experts(phase["params"], base["params"])
    failures += [
        f"{group}: expert {worst['expert']} is {worst['rel_l2']:.2%} off"
        for group, worst in record["experts"].items()
        if worst["rel_l2"] > LAYOUT_EXPERT_BAR
    ]
    return dict(record, failures=failures)


def _same_selection(phase: dict, base: dict, index: int, number: int, kind: str) -> bool:
    a, b = phase["sequences"][index][number].get(kind), base["sequences"][index][number].get(kind)
    return (a is None) == (b is None) and (a is None or torch.equal(a, b))


def _fails(comparison: dict, scalar: bool = False) -> bool:
    if comparison["norm_ratio"] == 0 and comparison["rel_l2"] == 0:
        return False  # zero on both sides
    if "judged_vs_token_sum" in comparison:
        return comparison["cosine"] <= 0 or comparison["judged_vs_token_sum"] > LAYOUT_TOKEN_SUM_BAR
    if scalar:
        # A one-element gradient (DeepSeek-V4's mHC alphas) is a near-cancelling sum over tokens, its net 0.2-1.3% of
        # its per-token terms (the cp1 verdict's cancellation ratios): reduction order moves the net by multiples
        # (measured 3.3x between PP2 and CP1), so only its sign is compared.
        return comparison["cosine"] <= 0
    return (
        comparison["rel_l2"] > LAYOUT_REL_L2_BAR
        or comparison["cosine"] < LAYOUT_COSINE_MIN
        or abs(comparison["norm_ratio"] - 1) > LAYOUT_NORM_RATIO_BAND
    )
