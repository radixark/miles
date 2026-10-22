"""Compare per-token logprobs between baseline and target runs."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_LOG_RATIO_EXP_CLAMP = 20.0
_LOW_VAR_KL_MAX = 10.0


@dataclass
class _PositionLogprob:
    global_position: int
    token_id: int
    logprob: float


@dataclass
class _CompareResult:
    passed: bool
    failure_reasons: tuple[str, ...]
    num_baseline_positions: int
    num_target_positions: int
    num_positions: int
    max_abs_diff: float
    max_diff_position: int
    max_diff_baseline_logprob: float
    max_diff_target_logprob: float
    max_diff_token_id: int
    mean_abs_diff: float
    median_abs_diff: float
    p95_abs_diff: float
    p99_abs_diff: float
    k3_kl: float
    first_position_mean_abs_diff: float | None
    remaining_position_mean_abs_diff: float | None
    baseline_mean_logprob: float
    target_mean_logprob: float
    threshold: float
    per_position_diffs: list[float] = field(repr=False)


def compare_logprobs(
    *,
    baseline_dir: Path,
    target_dir: Path,
    threshold: float = 1e-3,
) -> bool:
    """Compare logprob JSON files between baseline and target.

    Returns True only when both inputs contain the same finite token positions
    and the mean absolute difference is within ``threshold``.
    """
    try:
        baseline_entries = _load_and_merge(baseline_dir)
        target_entries = _load_and_merge(target_dir)
        result = _compute_comparison(
            baseline_entries=baseline_entries,
            target_entries=target_entries,
            threshold=threshold,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"[logprob-compare] FAILED to compare logprobs: {error}", flush=True)
        return False

    _print_report(result)
    return result.passed


def _load_and_merge(directory: Path) -> dict[tuple[int, int], _PositionLogprob]:
    """Load all rank JSON files and merge by (batch_index, global_position).

    TP ranks produce identical logprobs (runtime_gather_output=True), so we deduplicate.
    PP intermediate stages have no logits and produce no files.
    """
    merged: dict[tuple[int, int], _PositionLogprob] = {}

    json_files = sorted(directory.glob("rank_*.json"))
    if not json_files:
        print(f"[logprob-compare] WARNING: no rank_*.json files found in {directory}", flush=True)
        return merged

    for json_file in json_files:
        data: dict[str, Any] = json.loads(json_file.read_text())
        entries_by_batch: list[list[dict[str, Any]]] = data["logprob_entries"]

        for batch_idx, batch_entries in enumerate(entries_by_batch):
            for entry in batch_entries:
                if not entry["is_valid"]:
                    continue

                key = (batch_idx, entry["global_position"])
                position = _PositionLogprob(
                    global_position=entry["global_position"],
                    token_id=entry["token_id"],
                    logprob=entry["logprob"],
                )
                previous = merged.get(key)
                if previous is not None and previous != position:
                    raise ValueError(f"conflicting duplicate logprob entry for batch={batch_idx}, position={entry['global_position']} in {json_file}")
                merged[key] = position

    return merged


def _compute_comparison(
    *,
    baseline_entries: dict[tuple[int, int], _PositionLogprob],
    target_entries: dict[tuple[int, int], _PositionLogprob],
    threshold: float,
) -> _CompareResult:
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError(f"logprob threshold must be finite and non-negative, got {threshold}")

    baseline_keys = set(baseline_entries)
    target_keys = set(target_entries)
    common_keys = sorted(baseline_keys & target_keys)
    failure_reasons: list[str] = []

    if not baseline_keys:
        failure_reasons.append("baseline has no valid logprob positions")
    if not target_keys:
        failure_reasons.append("target has no valid logprob positions")

    missing_from_target = baseline_keys - target_keys
    missing_from_baseline = target_keys - baseline_keys
    if missing_from_target:
        failure_reasons.append(f"target is missing {len(missing_from_target)} baseline positions")
    if missing_from_baseline:
        failure_reasons.append(f"baseline is missing {len(missing_from_baseline)} target positions")

    mismatched_token_keys = [key for key in common_keys if baseline_entries[key].token_id != target_entries[key].token_id]
    if mismatched_token_keys:
        failure_reasons.append(f"token IDs differ at {len(mismatched_token_keys)} positions")

    nonfinite_keys = [key for key in common_keys if not (math.isfinite(baseline_entries[key].logprob) and math.isfinite(target_entries[key].logprob))]
    if nonfinite_keys:
        failure_reasons.append(f"non-finite logprobs found at {len(nonfinite_keys)} positions")

    invalid_keys = set(mismatched_token_keys) | set(nonfinite_keys)
    comparable_keys = [key for key in common_keys if key not in invalid_keys]
    if not comparable_keys:
        if common_keys:
            failure_reasons.append("no aligned finite logprob positions remain")
        elif baseline_keys or target_keys:
            failure_reasons.append("baseline and target have no common positions")
        return _empty_result(
            failure_reasons=failure_reasons,
            num_baseline_positions=len(baseline_keys),
            num_target_positions=len(target_keys),
            threshold=threshold,
        )

    diffs: list[float] = []
    max_abs_diff = 0.0
    max_diff_key: tuple[int, int] = comparable_keys[0]

    baseline_logprobs: list[float] = []
    target_logprobs: list[float] = []
    k3_values: list[float] = []

    for key in comparable_keys:
        baseline = baseline_entries[key]
        target = target_entries[key]
        abs_diff = abs(baseline.logprob - target.logprob)
        diffs.append(abs_diff)
        baseline_logprobs.append(baseline.logprob)
        target_logprobs.append(target.logprob)
        log_ratio = max(-_LOG_RATIO_EXP_CLAMP, min(_LOG_RATIO_EXP_CLAMP, target.logprob - baseline.logprob))
        k3_values.append(min(_LOW_VAR_KL_MAX, math.exp(log_ratio) - 1.0 - log_ratio))

        if abs_diff > max_abs_diff:
            max_abs_diff = abs_diff
            max_diff_key = key

    sorted_diffs = sorted(diffs)
    num = len(sorted_diffs)
    first_keys_by_batch: dict[int, tuple[int, int]] = {}
    for key in comparable_keys:
        first_keys_by_batch.setdefault(key[0], key)
    first_keys = set(first_keys_by_batch.values())
    first_diffs = [diff for key, diff in zip(comparable_keys, diffs, strict=True) if key in first_keys]
    remaining_diffs = [diff for key, diff in zip(comparable_keys, diffs, strict=True) if key not in first_keys]

    baseline_worst = baseline_entries[max_diff_key]
    target_worst = target_entries[max_diff_key]

    mean_abs_diff = statistics.mean(diffs)
    if mean_abs_diff > threshold:
        failure_reasons.append(f"mean absolute difference {mean_abs_diff:.6e} exceeds threshold {threshold:.6e}")

    return _CompareResult(
        passed=not failure_reasons,
        failure_reasons=tuple(failure_reasons),
        num_baseline_positions=len(baseline_keys),
        num_target_positions=len(target_keys),
        num_positions=num,
        max_abs_diff=max_abs_diff,
        max_diff_position=max_diff_key[1],
        max_diff_baseline_logprob=baseline_worst.logprob,
        max_diff_target_logprob=target_worst.logprob,
        max_diff_token_id=baseline_worst.token_id,
        mean_abs_diff=mean_abs_diff,
        median_abs_diff=statistics.median(diffs),
        p95_abs_diff=_nearest_rank_percentile(sorted_diffs, 0.95),
        p99_abs_diff=_nearest_rank_percentile(sorted_diffs, 0.99),
        k3_kl=statistics.mean(k3_values),
        first_position_mean_abs_diff=statistics.mean(first_diffs),
        remaining_position_mean_abs_diff=statistics.mean(remaining_diffs) if remaining_diffs else None,
        baseline_mean_logprob=statistics.mean(baseline_logprobs),
        target_mean_logprob=statistics.mean(target_logprobs),
        threshold=threshold,
        per_position_diffs=diffs,
    )


def _empty_result(
    *,
    failure_reasons: list[str],
    num_baseline_positions: int,
    num_target_positions: int,
    threshold: float,
) -> _CompareResult:
    return _CompareResult(
        passed=False,
        failure_reasons=tuple(failure_reasons),
        num_baseline_positions=num_baseline_positions,
        num_target_positions=num_target_positions,
        num_positions=0,
        max_abs_diff=0.0,
        max_diff_position=-1,
        max_diff_baseline_logprob=0.0,
        max_diff_target_logprob=0.0,
        max_diff_token_id=-1,
        mean_abs_diff=0.0,
        median_abs_diff=0.0,
        p95_abs_diff=0.0,
        p99_abs_diff=0.0,
        k3_kl=0.0,
        first_position_mean_abs_diff=None,
        remaining_position_mean_abs_diff=None,
        baseline_mean_logprob=0.0,
        target_mean_logprob=0.0,
        threshold=threshold,
        per_position_diffs=[],
    )


def _nearest_rank_percentile(sorted_values: list[float], quantile: float) -> float:
    index = max(0, math.ceil(quantile * len(sorted_values)) - 1)
    return sorted_values[index]


def _print_report(result: _CompareResult) -> None:
    status = "PASSED" if result.passed else "FAILED"
    print(f"\n{'=' * 70}", flush=True)
    print(f"Logprob Comparison: {status}", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  Baseline positions : {result.num_baseline_positions}", flush=True)
    print(f"  Target positions   : {result.num_target_positions}", flush=True)
    print(f"  Positions compared : {result.num_positions}", flush=True)
    for reason in result.failure_reasons:
        print(f"  Failure            : {reason}", flush=True)

    if result.num_positions == 0:
        print(f"{'=' * 70}\n", flush=True)
        return

    print(f"  Threshold (mean)   : {result.threshold}", flush=True)
    print(f"  Max abs diff       : {result.max_abs_diff:.6e}", flush=True)
    threshold_status = "<= threshold" if result.mean_abs_diff <= result.threshold else "> threshold"
    print(
        f"  Mean abs diff      : {result.mean_abs_diff:.6e}  {threshold_status}",
        flush=True,
    )
    print(f"  Median abs diff    : {result.median_abs_diff:.6e}", flush=True)
    print(f"  P95 abs diff       : {result.p95_abs_diff:.6e}", flush=True)
    print(f"  P99 abs diff       : {result.p99_abs_diff:.6e}", flush=True)
    print(f"  K3 KL (clamped)    : {result.k3_kl:.6e}", flush=True)
    print(f"  First-position mean: {result.first_position_mean_abs_diff:.6e}", flush=True)
    if result.remaining_position_mean_abs_diff is not None:
        print(f"  Remaining-pos mean : {result.remaining_position_mean_abs_diff:.6e}", flush=True)

    if result.num_positions > 0:
        print(f"\n  Worst position     : {result.max_diff_position}", flush=True)
        print(f"    token_id         : {result.max_diff_token_id}", flush=True)
        print(f"    baseline logprob : {result.max_diff_baseline_logprob:.6f}", flush=True)
        print(f"    target logprob   : {result.max_diff_target_logprob:.6f}", flush=True)

    print(f"\n  Baseline mean logprob : {result.baseline_mean_logprob:.6f}", flush=True)
    print(f"  Target mean logprob   : {result.target_mean_logprob:.6f}", flush=True)
    print(f"{'=' * 70}\n", flush=True)
