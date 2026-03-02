"""Compare per-token logprobs between baseline and target runs."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class _PositionLogprob:
    global_position: int
    token_id: int
    logprob: float


@dataclass
class _CompareResult:
    passed: bool
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

    Returns True if the comparison passes (max abs diff <= threshold).
    """
    baseline_entries = _load_and_merge(baseline_dir)
    target_entries = _load_and_merge(target_dir)

    if not baseline_entries:
        print("[logprob-compare] WARNING: no valid baseline logprob entries found", flush=True)
        return True
    if not target_entries:
        print("[logprob-compare] WARNING: no valid target logprob entries found", flush=True)
        return True

    result = _compute_comparison(
        baseline_entries=baseline_entries,
        target_entries=target_entries,
        threshold=threshold,
    )

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
        entries_by_batch: list[list[dict[str, Any]]] = data["entries"]

        for batch_idx, batch_entries in enumerate(entries_by_batch):
            for entry in batch_entries:
                if not entry["is_valid"]:
                    continue

                key = (batch_idx, entry["global_position"])
                if key not in merged:
                    merged[key] = _PositionLogprob(
                        global_position=entry["global_position"],
                        token_id=entry["token_id"],
                        logprob=entry["logprob"],
                    )

    return merged


def _compute_comparison(
    *,
    baseline_entries: dict[tuple[int, int], _PositionLogprob],
    target_entries: dict[tuple[int, int], _PositionLogprob],
    threshold: float,
) -> _CompareResult:
    common_keys = sorted(set(baseline_entries.keys()) & set(target_entries.keys()))

    if not common_keys:
        return _CompareResult(
            passed=True,
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
            baseline_mean_logprob=0.0,
            target_mean_logprob=0.0,
            threshold=threshold,
            per_position_diffs=[],
        )

    diffs: list[float] = []
    max_abs_diff = 0.0
    max_diff_key: tuple[int, int] = common_keys[0]

    baseline_logprobs: list[float] = []
    target_logprobs: list[float] = []

    for key in common_keys:
        baseline = baseline_entries[key]
        target = target_entries[key]
        abs_diff = abs(baseline.logprob - target.logprob)
        diffs.append(abs_diff)
        baseline_logprobs.append(baseline.logprob)
        target_logprobs.append(target.logprob)

        if abs_diff > max_abs_diff:
            max_abs_diff = abs_diff
            max_diff_key = key

    sorted_diffs = sorted(diffs)
    num = len(sorted_diffs)

    baseline_worst = baseline_entries[max_diff_key]
    target_worst = target_entries[max_diff_key]

    return _CompareResult(
        passed=max_abs_diff <= threshold,
        num_positions=num,
        max_abs_diff=max_abs_diff,
        max_diff_position=max_diff_key[1],
        max_diff_baseline_logprob=baseline_worst.logprob,
        max_diff_target_logprob=target_worst.logprob,
        max_diff_token_id=baseline_worst.token_id,
        mean_abs_diff=statistics.mean(diffs),
        median_abs_diff=statistics.median(diffs),
        p95_abs_diff=sorted_diffs[int(num * 0.95)] if num > 0 else 0.0,
        p99_abs_diff=sorted_diffs[int(num * 0.99)] if num > 0 else 0.0,
        baseline_mean_logprob=statistics.mean(baseline_logprobs),
        target_mean_logprob=statistics.mean(target_logprobs),
        threshold=threshold,
        per_position_diffs=diffs,
    )


def _print_report(result: _CompareResult) -> None:
    status = "PASSED" if result.passed else "FAILED"
    print(f"\n{'=' * 70}", flush=True)
    print(f"Logprob Comparison: {status}", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  Positions compared : {result.num_positions}", flush=True)
    print(f"  Threshold          : {result.threshold}", flush=True)
    print(f"  Max abs diff       : {result.max_abs_diff:.6e}", flush=True)
    print(f"  Mean abs diff      : {result.mean_abs_diff:.6e}", flush=True)
    print(f"  Median abs diff    : {result.median_abs_diff:.6e}", flush=True)
    print(f"  P95 abs diff       : {result.p95_abs_diff:.6e}", flush=True)
    print(f"  P99 abs diff       : {result.p99_abs_diff:.6e}", flush=True)

    if result.num_positions > 0:
        print(f"\n  Worst position     : {result.max_diff_position}", flush=True)
        print(f"    token_id         : {result.max_diff_token_id}", flush=True)
        print(f"    baseline logprob : {result.max_diff_baseline_logprob:.6f}", flush=True)
        print(f"    target logprob   : {result.max_diff_target_logprob:.6f}", flush=True)

    print(f"\n  Baseline mean logprob : {result.baseline_mean_logprob:.6f}", flush=True)
    print(f"  Target mean logprob   : {result.target_mean_logprob:.6f}", flush=True)
    print(f"{'=' * 70}\n", flush=True)
