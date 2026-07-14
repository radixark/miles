import subprocess
import sys
from pathlib import Path


def run_comparator(
    *,
    baseline_path: Path,
    target_path: Path,
    diff_thresholds: list[tuple[str, str]],
    allow_skipped_pattern: str,
    allow_failed_pattern: str,
    grouping_skip_keys: list[str] | None = None,
    extra_args: list[str] | None,
) -> subprocess.CompletedProcess[str]:
    # Skip 'rank' when grouping bundles: under FT (target) and non-FT (baseline) the same
    # logical (pp_rank, cp_rank, ep_rank, tp_rank) coordinate maps to a different absolute
    # rank ID (e.g. baseline rank=4 vs target cell0 rank=2 for PP=1, CP=0). Without skipping
    # 'rank' the comparator gets `baseline_load_failed` for every tensor and fails with rc=1.
    # Callers may pass extra keys (e.g. no_failure skips 'dp'/'edp' too). (Grouping is a
    # comparator-matching detail, not a pass/fail threshold.)
    skip_keys: list[str] = list(grouping_skip_keys) if grouping_skip_keys is not None else ["rank"]
    assert "rank" in skip_keys, f"grouping_skip_keys must include 'rank', got {skip_keys}"

    cmd: list[str] = [
        sys.executable,
        "-m",
        "sglang.srt.debug_utils.comparator",
        "--baseline-path",
        str(baseline_path),
        "--target-path",
        str(target_path),
        "--output-format",
        "json",
        "--grouping-skip-keys",
        *skip_keys,
        "--allow-skipped-pattern",
        allow_skipped_pattern,
        "--allow-failed-pattern",
        allow_failed_pattern,
    ]
    if extra_args:
        cmd.extend(extra_args)
    # Keep --diff-threshold strictly last: its nargs="*" greedily consumes every
    # following token, so no flag with a bare value may come after it.
    cmd.append("--diff-threshold")
    for pattern, predicate in diff_thresholds:
        cmd.extend([pattern, predicate])

    result: subprocess.CompletedProcess[str] = subprocess.run(
        cmd,
        text=True,
    )

    return result
