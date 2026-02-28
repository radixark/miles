# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.
# The comparator must report all-passed with zero failures — no exceptions.

import os
import subprocess
import sys
from pathlib import Path

import torch
from sglang.srt.debug_utils.comparator.output_types import ComparisonRecord, SummaryRecord, parse_record_json


def clear_proxy_env() -> None:
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)


def check_dump_dir(
    phase_dir: Path,
    exp_pattern: str,
    expected_fields: list[str] | None = None,
) -> None:
    """Verify dump directory structure: .pt files exist, contain value + meta keys, contain expected fields."""
    assert phase_dir.exists(), f"Missing dump dir: {phase_dir}"
    dump_subdirs: list[Path] = list(phase_dir.glob(exp_pattern))
    assert len(dump_subdirs) > 0, f"No {exp_pattern} subdirs in {phase_dir}"
    dump_files: list[Path] = list(dump_subdirs[0].glob("*.pt"))
    assert len(dump_files) > 0, f"No .pt files in {dump_subdirs[0]}"
    sample: dict = torch.load(dump_files[0], weights_only=False)
    assert isinstance(sample, dict), f"Unexpected type: {type(sample)}"
    assert "value" in sample and "meta" in sample, f"Missing keys: {sample.keys()}"

    if expected_fields:
        for field in expected_fields:
            matches: list[Path] = list(phase_dir.rglob(f"*name={field}*.pt"))
            assert len(matches) > 0, f"Expected field '{field}' not found under {phase_dir}"


def log_comparator_output(stdout: str, stderr: str) -> None:
    if stdout.strip():
        print(f"[comparator stdout]\n{stdout}")
    if stderr.strip():
        print(f"[comparator stderr]\n{stderr}")


def run_and_verify_comparator(
    baseline_dir: Path,
    target_dir: Path,
    extra_args: list[str] | None = None,
) -> None:
    """Run comparator subprocess, parse JSONL, assert all passed + 0 failed + 0 skipped."""
    cmd: list[str] = [
        sys.executable,
        "-m",
        "sglang.srt.debug_utils.comparator",
        "--baseline-path",
        str(baseline_dir),
        "--target-path",
        str(target_dir),
        "--output-format",
        "json",
        "--grouping",
        "logical",
    ]
    if extra_args:
        cmd.extend(extra_args)

    result: subprocess.CompletedProcess[str] = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    log_comparator_output(stdout=result.stdout, stderr=result.stderr)

    assert result.returncode == 0, f"Comparator failed (rc={result.returncode})\nstderr: {result.stderr[-2000:]}"

    records = [parse_record_json(line) for line in result.stdout.strip().splitlines() if line.strip()]
    assert len(records) > 0

    comparisons: list[ComparisonRecord] = [r for r in records if isinstance(r, ComparisonRecord)]
    assert len(comparisons) > 0, "No comparison records produced"

    diff_passed: int = 0
    diff_failed: list[str] = []
    for comp in comparisons:
        if comp.diff is not None and comp.diff.passed:
            diff_passed += 1
        else:
            rel_diff: float = comp.diff.rel_diff if comp.diff is not None else float("nan")
            diff_failed.append(f"{comp.name} (rel_diff={rel_diff:.6f})")

    assert (
        len(diff_failed) == 0
    ), f"Comparator found {len(diff_failed)} diff failures out of {len(comparisons)} comparisons: " + ", ".join(
        diff_failed[:10]
    )
    assert diff_passed > 0, f"No comparisons passed (total={len(comparisons)})"

    summaries: list[SummaryRecord] = [r for r in records if isinstance(r, SummaryRecord)]
    assert len(summaries) == 1, f"Expected exactly 1 summary record, got {len(summaries)}"
    summary: SummaryRecord = summaries[0]
    assert summary.passed > 0, f"Summary passed must be > 0, got {summary.passed}"
    assert summary.failed == 0, f"Summary failed must be 0, got {summary.failed}"
    assert summary.skipped == 0, f"Summary skipped must be 0, got {summary.skipped}"

    print(
        f"Comparator verification passed: "
        f"total={len(comparisons)}, diff_passed={diff_passed}, diff_failed={len(diff_failed)}, "
        f"summary: passed={summary.passed}, failed={summary.failed}, skipped={summary.skipped}"
    )
