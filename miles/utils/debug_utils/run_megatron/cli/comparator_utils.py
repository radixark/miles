"""Comparator output parsing and validation helpers."""

import json


def print_json_summary(stdout: str) -> None:
    """Print a human-readable summary from JSON comparator output."""
    lines: list[str] = [line for line in stdout.strip().splitlines() if line.strip()]
    for line in lines:
        try:
            record: dict[str, object] = json.loads(line)
            if record.get("type") == "summary":
                print(
                    f"[summary] passed={record.get('passed')}, "
                    f"failed={record.get('failed')}, "
                    f"skipped={record.get('skipped')}",
                    flush=True,
                )
        except json.JSONDecodeError:
            pass


def assert_all_passed(stdout: str) -> None:
    """Assert all comparisons passed in JSON output (strict mode)."""
    from sglang.srt.debug_utils.comparator.output_types import ComparisonRecord, SummaryRecord, parse_record_json

    lines: list[str] = [line for line in stdout.strip().splitlines() if line.strip()]
    records = [parse_record_json(line) for line in lines]

    comparisons: list[ComparisonRecord] = [r for r in records if isinstance(r, ComparisonRecord)]
    assert len(comparisons) > 0, "No comparison records produced"

    failed: list[str] = []
    for comp in comparisons:
        if comp.diff is None or not comp.diff.passed:
            rel_diff: float = comp.diff.rel_diff if comp.diff is not None else float("nan")
            failed.append(f"{comp.name} (rel_diff={rel_diff:.6f})")

    assert len(failed) == 0, f"Comparator found {len(failed)} failures out of {len(comparisons)}: " + ", ".join(
        failed[:10]
    )

    summaries: list[SummaryRecord] = [r for r in records if isinstance(r, SummaryRecord)]
    assert len(summaries) == 1, f"Expected 1 summary, got {len(summaries)}"
    summary: SummaryRecord = summaries[0]
    assert summary.passed > 0, f"Summary passed must be > 0, got {summary.passed}"
    assert summary.failed == 0, f"Summary failed must be 0, got {summary.failed}"
    assert summary.skipped == 0, f"Summary skipped must be 0, got {summary.skipped}"

    print(
        f"[cli] All passed: total={len(comparisons)}, "
        f"summary: passed={summary.passed}, failed={summary.failed}, skipped={summary.skipped}",
        flush=True,
    )
