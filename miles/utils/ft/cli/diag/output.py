from __future__ import annotations

import json
import sys

from miles.utils.ft.models.diagnostics import DiagnosticResult


def print_results(
    results: list[DiagnosticResult],
    *,
    json_output: bool = False,
    node_id: str = "local",
) -> None:
    if json_output:
        print(json.dumps([r.model_dump() for r in results], indent=2, default=str))
        return

    print(f"\n=== Node Diagnostic: {node_id} ===\n")

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  {r.diagnostic_type:<20s} {status:<6s} {r.details}")

    fail_count = sum(1 for r in results if not r.passed)
    pass_count = len(results) - fail_count
    print(f"\nResult: {fail_count} FAIL, {pass_count} PASS\n")


def exit_with_results(results: list[DiagnosticResult]) -> None:
    if any(not r.passed for r in results):
        raise SystemExit(1)
