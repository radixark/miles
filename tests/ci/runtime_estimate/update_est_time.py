# doc-dev: docs/ci/04-runtime-est-time.md
from __future__ import annotations

import argparse
import ast
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

from tests.ci.ci_register import CIRegistry, HWBackend, collect_tests, discover_ci_files, ut_parse_one_file
from tests.ci.runtime_estimate.runtime_history import NeonRuntimeHistoryStore, RuntimeSample

WINDOW_DAYS = 21
MAX_SAMPLES = 15
MIN_SAMPLES = 3

RuntimeIdentity = tuple[str, str, str]


@dataclass(frozen=True)
class RuntimeEstimate:
    sample_count: int
    p90_seconds: float
    est_time: int
    run_attempts: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class EstimateChange:
    test_path: str
    suite: str
    old_est_time: float
    new_est_time: int
    sample_count: int
    p90_seconds: float
    run_attempts: tuple[tuple[int, int], ...]


def inclusive_percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= percentile <= 1:
        raise ValueError(f"percentile must be between 0 and 1, got {percentile}")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def bucket_estimate(seconds: float) -> int:
    bucket = 10 if seconds <= 200 else 100
    return max(10, math.ceil(seconds / bucket) * bucket)


def build_estimates(
    samples: Sequence[RuntimeSample],
    *,
    min_samples: int = MIN_SAMPLES,
    max_samples: int = MAX_SAMPLES,
) -> dict[RuntimeIdentity, RuntimeEstimate]:
    grouped: dict[RuntimeIdentity, list[RuntimeSample]] = defaultdict(list)
    for sample in samples:
        grouped[(sample.test_path, sample.backend, sample.suite)].append(sample)

    estimates = {}
    for identity, identity_samples in grouped.items():
        recent = sorted(
            identity_samples,
            key=lambda sample: (sample.recorded_at, sample.github_run_id, sample.github_run_attempt),
            reverse=True,
        )[:max_samples]
        if len(recent) < min_samples:
            continue
        p90_seconds = inclusive_percentile([sample.elapsed_seconds for sample in recent], 0.9)
        estimates[identity] = RuntimeEstimate(
            sample_count=len(recent),
            p90_seconds=p90_seconds,
            est_time=bucket_estimate(p90_seconds),
            run_attempts=tuple(dict.fromkeys((sample.github_run_id, sample.github_run_attempt) for sample in recent)),
        )
    return estimates


def _active_cuda_e2e_registrations() -> list[CIRegistry]:
    registrations = collect_tests(discover_ci_files(), sanity_check=True)
    active = [
        registry
        for registry in registrations
        if registry.backend == HWBackend.CUDA
        and registry.filename.startswith("tests/e2e/")
        and registry.disabled is None
    ]
    identities = [(registry.filename, registry.backend.name, registry.suite) for registry in active]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate active CUDA CI runtime identity")
    return active


def _registration_calls(tree: ast.Module) -> list[ast.Call]:
    names = {"register_cpu_ci", "register_cuda_ci", "register_rocm_ci"}
    return [
        statement.value
        for statement in tree.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id in names
    ]


def _est_time_node(call: ast.Call) -> ast.AST:
    if call.args:
        return call.args[0]
    for keyword in call.keywords:
        if keyword.arg == "est_time":
            return keyword.value
    raise ValueError("validated CI registration has no est_time node")


def render_file_updates(
    filename: str,
    estimates: Mapping[RuntimeIdentity, RuntimeEstimate],
) -> tuple[bytes, list[EstimateChange]]:
    source_bytes = Path(filename).read_bytes()
    source = source_bytes.decode("utf-8")
    tree = ast.parse(source, filename=filename)
    calls = _registration_calls(tree)
    registrations = ut_parse_one_file(filename)
    if len(calls) != len(registrations):
        raise ValueError(f"{filename}: registration parser and update targets disagree")

    target_identities = {identity for identity in estimates if identity[0] == filename}
    for identity in sorted(target_identities):
        matches = [
            registry
            for registry in registrations
            if registry.backend == HWBackend.CUDA
            and registry.disabled is None
            and (filename, registry.backend.name, registry.suite) == identity
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{filename}: expected exactly one active CUDA registration for {identity!r}; found {len(matches)}"
            )

    line_offsets = [0]
    for line in source.splitlines(keepends=True):
        line_offsets.append(line_offsets[-1] + len(line.encode("utf-8")))

    edits: list[tuple[int, int, bytes]] = []
    changes: list[EstimateChange] = []
    for call, registry in zip(calls, registrations, strict=True):
        if registry.backend != HWBackend.CUDA or registry.disabled is not None:
            continue
        identity = (filename, registry.backend.name, registry.suite)
        estimate = estimates.get(identity)
        if estimate is None or registry.est_time == estimate.est_time:
            continue
        node = _est_time_node(call)
        if node.lineno != node.end_lineno:
            raise ValueError(f"{filename}: est_time literal spans multiple lines")
        start = line_offsets[node.lineno - 1] + node.col_offset
        end = line_offsets[node.end_lineno - 1] + node.end_col_offset
        edits.append((start, end, str(estimate.est_time).encode("ascii")))
        changes.append(
            EstimateChange(
                test_path=filename,
                suite=registry.suite,
                old_est_time=registry.est_time,
                new_est_time=estimate.est_time,
                sample_count=estimate.sample_count,
                p90_seconds=estimate.p90_seconds,
                run_attempts=estimate.run_attempts,
            )
        )

    for start, end, replacement in reversed(edits):
        source_bytes = source_bytes[:start] + replacement + source_bytes[end:]
    return source_bytes, changes


def update_registered_estimates(
    estimates: Mapping[RuntimeIdentity, RuntimeEstimate],
    *,
    dry_run: bool,
) -> list[EstimateChange]:
    registrations = _active_cuda_e2e_registrations()
    eligible = {
        (registry.filename, registry.backend.name, registry.suite): estimates[
            (registry.filename, registry.backend.name, registry.suite)
        ]
        for registry in registrations
        if (registry.filename, registry.backend.name, registry.suite) in estimates
    }
    by_file: dict[str, dict[RuntimeIdentity, RuntimeEstimate]] = defaultdict(dict)
    for identity, estimate in eligible.items():
        by_file[identity[0]][identity] = estimate

    rendered = []
    changes = []
    for filename in sorted(by_file):
        new_source, file_changes = render_file_updates(filename, by_file[filename])
        rendered.append((filename, new_source))
        changes.extend(file_changes)

    if not dry_run:
        for filename, new_source in rendered:
            Path(filename).write_bytes(new_source)
    return changes


def render_report(changes: Sequence[EstimateChange], cutoff: datetime, upper: datetime) -> str:
    lines = [
        "# CI runtime estimate update",
        "",
        f"PASS samples from scheduled main runs in UTC window `[{cutoff.isoformat()}, {upper.isoformat()})`; p90 uses at most {MAX_SAMPLES} samples and requires {MIN_SAMPLES}.",
        "",
    ]
    if not changes:
        lines.append("No `est_time` changes were needed.")
        return "\n".join(lines) + "\n"

    repository = os.environ.get("GITHUB_REPOSITORY", "radixark/miles")
    lines.extend(
        [
            "| Test | Suite | Samples | p90 | Old | New | Run attempts |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for change in changes:
        runs = ", ".join(
            f"[{run_id}/{run_attempt}](https://github.com/{repository}/actions/runs/{run_id}/attempts/{run_attempt})"
            for run_id, run_attempt in change.run_attempts
        )
        lines.append(
            f"| `{change.test_path}` | `{change.suite}` | {change.sample_count} | {change.p90_seconds:.1f}s | "
            f"{change.old_est_time:g}s | {change.new_est_time}s | {runs} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Update CUDA e2e est_time literals from CI runtime history")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing test files")
    parser.add_argument("--report-file", type=Path, help="Write a Markdown audit report")
    parser.add_argument("--as-of", type=date.fromisoformat, help="UTC date used to anchor the history window")
    args = parser.parse_args()

    as_of = args.as_of or datetime.now(UTC).date()
    upper = datetime.combine(as_of, time.min, tzinfo=UTC)
    cutoff = upper - timedelta(days=WINDOW_DAYS)
    store = NeonRuntimeHistoryStore()
    samples = store.recent_successful_attempts(cutoff, upper, MAX_SAMPLES)
    estimates = build_estimates(samples)
    changes = update_registered_estimates(estimates, dry_run=args.dry_run)
    report = render_report(changes, cutoff, upper)
    print(report, end="")
    if args.report_file:
        args.report_file.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
