"""Replay all persisted candidates against earlier trusted history without writing to the store.

Export SQL with --export-sql, execute it through Neon access, then pass the
complete decrypted response to --input. Run from the repository root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from itertools import groupby
from pathlib import Path

from tests.ci.ci_register import HWBackend, discover_ci_files, ut_parse_one_file
from tests.ci.metric_history.gate import HISTORY_LIMIT, MIN_ENFORCED_BASELINES, GateStatus, evaluate_coordinate
from tests.ci.metric_history.register import CiGateSpec, parse_ci_gate_specs

_COLUMNS = [
    "run_id",
    "test_path",
    "backend",
    "suite",
    "created_at",
    "github_run_id",
    "github_run_attempt",
    "trusted",
    "values",
]
# Consecutive latest failures on one side of the baseline that mark a
# coordinate's trusted history as stale rather than noisy.
STALE_STREAK = 3


def enforced_specs() -> dict[tuple[str, str], CiGateSpec]:
    """Discover all enabled declarations without importing GPU test modules."""
    specs = {}
    for filename in sorted(discover_ci_files()):
        if not any(r.backend == HWBackend.CUDA and not r.disabled for r in ut_parse_one_file(filename)):
            continue
        for spec in parse_ci_gate_specs(filename):
            if spec.enforce:
                specs[(filename, spec.metric_key)] = spec
    if not specs:
        raise ValueError("no enabled CUDA enforcement declarations found")
    return specs


def export_sql(specs: dict[tuple[str, str], CiGateSpec]) -> str:
    literals = ",\n".join("(" + ",".join("'" + v.replace("'", "''") + "'" for v in key) + ")" for key in sorted(specs))
    return f"""BEGIN TRANSACTION READ ONLY;
SET LOCAL statement_timeout = '120s';
WITH specs(test_path, metric_key) AS (VALUES
{literals}
)
SELECT r.run_id, r.test_path, r.backend, r.suite, r.created_at::text,
       r.github_run_id, r.github_run_attempt, r.trusted,
       COALESCE(jsonb_agg(jsonb_build_array(mv.metric_key, mv.step, mv.value) ORDER BY mv.metric_key, mv.step)
                 FILTER (WHERE mv.run_id IS NOT NULL), '[]'::jsonb) AS values
FROM runs r LEFT JOIN metric_values mv ON mv.run_id = r.run_id
    AND EXISTS (SELECT 1 FROM specs s WHERE r.test_path = s.test_path AND mv.metric_key = s.metric_key)
WHERE r.backend = 'cuda' AND r.test_path IN (SELECT test_path FROM specs)
GROUP BY r.run_id, r.test_path, r.backend, r.suite, r.created_at, r.github_run_id, r.github_run_attempt, r.trusted
ORDER BY r.created_at, r.run_id;
COMMIT;
"""


def load_export(path: Path, sql: str) -> list[dict]:
    payload = json.loads(path.read_text())
    if payload["status"] != "ok" or payload["truncated"] is not False or payload["error"] is not None:
        raise ValueError("Neon export failed or was truncated; a partial replay is not valid")
    if payload["sql_sha256"] != hashlib.sha256(sql.encode()).hexdigest():
        raise ValueError("export SQL does not match the current enforcement declarations; export again")
    tables = [result for result in payload["results"] if result["columns"]]
    if len(tables) != 1 or tables[0]["columns"] != _COLUMNS:
        raise ValueError("unexpected Neon export columns")
    return [dict(zip(_COLUMNS, row, strict=True)) for row in tables[0]["rows"]]


def replay(rows: list[dict], specs: dict[tuple[str, str], CiGateSpec]) -> dict:
    coordinates = defaultdict(list)
    seen_runs = set()
    suites_by_test = defaultdict(set)
    run_coords = []
    for row in rows:
        if row["run_id"] in seen_runs or row["backend"] != "cuda":
            raise ValueError("duplicate run or unexpected backend in export")
        if not isinstance(row["trusted"], bool):
            raise ValueError("trusted must be a boolean")
        seen_runs.add(row["run_id"])
        suites_by_test[row["test_path"]].add(row["suite"])
        timestamp = datetime.fromisoformat(row["created_at"])
        if timestamp.tzinfo is None:
            raise ValueError("history timestamps must include a timezone")
        seen_coords = set()
        for metric, step, value in row["values"]:
            declaration = (row["test_path"], metric)
            if declaration not in specs or not isinstance(step, int) or isinstance(step, bool):
                raise ValueError("unexpected coordinate in export")
            spec = specs[declaration]
            # Values an earlier steps selection stored at other points are not this declaration's history.
            if (step == -1) != (spec.steps == "last") or (isinstance(spec.steps, list) and step not in spec.steps):
                continue
            key = (*declaration, row["suite"], step)
            if (
                key in seen_coords
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("duplicate coordinate or invalid golden value in export")
            seen_coords.add(key)
            coordinates[key].append((timestamp, row, value))
        run_coords.append((timestamp, row, seen_coords))

    # A run older than a coordinate's first stored value predates its declaration, so it is not missing it.
    first_seen = {}
    for timestamp, _, seen_coords in run_coords:
        for key in seen_coords:
            for point in (key, (*key[:3], None)):
                first_seen[point] = min(first_seen.get(point, timestamp), timestamp)
    runs_with_missing_values = []
    for timestamp, row, seen_coords in run_coords:
        missing_values = []
        for declaration, spec in specs.items():
            if declaration[0] != row["test_path"]:
                continue
            steps = [-1] if spec.steps == "last" else spec.steps if isinstance(spec.steps, list) else [None]
            for step in steps:
                point = (*declaration, row["suite"], step)
                if point not in first_seen or timestamp < first_seen[point]:
                    continue
                present = any(key[:3] == point[:3] for key in seen_coords) if step is None else point in seen_coords
                if not present:
                    missing_values.append(
                        dict(
                            metric_key=spec.metric_key,
                            steps_key=spec.steps_key,
                            constraint_key=spec.constraint_key,
                            step=step,
                        )
                    )
        if missing_values:
            runs_with_missing_values.append(
                {**{k: v for k, v in row.items() if k != "values"}, "missing_values": missing_values}
            )

    for declaration, spec in specs.items():
        steps = [-1] if spec.steps == "last" else spec.steps if isinstance(spec.steps, list) else []
        for suite in suites_by_test[declaration[0]]:
            for step in steps:
                coordinates.setdefault((*declaration, suite, step), [])

    failures, coverage, stale, zero_flips = [], [], [], []
    eligible_declarations = set()
    totals = dict(
        values=0,
        checked=0,
        passed=0,
        failed=0,
        insufficient_history=0,
        enforcement_checked=0,
        enforcement_failed=0,
        trusted_checked=0,
        trusted_failed=0,
        untrusted_checked=0,
        untrusted_failed=0,
        trusted_enforcement_failed=0,
        untrusted_enforcement_failed=0,
    )
    for key, points in sorted(coordinates.items()):
        declaration, suite, step = key[:2], key[2], key[3]
        spec = specs[declaration]
        coordinate = dict(
            test_path=key[0],
            suite=suite,
            metric_key=key[1],
            steps_key=spec.steps_key,
            constraint_key=spec.constraint_key,
            step=step,
        )
        stats = {name: 0 for name in totals}
        history, outcomes = [], []
        # Timestamp peers must all be evaluated before any enters history.
        for _, peers in groupby(sorted(points, key=lambda p: (p[0], p[1]["run_id"])), key=lambda p: p[0]):
            peers = list(peers)
            if len(history) > HISTORY_LIMIT and history[-HISTORY_LIMIT - 1][0] == history[-HISTORY_LIMIT][0]:
                raise ValueError("ambiguous timestamp tie at history window boundary")
            baseline = [value for _, _, value in history[-HISTORY_LIMIT:]][::-1]
            for _, row, value in peers:
                result = evaluate_coordinate(spec, current=value, step=step, at_step=None, trusted_values=baseline)
                eligible = result.baseline_n >= MIN_ENFORCED_BASELINES
                stats["values"] += 1
                stats["insufficient_history"] += int(not eligible)
                stats["enforcement_checked"] += int(eligible)
                if eligible:
                    eligible_declarations.add((*declaration, suite))
                if result.historical_status == GateStatus.INACTIVE:
                    continue
                stats["checked"] += 1
                trust = "trusted" if row["trusted"] else "untrusted"
                stats[f"{trust}_checked"] += 1
                failed = result.historical_status != GateStatus.PASS
                outcomes.append((row, value, failed, result.baseline_mean))
                stats["failed" if failed else "passed"] += 1
                stats["enforcement_failed"] += int(failed and eligible)
                stats[f"{trust}_failed"] += int(failed)
                stats[f"{trust}_enforcement_failed"] += int(failed and eligible)
                if failed:
                    failures.append(
                        dict(
                            **coordinate,
                            run_id=row["run_id"],
                            created_at=row["created_at"],
                            github_run_id=row["github_run_id"],
                            github_run_attempt=row["github_run_attempt"],
                            trusted=row["trusted"],
                            current=value,
                            baseline_n=result.baseline_n,
                            baseline_mean=result.baseline_mean,
                            enforcement_eligible=eligible,
                            reason=result.reason,
                            baseline_run_ids=[run_id for _, run_id, _ in history[-HISTORY_LIMIT:]][::-1],
                        )
                    )
            history.extend((timestamp, row["run_id"], value) for timestamp, row, value in peers if row["trusted"])
        coverage.append(dict(**coordinate, **stats))
        values = [value for _, _, value in points]
        streak = outcomes[-STALE_STREAK:]
        # The lists are independent; a coordinate in both needs its values read before choosing a fix.
        if (
            len(streak) == STALE_STREAK
            and all(f for _, _, f, _ in streak)
            and len({v > mean for _, v, _, mean in streak}) == 1
        ):
            stale.append(
                dict(
                    **coordinate,
                    baseline_mean=streak[-1][3],
                    recent=[dict(run_id=r["run_id"], created_at=r["created_at"], current=v) for r, v, _, _ in streak],
                    reset_run_ids=[run_id for _, run_id, _ in history],
                )
            )
        if stats["failed"] and 0 in values and any(values):
            # Exact zeros beside nonzero values may leave no band to tune, so
            # enforce=False may fit better than a history reset.
            zero_flips.append(
                dict(**coordinate, zero_values=values.count(0), nonzero_values=len(values) - values.count(0))
            )
        for name, count in stats.items():
            totals[name] += count
    missing = [
        dict(
            test_path=key[0],
            suite=suite,
            metric_key=key[1],
            steps_key=specs[key].steps_key,
            constraint_key=specs[key].constraint_key,
        )
        for key in sorted(specs)
        for suite in sorted(suites_by_test[key[0]]) or [None]
        if (*key, suite) not in eligible_declarations
    ]
    incomplete_coordinates = sum(c["enforcement_checked"] == 0 for c in coverage)
    status = (
        "fail"
        if failures
        else "incomplete" if missing or incomplete_coordinates or runs_with_missing_values else "pass"
    )
    return dict(
        status=status,
        tests=len({key[0] for key in specs}),
        declarations=len(specs),
        runs=len(rows),
        history_limit=HISTORY_LIMIT,
        min_enforced_baselines=MIN_ENFORCED_BASELINES,
        **totals,
        declarations_without_enforcement_checks=missing,
        coordinates_without_enforcement_checks=incomplete_coordinates,
        runs_with_missing_values=runs_with_missing_values,
        coverage=coverage,
        failures=failures,
        stale_coordinates=stale,
        zero_flip_coordinates=zero_flips,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--export-sql", action="store_true")
    mode.add_argument("--input", type=Path, help="complete decrypted Neon access result.json")
    args = parser.parse_args()
    specs = enforced_specs()
    sql = export_sql(specs)
    if args.export_sql:
        print(sql, end="")
        return 0
    report = replay(load_export(args.input, sql), specs)
    print(json.dumps(report, indent=2, allow_nan=False))
    return {"pass": 0, "fail": 1, "incomplete": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
