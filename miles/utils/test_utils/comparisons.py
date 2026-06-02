import logging
import math
import subprocess
import sys
from pathlib import Path

from miles.utils.event_logger.logger import read_events
from miles.utils.event_logger.models import MetricEvent

logger = logging.getLogger(__name__)

_REQUIRED_METRIC_KEYS: list[str] = ["train/grad_norm", "train/loss"]


def compare_dumps(
    baseline_dir: str,
    target_dir: str,
    *,
    diff_threshold: float = 0.0085,
    abs_diff_threshold: float = 0.0,
    allow_skipped_pattern: str = "input_ids|positions|cu_seqlens_q|cu_seqlens_kv|qkv_format|.*witness.*",
    allow_failed_pattern: str = "input_ids|positions|cu_seqlens_q|cu_seqlens_kv|qkv_format",
    extra_args: list[str] | None = None,
) -> None:
    """Run the sglang dump comparator and assert no meaningful tensor diverged.

    The comparator's pass criterion is purely relative (cosine ``rel_diff <=
    diff_threshold``). That metric is degenerate for near-zero tensors: a tiny
    absolute difference on a near-zero value yields a huge relative diff. At test
    scale (few tokens, 128 experts) many low-traffic MoE experts have near-zero
    gradients, so a fault+recovery run whose rebuilt collective reduces in a
    different order produces large *relative* diffs on those experts despite an
    absolute difference that is negligible vs the model gradient scale.

    ``abs_diff_threshold`` adds an absolute floor (``torch.allclose`` semantics):
    a tensor is accepted if it passes the relative check OR its max absolute diff
    is within the floor. Normal-magnitude tensors are unaffected (their abs diff
    dwarfs the floor), so the strict relative check still governs them. The floor
    defaults to 0.0 (no effect); only fault-tolerance comparisons opt in.
    """
    baseline_path = Path(baseline_dir) / "dumps"
    target_path = Path(target_dir) / "dumps"

    assert baseline_path.exists(), f"Baseline dump dir does not exist: {baseline_path}"
    assert target_path.exists(), f"Target dump dir does not exist: {target_path}"

    result = _run_comparator(
        baseline_path=baseline_path,
        target_path=target_path,
        diff_threshold=diff_threshold,
        allow_skipped_pattern=allow_skipped_pattern,
        allow_failed_pattern=allow_failed_pattern,
        extra_args=extra_args,
    )

    if result.returncode == 0:
        print(f"Dump comparison passed: {baseline_path} vs {target_path}")
        return

    report = _find_comparator_report(target_path)
    assert report is not None, (
        f"Dump comparator failed (rc={result.returncode}) and no comparator_report.jsonl "
        f"found under {target_path}"
    )

    intolerable = _intolerable_dump_failures(
        report=report,
        abs_diff_threshold=abs_diff_threshold,
        allow_skipped_pattern=allow_skipped_pattern,
        allow_failed_pattern=allow_failed_pattern,
    )
    assert not intolerable, (
        f"Dump comparator failed (rc={result.returncode}): "
        f"{len(intolerable)} tensor(s) exceed both rel_diff>{diff_threshold} and "
        f"abs_diff>{abs_diff_threshold} (near-zero floor). First offenders: "
        + ", ".join(
            f"{t['name']}(rel={t['rel_diff']}, abs={t['max_abs_diff']})" for t in intolerable[:10]
        )
        + f". Full report: {report}"
    )
    print(
        f"Dump comparison passed: {baseline_path} vs {target_path} "
        f"(comparator rc={result.returncode}; all relative-diff failures within "
        f"near-zero abs floor {abs_diff_threshold})"
    )


def _find_comparator_report(target_path: Path) -> Path | None:
    candidates = sorted(target_path.rglob("comparator_report.jsonl"))
    return candidates[0] if candidates else None


def _intolerable_dump_failures(
    *,
    report: Path,
    abs_diff_threshold: float,
    allow_skipped_pattern: str,
    allow_failed_pattern: str,
) -> list[dict]:
    """Re-derive the comparator's failure verdict, tolerating near-zero abs diffs.

    Returns the list of failures that are NOT tolerable. A non-empty list means
    the comparison must fail. Mirrors the comparator's allow patterns so the only
    behavior change vs the raw rc is the absolute-floor tolerance for failures.
    Anything the floor cannot explain (errored tensors, disallowed skips, or an
    rc!=0 with no recognizable failure record) is reported as intolerable so the
    floor never silently hides a non-near-zero problem.
    """
    import json
    import math
    import re

    allow_failed_re = re.compile(allow_failed_pattern)
    allow_skipped_re = re.compile(allow_skipped_pattern)

    intolerable: list[dict] = []
    saw_failure_record = False
    errored = 0

    for line in report.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue

        rtype = rec.get("type")
        if rtype == "summary":
            errored = rec.get("errored", 0) or 0
            continue
        if rtype == "comparison_skip":
            name = rec.get("name", "")
            if not allow_skipped_re.search(name):
                intolerable.append({"name": f"<disallowed skip: {name}>", "rel_diff": None, "max_abs_diff": None})
            continue
        if rtype != "comparison_tensor":
            continue

        diff = rec.get("diff") or {}
        if diff.get("passed") is not False:
            continue
        name = rec.get("name", "")
        saw_failure_record = True
        if allow_failed_re.search(name):
            continue
        max_abs = diff.get("max_abs_diff")
        if max_abs is None or math.isnan(max_abs) or max_abs > abs_diff_threshold:
            intolerable.append(
                {"name": name, "rel_diff": diff.get("rel_diff"), "max_abs_diff": max_abs}
            )

    if errored:
        intolerable.append({"name": f"<{errored} errored tensors>", "rel_diff": None, "max_abs_diff": None})
    if not saw_failure_record and not intolerable:
        # rc!=0 but nothing recognizable to tolerate — preserve strictness.
        intolerable.append({"name": "<comparator rc!=0 with no tolerable failure record>", "rel_diff": None, "max_abs_diff": None})
    return intolerable


def compare_metrics(
    baseline_dir: str,
    target_dir: str,
    *,
    rtol: float,
    atol: float,
    key_prefixes: list[str] | None = None,
    exclude_keys: list[str] | None = None,
) -> None:
    if key_prefixes is None:
        key_prefixes = ["train/"]

    baseline_events = _read_metric_events(Path(baseline_dir))
    target_events = _read_metric_events(Path(target_dir))

    # FT retries (healing path) leave events from earlier failed attempts. Only
    # the highest-attempt events per rollout_id reflect the successful run.
    baseline_events = _keep_only_final_attempt(baseline_events)
    target_events = _keep_only_final_attempt(target_events)

    issues: list[str] = []
    issues += _check_event_counts(baseline_events, target_events, baseline_dir, target_dir)

    if not issues:
        for step_idx, (b_event, t_event) in enumerate(zip(baseline_events, target_events, strict=True)):
            _print_step_comparison_table(step_idx, b_event, t_event, key_prefixes, exclude_keys=exclude_keys)
            issues += _check_step_metrics(step_idx, b_event, t_event, key_prefixes, rtol, atol=atol, exclude_keys=exclude_keys)

    issues += _check_required_keys_exist(baseline_events)

    assert not issues, f"MetricEvent comparison found {len(issues)} issue(s):\n" + "\n".join(
        f"  - {i}" for i in issues
    )
    print(f"MetricEvent comparison passed: {len(baseline_events)} steps compared")


def _keep_only_final_attempt(events: list[MetricEvent]) -> list[MetricEvent]:
    """Keep only events from the highest-attempt for each rollout_id.

    During FT healing, a crashed rollout is retried at attempt+1; events from
    the failed attempt are partial and should be discarded for comparison.

    Rollout-side metrics (e.g. RolloutManager log_rollout_metrics) have
    attempt=None — they are not part of the FT retry stream, so we treat them
    as a single attempt (normalized to 0).
    """
    def _attempt(e: MetricEvent) -> int:
        return e.attempt if e.attempt is not None else 0

    max_attempt_by_rollout: dict[int, int] = {
        rollout_id: max(_attempt(e) for e in events if e.rollout_id == rollout_id)
        for rollout_id in {e.rollout_id for e in events}
    }
    return [e for e in events if _attempt(e) == max_attempt_by_rollout[e.rollout_id]]


def _check_event_counts(
    baseline: list[MetricEvent],
    target: list[MetricEvent],
    baseline_dir: str,
    target_dir: str,
) -> list[str]:
    issues: list[str] = []
    if len(baseline) == 0:
        issues.append(f"No MetricEvents found in baseline dir: {baseline_dir}")
    if len(target) == 0:
        issues.append(f"No MetricEvents found in target dir: {target_dir}")
    if len(baseline) > 0 and len(target) > 0 and len(baseline) != len(target):
        issues.append(f"MetricEvent count mismatch: baseline={len(baseline)}, target={len(target)}")
    return issues


def _check_step_metrics(
    step_idx: int,
    baseline_event: MetricEvent,
    target_event: MetricEvent,
    key_prefixes: list[str],
    rtol: float,
    *,
    atol: float,
    exclude_keys: list[str] | None = None,
) -> list[str]:
    issues: list[str] = []
    for key in baseline_event.metrics:
        if not any(key.startswith(prefix) for prefix in key_prefixes):
            continue
        if exclude_keys and key in exclude_keys:
            continue

        if key not in target_event.metrics:
            issues.append(f"Step {step_idx}: metric '{key}' present in baseline but missing in target")
            continue

        issues += _check_single_metric(step_idx, key, baseline_event.metrics[key], target_event.metrics[key], rtol, atol=atol)
    return issues


def _check_single_metric(
    step_idx: int,
    key: str,
    baseline_val: object,
    target_val: object,
    rtol: float,
    atol: float,
) -> list[str]:
    if not isinstance(baseline_val, (int, float)) or not isinstance(target_val, (int, float)):
        return []

    if math.isnan(baseline_val) or math.isnan(target_val):
        return [f"Step {step_idx}, metric '{key}': NaN detected (baseline={baseline_val}, target={target_val})"]
    if math.isinf(baseline_val) or math.isinf(target_val):
        if baseline_val != target_val:
            return [f"Step {step_idx}, metric '{key}': inf mismatch (baseline={baseline_val}, target={target_val})"]
        return []

    if baseline_val == 0.0 and target_val == 0.0:
        return []

    abs_diff = abs(baseline_val - target_val)
    if abs_diff <= atol:
        return []

    rel_diff = abs_diff / max(abs(baseline_val), abs(target_val), 1e-12)
    if rel_diff > rtol:
        return [
            f"Step {step_idx}, metric '{key}': baseline={baseline_val}, target={target_val}, "
            f"rel_diff={rel_diff:.6f} > rtol={rtol}"
        ]
    return []


def _print_step_comparison_table(
    step_idx: int,
    baseline_event: MetricEvent,
    target_event: MetricEvent,
    key_prefixes: list[str],
    *,
    exclude_keys: list[str] | None = None,
) -> None:
    import polars as pl
    from sglang.srt.debug_utils.comparator.display import _render_polars_as_text

    rows: list[dict[str, str]] = []
    for key in sorted(baseline_event.metrics):
        if not any(key.startswith(p) for p in key_prefixes):
            continue
        b_val = baseline_event.metrics[key]
        t_val = target_event.metrics.get(key)
        if not isinstance(b_val, (int, float)) or t_val is None or not isinstance(t_val, (int, float)):
            continue
        excluded = "(excluded)" if exclude_keys and key in exclude_keys else ""
        abs_diff = abs(b_val - t_val)
        denom = max(abs(b_val), abs(t_val), 1e-12)
        rel_diff = abs_diff / denom
        rows.append({
            "metric": key,
            "baseline": f"{b_val:.6e}",
            "target": f"{t_val:.6e}",
            "abs_diff": f"{abs_diff:.2e}",
            "rel_diff": f"{rel_diff:.4%}{excluded}",
        })

    if not rows:
        return
    df = pl.DataFrame(rows)
    print(_render_polars_as_text(df, title=f"Step {step_idx} metric comparison"))


def _check_required_keys_exist(events: list[MetricEvent]) -> list[str]:
    all_keys: set[str] = set()
    for event in events:
        all_keys.update(event.metrics.keys())

    issues: list[str] = []
    for required in _REQUIRED_METRIC_KEYS:
        if required not in all_keys:
            issues.append(
                f"Required metric '{required}' not found in any baseline MetricEvent. "
                f"Available keys: {sorted(all_keys)}"
            )
    return issues


def _run_comparator(
    *,
    baseline_path: Path,
    target_path: Path,
    diff_threshold: float,
    allow_skipped_pattern: str,
    allow_failed_pattern: str,
    extra_args: list[str] | None,
) -> subprocess.CompletedProcess[str]:
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
        # Skip 'rank' when grouping bundles: under FT (target) and non-FT (baseline)
        # the same logical (pp_rank, cp_rank, ep_rank, tp_rank) coordinate maps to a
        # different absolute rank ID (e.g. baseline rank=4 vs target cell0 rank=2 for
        # PP=1, CP=0). Without skipping 'rank' the comparator gets `baseline_load_failed`
        # for every tensor and fails with rc=1.
        "--grouping-skip-keys",
        "rank",
        "--diff-threshold",
        str(diff_threshold),
        "--allow-skipped-pattern",
        allow_skipped_pattern,
        "--allow-failed-pattern",
        allow_failed_pattern,
    ]
    if extra_args:
        cmd.extend(extra_args)

    result: subprocess.CompletedProcess[str] = subprocess.run(
        cmd,
        text=True,
    )

    return result


def _read_metric_events(dump_dir: Path) -> list[MetricEvent]:
    """Read all MetricEvents from the events directory."""
    events_dir: Path = dump_dir / "events"
    if not events_dir.exists():
        return []
    all_events = read_events(events_dir)
    return [e for e in all_events if isinstance(e, MetricEvent)]
