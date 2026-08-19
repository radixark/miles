import logging
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import polars as pl
from sglang.srt.debug_utils.comparator.display import _render_polars_as_text

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME, read_events
from miles.utils.audit_utils.event_logger.models import MetricEvent

logger = logging.getLogger(__name__)

_REQUIRED_METRIC_KEYS: list[str] = ["train/grad_norm", "train/loss"]


def compare_metrics(
    baseline_dir: str,
    target_dir: str,
    *,
    rtol: float,
    atol: float,
    key_prefixes: list[str],
    exclude_keys: list[str],
) -> None:
    baseline_events = _read_metric_events(Path(baseline_dir))
    target_events = _read_metric_events(Path(target_dir))

    # FT retries (healing path) leave events from earlier failed attempts. Only
    # the highest-attempt events per rollout_id reflect the successful run.
    baseline_events = _keep_only_final_attempt(baseline_events)
    target_events = _keep_only_final_attempt(target_events)

    issues: list[str] = []
    issues += _check_event_counts(baseline_events, target_events, baseline_dir, target_dir)
    issues += _check_events_line_up(baseline_events, target_events)

    if not issues:
        for step_idx, (b_event, t_event) in enumerate(zip(baseline_events, target_events, strict=True)):
            _print_step_comparison_table(step_idx, b_event, t_event, key_prefixes, exclude_keys=exclude_keys)
            issues += _check_step_metrics(
                step_idx, b_event, t_event, key_prefixes, rtol, atol=atol, exclude_keys=exclude_keys
            )

    issues += _check_required_keys_exist(baseline_events)

    assert not issues, f"MetricEvent comparison found {len(issues)} issue(s):\n" + "\n".join(
        f"  - {i}" for i in issues
    )
    print(f"MetricEvent comparison passed: {len(baseline_events)} steps compared")


def read_metric_series(dump_dir: str, *, key: str) -> list[tuple[int, float]]:
    events = _keep_only_final_attempt(_read_metric_events(Path(dump_dir)))
    return [
        (event.rollout_id, float(value))
        for event in events
        if isinstance(value := event.metrics.get(key), (int, float)) and not isinstance(value, bool)
    ]


def assert_gradients_nonzero(*, side: str, dump_dir: str, min_trained_rollouts: int) -> None:
    series = read_metric_series(dump_dir, key="train/grad_norm")
    usable = [(rollout_id, value) for rollout_id, value in series if math.isfinite(value) and value != 0.0]

    assert len(usable) >= min_trained_rollouts, (
        f"{side}: train/grad_norm is finite and non-zero in only {len(usable)} of {len(series)} rollout(s) "
        f"({series}), so this run's weights could have moved on weight decay alone"
    )


def assert_metrics_classified(dump_dir: str, *, compared: tuple[str, ...], ignored: tuple[str, ...]) -> None:
    keys = {key for event in _keep_only_final_attempt(_read_metric_events(Path(dump_dir))) for key in event.metrics}
    unclassified: list[str] = sorted(key for key in keys if not key.startswith(compared + ignored))

    assert not unclassified, (
        f"metrics {unclassified} belong to no namespace this comparison has classified, so they would be dropped "
        f"from one that claims to cover everything; add them to the compared prefixes {list(compared)}, or to the "
        f"ignored ones {list(ignored)} with a reason they cannot match"
    )


def read_rollout_completion_times(dump_dir: str) -> list[tuple[int, datetime]]:
    events = _keep_only_final_attempt(_read_metric_events(Path(dump_dir)))
    return sorted(
        ((event.rollout_id, event.timestamp) for event in events if event.rollout_id is not None),
        key=lambda one: one[1],
    )


def _check_events_line_up(baseline_events: list[MetricEvent], target_events: list[MetricEvent]) -> list[str]:
    return [
        f"step {index}: baseline is rollout {b.rollout_id} while target is rollout {t.rollout_id}, so the two "
        f"sides are not describing the same step"
        for index, (b, t) in enumerate(zip(baseline_events, target_events, strict=False))
        if b.rollout_id != t.rollout_id
    ]


def _keep_only_final_attempt(events: list[MetricEvent]) -> list[MetricEvent]:
    """Keep only events from the highest-attempt for each rollout_id.

    During FT healing, a crashed rollout is retried at attempt+1; events from
    the failed attempt are partial and should be discarded for comparison.

    Rollout-side metrics (e.g. RolloutExecutor log_rollout_metrics) have
    attempt=None — they are not part of the FT retry stream, so we treat them
    as a single attempt (normalized to 0).
    """

    def _attempt(e: MetricEvent) -> int:
        return e.attempt if e.attempt is not None else 0

    max_attempt_by_rollout: dict[int, int] = defaultdict(int)
    for e in events:
        max_attempt_by_rollout[e.rollout_id] = max(max_attempt_by_rollout[e.rollout_id], _attempt(e))
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

        issues += _check_single_metric(
            step_idx, key, baseline_event.metrics[key], target_event.metrics[key], rtol, atol=atol
        )
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
        rows.append(
            {
                "metric": key,
                "baseline": f"{b_val:.6e}",
                "target": f"{t_val:.6e}",
                "abs_diff": f"{abs_diff:.2e}",
                "rel_diff": f"{rel_diff:.4%}{excluded}",
            }
        )

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


def _read_metric_events(dump_dir: Path) -> list[MetricEvent]:
    """Read all MetricEvents from the events directory."""
    events_dir: Path = dump_dir / EVENTS_DIRNAME
    if not events_dir.exists():
        return []
    all_events = read_events(events_dir)
    return [e for e in all_events if isinstance(e, MetricEvent)]
