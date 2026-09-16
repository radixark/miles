"""Heuristic run-health alarms and sglang config-tuning suggestions.

v2 orders the rules by what they claim: pipeline-liveness alarms first
(stalled phases, engine abort storms, degenerate reward groups), config
tuning last — and only on a run with no active alarm. The same observation
(low concurrency, low cache hit) means opposite things on a healthy and on
a starved run, so tuning advice on an unhealthy run would point the wrong
way: every production incident that motivated these rules began with "low
utilization" that was a symptom, not headroom.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from statistics import median

from miles.dashboard.dump_reader import DumpReader
from miles.dashboard.store import MetricStore, Stream

# an open phase this much beyond its own median closed duration is a stall,
# not a slow interval (with a floor so short phases don't false-positive)
STALL_FACTOR = 6.0
STALL_MIN_AGE_S = 1800.0
STALL_MIN_CLOSED = 3

ABORT_RATIO_WARN = 0.05
ABORT_WINDOW_S = 1800.0
# a short window can cover an idle stretch; 1/12 aborted is not a storm
ABORT_MIN_REQUESTS = 100.0
ZERO_STD_FRAC_WARN = 0.5
TRUNCATED_FRAC_WARN = 0.25
KV_BOUND_TOKEN_USAGE = 0.90
KV_BOUND_RUNNING_RATIO = 0.5

# how far below the configured cap "peak usage" must stay before flagging it
# as headroom worth reclaiming (colocate scenarios care about this most)
LOW_CONCURRENCY_RATIO = 0.3
LOW_CACHE_HIT_RATE = 0.10
HIGH_TOKEN_USAGE = 0.95
# dp-attention imbalance: flag when the busiest dp rank carries real load but
# the idlest sits under this fraction of it (requests piling on few ranks)
DP_IMBALANCE_MIN_RUNNING = 1.0
DP_IMBALANCE_RATIO = 0.25

DEFAULT_LOW_MFU = 0.15
MFU_KEY = "perf/actor_train_mfu"
MFU_PEAK_KEY = "perf/mfu_peak_tflops"
MFU_STEP_KEY = "rollout/step"
MFU_MIN_STEPS = 3


@dataclass
class Advisory:
    level: str  # "critical" | "warning" | "info"
    message: str


def compute_advisories(
    store: MetricStore,
    reader: DumpReader | None = None,
    *,
    t0: float | None = None,
    t1: float | None = None,
    mfu: dict | None = None,
    low_mfu: float = DEFAULT_LOW_MFU,
) -> list[Advisory]:
    args = store.meta.args if store.meta else {}
    alarms = _stall_advisories(store)
    alarms += _engine_alarm_advisories(store, args, t0, t1)
    if reader is not None:
        alarms += _rollout_advisories(reader, args)
    if alarms:
        return alarms
    return _tuning_advisories(store, args, t0, t1, mfu=mfu, low_mfu=low_mfu)


def _fmt_duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{seconds / 60:.0f}min"
    return f"{seconds:.0f}s"


def _stall_advisories(store: MetricStore) -> list[Advisory]:
    """Open phases far beyond their own median closed duration.

    Judged at the newest data timestamp over the whole run (not the query
    window): a stall is a statement about the current state, and the open
    marker may be arbitrarily older than any window the viewer looks at."""
    time_range = store.time_range()
    if time_range is None:
        return []
    edge = time_range[1]
    # resuming into the same dump dir appends to the existing streams, so the
    # crashed attempt's open markers are still there and never close; measuring
    # their age against this attempt's clock would latch a critical forever.
    # meta.json is rewritten per attempt, so start_ts is the boundary (same one
    # phases_by_lane anchors the synthesized initialize band to). It also keeps
    # the previous attempt's closed durations out of the median baseline.
    attempt_start = store.meta.start_ts if store.meta else float("-inf")
    events = [event for event in store.phase_events() if event.t0 >= attempt_start]
    closed_keys = {(e.node, e.rank, e.name, e.t0) for e in events if not e.open}
    durations: dict[str, list[float]] = {}
    open_ages: dict[str, list[float]] = {}
    for event in events:
        if event.open:
            if (event.node, event.rank, event.name, event.t0) not in closed_keys:
                open_ages.setdefault(event.name, []).append(edge - event.t0)
        else:
            durations.setdefault(event.name, []).append(event.t1 - event.t0)

    out = []
    for name, ages in sorted(open_ages.items()):
        closed = durations.get(name, [])
        if len(closed) < STALL_MIN_CLOSED:
            continue  # phases that are open by design never close: no baseline, no claim
        typical = median(closed)
        age = max(ages)
        if age <= max(STALL_FACTOR * typical, STALL_MIN_AGE_S):
            continue
        message = (
            f"Phase '{name}' has been open for {_fmt_duration(age)} on {len(ages)} rank(s) — "
            f"{age / typical:.0f}x its median duration ({_fmt_duration(typical)}); "
            "the pipeline is stalled, not slow"
        )
        if name == "train_wait":
            message += " (training is starved of rollout data — check the data buffer and engine health)"
        out.append(Advisory(level="critical", message=message))
    return out


def _counter_delta(series: list[dict], *, since: float = float("-inf")) -> float:
    """Total increase of a cumulative counter over ``engine_series`` results.

    Summed per interval, dropping the negative ones: an engine restarting on
    the same address resets its counters mid-series, and last-minus-first
    would then read the post-restart segment alone (same reset rule as
    ``MetricStore._derived_series``). ``since`` keeps only intervals ending at
    or after it."""
    total = 0.0
    for one in series:
        for ts, (before, after) in zip(one["ts"][1:], pairwise(one["value"]), strict=True):
            if ts >= since:
                total += max(after - before, 0.0)
    return total


def _engine_alarm_advisories(store: MetricStore, args: dict, t0: float | None, t1: float | None) -> list[Advisory]:
    if not store.has_stream(Stream.ENGINE_SERIES):
        return []
    out = []

    # like the stall watchdog, the abort ratio claims something about the
    # engine's current state, so it is judged over a recent window rather than
    # the query window: a run-lifetime ratio stays above the threshold long
    # after the storm cleared, latching the alarm — and with it the suppression
    # of the tuning tier. The window is anchored to the newest engine scrape,
    # not store.time_range(): when the engines die the other streams keep
    # advancing, and a window that slid past the engine data would drop the
    # alarm exactly when it matters most.
    requests = store.engine_series("sglang_num_requests_total")
    aborts = store.engine_series("sglang_num_aborted_requests_total")
    edge = max((one["ts"][-1] for one in requests + aborts if one["ts"]), default=0.0)
    total = _counter_delta(requests, since=edge - ABORT_WINDOW_S)
    aborted = _counter_delta(aborts, since=edge - ABORT_WINDOW_S)
    if total >= ABORT_MIN_REQUESTS and aborted / total > ABORT_RATIO_WARN:
        run_total = _counter_delta(requests)
        run_aborted = _counter_delta(aborts)
        message = (
            f"{aborted / total:.0%} of engine requests were aborted client-side in the last "
            f"{_fmt_duration(ABORT_WINDOW_S)} ({aborted:.0f}/{total:.0f}; "
            f"{run_aborted / run_total:.0%} over the run) — request timeouts are poisoning samples"
        )
        group_size = args.get("n_samples_per_prompt")
        if group_size:
            message += (
                f"; one aborted sample discards its whole {group_size:g}-sample group, "
                "and backfill resubmission amplifies the engine load"
            )
        out.append(Advisory(level="warning", message=message))

    peak_running = _aggregate(store.engine_series("sglang_num_running_reqs", t0=t0, t1=t1), agg="max")
    token_usage = _aggregate(store.engine_series("sglang_token_usage", t0=t0, t1=t1), agg="mean")
    max_running = args.get("sglang_max_running_requests")
    if token_usage is not None and token_usage > KV_BOUND_TOKEN_USAGE:
        if max_running and peak_running is not None and peak_running < KV_BOUND_RUNNING_RATIO * max_running:
            mem_fraction = args.get("sglang_mem_fraction_static")
            out.append(
                Advisory(
                    level="warning",
                    message=(
                        f"KV cache usage is pegged at {token_usage:.0%} while peak concurrency "
                        f"({peak_running:.0f}) stays far below --sglang-max-running-requests ({max_running:g}): "
                        "concurrency is KV-pool-bound, not cap-bound; raise --sglang-mem-fraction-static"
                        + (f" (currently {mem_fraction:g})" if mem_fraction is not None else "")
                        + " — do NOT lower the request cap"
                    ),
                )
            )
        elif token_usage > HIGH_TOKEN_USAGE:
            out.append(
                Advisory(
                    level="warning",
                    message=(
                        f"KV cache usage is consistently high ({token_usage:.1%}) — likely a throughput "
                        "bottleneck; consider more GPUs or a smaller rollout batch"
                    ),
                )
            )
    return out


def _rollout_advisories(reader: DumpReader, args: dict) -> list[Advisory]:
    """Degenerate-signal checks on the newest completed train rollout."""
    train_ids = reader.rollout_ids().train
    if not train_ids:
        return []
    rollout_id = train_ids[-1]
    out = []

    groups = reader.groups(rollout_id)
    with_rewards = groups.filter(groups["reward_mean"].is_not_null())
    if len(with_rewards):
        zero_std = int(with_rewards["zero_std"].sum())
        if zero_std / len(with_rewards) >= ZERO_STD_FRAC_WARN:
            message = (
                f"{zero_std}/{len(with_rewards)} groups in rollout {rollout_id} have zero reward std — "
                "those groups contribute zero policy gradient"
            )
            if with_rewards["reward_mean"].abs().max() < 1e-12:
                message += (
                    "; every reward is 0, pointing at systematic failure (timeouts/truncation), not task difficulty"
                )
            out.append(Advisory(level="warning", message=message))

    summary = reader.summary(rollout_id)
    if summary.height == 0:
        return out
    truncated_frac = summary["truncated"].cast(float).mean()
    if truncated_frac is not None and truncated_frac >= TRUNCATED_FRAC_WARN:
        message = f"{truncated_frac:.0%} of samples in rollout {rollout_id} are truncated"
        cap = args.get("rollout_max_response_len")
        if cap:
            message += (
                f" — check --rollout-max-response-len ({cap:g}): in multi-turn agentic runs "
                "the per-turn cap is usually the binding limit, not the context length"
            )
        out.append(Advisory(level="warning", message=message))

    if "alignment_failed" in summary.columns and bool(summary["alignment_failed"].any()):
        unreadable = int(summary["alignment_failed"].sum())
        out.append(
            Advisory(
                level="info",
                message=(
                    f"per-token columns are blank for {unreadable} samples in rollout {rollout_id}: their "
                    "context-parallel slices could not be placed, because this dump predates the cp_rank/cp_size "
                    "fields. Blank here means unreadable, not undumped"
                ),
            )
        )
    return out


def _aggregate(series: list[dict], *, agg: str) -> float | None:
    """One scalar across every engine/value in a ``MetricStore.engine_series``
    result — ``agg`` is "max" or "mean"."""
    values = [v for s in series for v in s["value"]]
    if not values:
        return None
    return max(values) if agg == "max" else sum(values) / len(values)


def _dp_rank_means(series: list[dict]) -> dict[str, dict[str, float]]:
    """Per-engine ``{dp_rank: mean(value)}`` from a ``per_dp_rank`` result;
    series without a dp_rank label (non-dp engines) are skipped."""
    out: dict[str, dict[str, float]] = {}
    for s in series:
        rank = s["labels"].get("dp_rank")
        if rank is None or not s["value"]:
            continue
        out.setdefault(s["addr"], {})[rank] = sum(s["value"]) / len(s["value"])
    return out


def _dp_spread_hint(args: dict) -> str:
    """Which knob actually decides how requests spread across dp ranks. Only a
    dp-aware sglang router routes per rank; otherwise the engine looks like one
    worker and sglang's own dp controller dispatches."""
    if args.get("use_miles_router"):
        return "the miles router routes per engine; --sglang-load-balance-method decides the rank"
    if not args.get("router_dp_aware"):
        return "the router sees one worker per engine; --router-dp-aware makes it route per rank"
    policy = args.get("sglang_router_policy") or args.get("router_policy")
    if policy == "manual":
        return (
            f"manual routing pins each key to a rank via --router-assignment-mode "
            f"({args.get('router_assignment_mode')}); min_load spreads by load"
        )
    return f"--router-policy ({policy}) picks the rank"


def mfu_summary(store: MetricStore) -> dict | None:
    series = store.metric_series([MFU_KEY, MFU_PEAK_KEY], x_key=MFU_STEP_KEY)
    steady = series[MFU_KEY]["y"][1:]
    if not steady:
        return None
    return dict(
        latest=steady[-1],
        mean=sum(steady) / len(steady),
        steps=len(steady),
        peak=series[MFU_PEAK_KEY]["y"][-1],
    )


def _mfu_advisories(summary: dict | None, low_mfu: float) -> list[Advisory]:
    if low_mfu <= 0 or summary is None or summary["steps"] < MFU_MIN_STEPS:
        return []
    mean_mfu, peak = summary["mean"], summary["peak"]
    if mean_mfu >= low_mfu:
        return []
    return [
        Advisory(
            level="warning",
            message=(
                f"Model FLOPs utilization averaged {mean_mfu:.1%} of the device's {peak:g} TFLOP/s "
                f"over {summary['steps']} train steps — "
                "the training step is computing slowly, not waiting: this ratio counts actor train time only, "
                "so rollout stalls cannot depress it. Usual causes are activation recompute, a parallel split "
                "that leaves ranks idle, and small or ragged micro-batches"
            ),
        )
    ]


def _tuning_advisories(
    store: MetricStore,
    args: dict,
    t0: float | None,
    t1: float | None,
    *,
    mfu: dict | None = None,
    low_mfu: float = DEFAULT_LOW_MFU,
) -> list[Advisory]:
    out: list[Advisory] = _mfu_advisories(mfu if mfu is not None else mfu_summary(store), low_mfu)
    if not store.has_stream(Stream.ENGINE_SERIES):
        return out
    peak_running = _aggregate(store.engine_series("sglang_num_running_reqs", t0=t0, t1=t1), agg="max")
    cache_hit = _aggregate(store.engine_series("sglang_cache_hit_rate", t0=t0, t1=t1), agg="mean")

    colocate = bool(args.get("colocate"))

    max_running = args.get("sglang_max_running_requests")
    if max_running and peak_running is not None and peak_running < LOW_CONCURRENCY_RATIO * max_running:
        out.append(
            Advisory(
                level="info",
                message=(
                    f"Peak concurrency ({peak_running:.0f}) stayed under {LOW_CONCURRENCY_RATIO:.0%} of "
                    f"--sglang-max-running-requests ({max_running:g}); consider lowering it"
                    + (" to free memory for training (colocate)" if colocate else "")
                ),
            )
        )

    if not colocate and cache_hit is not None and cache_hit < LOW_CACHE_HIT_RATE:
        mem_fraction = args.get("sglang_mem_fraction_static")
        out.append(
            Advisory(
                level="info",
                message=(
                    f"Prefix cache hit rate is low ({cache_hit:.1%})"
                    + (
                        f"; consider raising --sglang-mem-fraction-static (currently {mem_fraction:g}) for a bigger KV cache"
                        if mem_fraction is not None
                        else ""
                    )
                ),
            )
        )

    per_rank = store.engine_series("sglang_num_running_reqs", t0=t0, t1=t1, per_dp_rank=True)
    for addr, means in sorted(_dp_rank_means(per_rank).items()):
        if len(means) < 2:
            continue
        busiest, idlest = max(means.values()), min(means.values())
        if busiest >= DP_IMBALANCE_MIN_RUNNING and idlest < DP_IMBALANCE_RATIO * busiest:
            detail = ", ".join(
                f"dp{rank}={mean:.1f}" for rank, mean in sorted(means.items(), key=lambda kv: int(kv[0]))
            )
            out.append(
                Advisory(
                    level="warning",
                    message=(
                        f"{addr}: dp ranks imbalanced (mean running reqs {detail}) — requests pile onto "
                        f"few ranks while others idle; {_dp_spread_hint(args)}"
                    ),
                )
            )
    return out
