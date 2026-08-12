"""Heuristic sglang config-tuning suggestions for the Efficiency view.

v1: a handful of observed-vs-configured comparisons (design doc's "config
tuning advisory" ask). These are heuristics meant as a starting point, not a
guarantee — thresholds are expected to get tuned as real runs surface false
positives/negatives.
"""

from __future__ import annotations

from dataclasses import dataclass

from miles.dashboard.store import MetricStore, Stream

# how far below the configured cap "peak usage" must stay before flagging it
# as headroom worth reclaiming (colocate scenarios care about this most)
LOW_CONCURRENCY_RATIO = 0.3
LOW_CACHE_HIT_RATE = 0.10
HIGH_TOKEN_USAGE = 0.95

DEFAULT_LOW_MFU = 0.15
MFU_KEY = "perf/actor_train_mfu"
MFU_PEAK_KEY = "perf/mfu_peak_tflops"
MFU_STEP_KEY = "rollout/step"
MFU_MIN_STEPS = 3


@dataclass
class Advisory:
    level: str  # "info" | "warning"
    message: str


def _aggregate(series: list[dict], *, agg: str) -> float | None:
    """One scalar across every engine/value in a ``MetricStore.engine_series``
    result — ``agg`` is "max" or "mean"."""
    values = [v for s in series for v in s["value"]]
    if not values:
        return None
    return max(values) if agg == "max" else sum(values) / len(values)


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


def compute_advisories(
    store: MetricStore,
    *,
    t0: float | None = None,
    t1: float | None = None,
    mfu: dict | None = None,
    low_mfu: float = DEFAULT_LOW_MFU,
) -> list[Advisory]:
    out: list[Advisory] = _mfu_advisories(mfu if mfu is not None else mfu_summary(store), low_mfu)
    if not store.has_stream(Stream.ENGINE_SERIES):
        return out
    args = store.meta.args if store.meta else {}
    peak_running = _aggregate(store.engine_series("sglang_num_running_reqs", t0=t0, t1=t1), agg="max")
    cache_hit = _aggregate(store.engine_series("sglang_cache_hit_rate", t0=t0, t1=t1), agg="mean")
    token_usage = _aggregate(store.engine_series("sglang_token_usage", t0=t0, t1=t1), agg="mean")

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

    if token_usage is not None and token_usage > HIGH_TOKEN_USAGE:
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
