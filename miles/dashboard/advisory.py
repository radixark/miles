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


def compute_advisories(store: MetricStore, *, t0: float | None = None, t1: float | None = None) -> list[Advisory]:
    if not store.has_stream(Stream.ENGINE_SERIES):
        return []  # no sglang scrape data to compare against
    args = store.meta.args if store.meta else {}
    peak_running = _aggregate(store.engine_series("sglang_num_running_reqs", t0=t0, t1=t1), agg="max")
    cache_hit = _aggregate(store.engine_series("sglang_cache_hit_rate", t0=t0, t1=t1), agg="mean")
    token_usage = _aggregate(store.engine_series("sglang_token_usage", t0=t0, t1=t1), agg="mean")

    out: list[Advisory] = []
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
