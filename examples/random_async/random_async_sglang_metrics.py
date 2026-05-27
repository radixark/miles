"""Example-local SGLang metrics scraper for random_async.

Miles can forward raw SGLang OpenMetrics to W&B, but this helper logs a small
set of stable aggregate keys under ``random_async_sglang/`` for the stress test.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

METRICS_INTERVAL_SECONDS = float(os.environ.get("RANDOM_ASYNC_SGLANG_METRICS_INTERVAL_SECONDS", "30"))

_METRIC_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')
_AGENT_METRICS_LOCK = threading.Lock()
_AGENT_WINDOW: dict[str, float] = {
    "output_tokens": 0.0,
    "prompt_tokens": 0.0,
    "cached_tokens": 0.0,
    "perfect_cacheable_tokens": 0.0,
    "request_time": 0.0,
    "max_request_time": 0.0,
    "request_count": 0.0,
}
_AGENT_CUMULATIVE: dict[str, float] = {
    "output_tokens": 0.0,
    "prompt_tokens": 0.0,
    "cached_tokens": 0.0,
    "perfect_cacheable_tokens": 0.0,
    "request_count": 0.0,
}


@dataclass(frozen=True)
class MetricSample:
    name: str
    labels: dict[str, str]
    value: float


def record_agent_request(
    *,
    output_tokens: int,
    prompt_tokens: int,
    cached_tokens: int,
    perfect_cacheable_tokens: int,
    request_time: float,
) -> None:
    with _AGENT_METRICS_LOCK:
        _AGENT_WINDOW["output_tokens"] += output_tokens
        _AGENT_WINDOW["prompt_tokens"] += prompt_tokens
        _AGENT_WINDOW["cached_tokens"] += cached_tokens
        _AGENT_WINDOW["perfect_cacheable_tokens"] += perfect_cacheable_tokens
        _AGENT_WINDOW["request_time"] += request_time
        _AGENT_WINDOW["max_request_time"] = max(_AGENT_WINDOW["max_request_time"], request_time)
        _AGENT_WINDOW["request_count"] += 1


def _pop_agent_metrics() -> dict[str, float]:
    with _AGENT_METRICS_LOCK:
        window = dict(_AGENT_WINDOW)
        for key in _AGENT_WINDOW:
            _AGENT_WINDOW[key] = 0.0

    if window["request_count"] == 0:
        return {}

    _AGENT_CUMULATIVE["output_tokens"] += window["output_tokens"]
    _AGENT_CUMULATIVE["prompt_tokens"] += window["prompt_tokens"]
    _AGENT_CUMULATIVE["cached_tokens"] += window["cached_tokens"]
    _AGENT_CUMULATIVE["perfect_cacheable_tokens"] += window["perfect_cacheable_tokens"]
    _AGENT_CUMULATIVE["request_count"] += window["request_count"]

    prompt_tokens = window["prompt_tokens"]
    cached_tokens = window["cached_tokens"]
    perfect_cacheable_tokens = window["perfect_cacheable_tokens"]
    request_count = window["request_count"]
    return {
        "random_async_sglang/agent_perfect_cache_hit_rate": (
            perfect_cacheable_tokens / prompt_tokens if prompt_tokens > 0 else 0.0
        ),
        "random_async_sglang/agent_cache_to_perfect_cache_ratio": (
            cached_tokens / perfect_cacheable_tokens if perfect_cacheable_tokens > 0 else 0.0
        ),
        "random_async_sglang/agent_avg_request_time": window["request_time"] / request_count,
        "random_async_sglang/agent_max_request_time": window["max_request_time"],
    }


def _parse_labels(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    return {
        match.group(1): bytes(match.group(2), "utf-8").decode("unicode_escape") for match in _LABEL_RE.finditer(raw)
    }


def parse_openmetrics(text: str) -> list[MetricSample]:
    samples: list[MetricSample] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_RE.match(line)
        if match is None:
            continue
        samples.append(
            MetricSample(
                name=match.group("name").replace(":", "_"),
                labels=_parse_labels(match.group("labels")),
                value=float(match.group("value")),
            )
        )
    return samples


def _matches_engine_type(labels: dict[str, str], engine_type: str | None) -> bool:
    if engine_type is None:
        return True
    return labels["engine_type"] == engine_type


def _sum(samples: list[MetricSample], name: str, *, engine_type: str | None = None, mode: str | None = None) -> float:
    total = 0.0
    for sample in samples:
        if sample.name != name:
            continue
        if not _matches_engine_type(sample.labels, engine_type):
            continue
        if mode is not None and sample.labels.get("mode") != mode:
            continue
        total += sample.value
    return total


def _has(samples: list[MetricSample], name: str, *, engine_type: str | None = None, mode: str | None = None) -> bool:
    return any(
        sample.name == name
        and _matches_engine_type(sample.labels, engine_type)
        and (mode is None or sample.labels.get("mode") == mode)
        for sample in samples
    )


def _avg(samples: list[MetricSample], name: str, *, engine_type: str | None = None) -> float | None:
    values = [
        sample.value for sample in samples if sample.name == name and _matches_engine_type(sample.labels, engine_type)
    ]
    if not values:
        return None
    return sum(values) / len(values)


class SGLangMetricsReporter:
    def __init__(self, router_url: str, prefill_num_gpus: int, decode_num_gpus: int):
        self.router_url = router_url.rstrip("/")
        self.prefill_num_gpus = max(1, prefill_num_gpus)
        self.decode_num_gpus = max(1, decode_num_gpus)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.failure: BaseException | None = None
        self._previous_realtime_tokens: dict[str, float] = {}
        self._previous_time: float | None = None
        self._last_failure_log_time = 0.0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="random-async-sglang-metrics", daemon=True)
        self._thread.start()
        logger.info("Started random_async SGLang metrics reporter for %s", self.router_url)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)

    def _run(self) -> None:
        session = requests.Session()
        while not self._stop_event.is_set():
            start = time.time()
            metrics: dict[str, float] = {}
            try:
                response = session.get(f"{self.router_url}/engine_metrics", timeout=10)
                response.raise_for_status()
                samples = parse_openmetrics(response.text)
                metrics.update(self._build_metrics(samples, start))
                metrics.update(_pop_agent_metrics())
                if metrics:
                    self._log_to_wandb(metrics)
            except Exception:
                now = time.time()
                if now - self._last_failure_log_time > 300:
                    logger.warning("random_async SGLang metrics scrape failed; continuing", exc_info=True)
                    self._last_failure_log_time = now
            elapsed = time.time() - start
            self._stop_event.wait(max(1.0, METRICS_INTERVAL_SECONDS - elapsed))

    def _build_metrics(self, samples: list[MetricSample], now: float) -> dict[str, float]:
        metrics: dict[str, float] = {}

        if _has(samples, "sglang_gen_throughput", engine_type="decode"):
            decode_gen_throughput = _sum(samples, "sglang_gen_throughput", engine_type="decode")
            metrics["random_async_sglang/sglang_reported_decode_gen_throughput_per_decode_gpu"] = (
                decode_gen_throughput / self.decode_num_gpus
            )

        prefill_cache_hit_rate = _avg(samples, "sglang_cache_hit_rate", engine_type="prefill")
        if prefill_cache_hit_rate is not None:
            metrics["random_async_sglang/sglang_cache_hit_rate"] = prefill_cache_hit_rate

        for engine_type in ("prefill", "decode"):
            for name, suffix in (
                ("sglang_token_usage", "token_usage"),
                ("sglang_num_queue_reqs", "queue_reqs"),
                ("sglang_num_running_reqs", "running_reqs"),
            ):
                if _has(samples, name, engine_type=engine_type):
                    value = _sum(samples, name, engine_type=engine_type)
                    metrics[f"random_async_sglang/{engine_type}_{suffix}"] = value

        for name, key in (
            ("sglang_num_prefill_prealloc_queue_reqs", "prefill_prealloc_queue_reqs"),
            ("sglang_num_prefill_inflight_queue_reqs", "prefill_inflight_queue_reqs"),
            ("sglang_num_decode_prealloc_queue_reqs", "decode_prealloc_queue_reqs"),
            ("sglang_num_decode_transfer_queue_reqs", "decode_transfer_queue_reqs"),
            ("sglang_kv_transfer_speed_gb_s", "kv_transfer_speed_gb_s"),
            ("sglang_kv_transfer_latency_ms", "kv_transfer_latency_ms"),
            ("sglang_kv_transfer_bootstrap_ms", "kv_transfer_bootstrap_ms"),
            ("sglang_kv_transfer_alloc_ms", "kv_transfer_alloc_ms"),
            ("sglang_kv_transfer_total_mb", "kv_transfer_total_mb"),
            ("sglang_num_retracted_reqs", "retracted_reqs"),
            ("sglang_num_bootstrap_failed_reqs_total", "bootstrap_failed_reqs_total"),
            ("sglang_num_transfer_failed_reqs_total", "transfer_failed_reqs_total"),
            ("sglang_num_prefill_retries_total", "prefill_retries_total"),
        ):
            if _has(samples, name):
                value = _sum(samples, name)
                metrics[f"random_async_sglang/{key}"] = value

        realtime_tokens = {
            mode: _sum(samples, "sglang_realtime_tokens_total", mode=mode)
            for mode in ("decode", "prefill_compute", "prefill_cache")
        }
        if self._previous_time is not None:
            dt = max(now - self._previous_time, 1e-6)
            tps_by_mode: dict[str, float] = {}
            for mode, value in realtime_tokens.items():
                previous = self._previous_realtime_tokens.get(mode)
                if previous is None:
                    continue
                delta = value - previous
                if delta >= 0:
                    tps_by_mode[mode] = delta / dt
            if "decode" in tps_by_mode:
                metrics["random_async_sglang/decode_tps_per_decode_gpu"] = tps_by_mode["decode"] / self.decode_num_gpus
            if "prefill_compute" in tps_by_mode:
                metrics["random_async_sglang/prefill_without_cache_tps_per_prefill_gpu"] = (
                    tps_by_mode["prefill_compute"] / self.prefill_num_gpus
                )
            if "prefill_cache" in tps_by_mode:
                metrics["random_async_sglang/prefill_cache_read_tps_per_prefill_gpu"] = (
                    tps_by_mode["prefill_cache"] / self.prefill_num_gpus
                )
            prefill_total_tps = tps_by_mode.get("prefill_compute", 0.0) + tps_by_mode.get("prefill_cache", 0.0)
            if prefill_total_tps > 0:
                metrics["random_async_sglang/prefill_total_tps_per_prefill_gpu"] = (
                    prefill_total_tps / self.prefill_num_gpus
                )
        self._previous_realtime_tokens = realtime_tokens
        self._previous_time = now

        return metrics

    def _log_to_wandb(self, metrics: dict[str, float]) -> None:
        import wandb

        if wandb.run is None:
            raise RuntimeError("random_async SGLang metrics reporter requires an active W&B run")
        wandb.log(metrics)
