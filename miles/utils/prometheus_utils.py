import logging
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from miles.utils.misc import get_current_node_ip

logger = logging.getLogger(__name__)

_METRIC_PREFIX = "miles_metric_"
_COLLECTOR_ACTOR_NAME = "miles_prometheus_collector"
_GET_ACTOR_TIMEOUT = 60
_GET_ACTOR_INTERVAL = 2

_collector_handle = None


def init_prometheus(args, start_server: bool = False):
    """Initialize the Prometheus metric collector.

    The driver process (``start_server=True``) creates a named Ray
    actor that holds the HTTP server and all gauges.  Actor processes
    (``start_server=False``) look up the existing named actor.  Ray
    remote calls transport metrics across nodes transparently.
    """
    global _collector_handle

    if start_server:
        current_node_id = ray.get_runtime_context().get_node_id()
        _collector_handle = (
            ray.remote(_PrometheusCollector)
            .options(
                name=_COLLECTOR_ACTOR_NAME,
                # Hard-pin: the HTTP server must bind on the driver node where
                # the Prometheus scraper expects it.  Driver node failure kills
                # the entire Ray job anyway, so soft fallback adds no value.
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=current_node_id,
                    soft=False,
                ),
            )
            .remote(args)
        )
        ray.get(_collector_handle.ping.remote())
        logger.info("Prometheus collector actor created")
    else:
        deadline = time.monotonic() + _GET_ACTOR_TIMEOUT
        while True:
            try:
                _collector_handle = ray.get_actor(_COLLECTOR_ACTOR_NAME)
                break
            except ValueError:
                if time.monotonic() >= deadline:
                    logger.warning(
                        "Prometheus collector actor not found "
                        f"after {_GET_ACTOR_TIMEOUT}s, "
                        "metrics will not be reported"
                    )
                    return
                time.sleep(_GET_ACTOR_INTERVAL)


def get_prometheus():
    """Return the collector actor handle, or ``None``."""
    return _collector_handle


class _PrometheusCollector:
    """Ray actor that owns the Prometheus HTTP server and gauges.

    Runs on the driver node.  All processes push metrics here via
    ``handle.update.remote(metrics)`` — works across nodes because
    Ray handles the RPC transparently.

    Supports per-engine metrics via ``update_with_labels``.  Each
    unique combination of (run_name, engine_id, ...) creates a
    separate time-series, so Grafana can show per-engine breakdowns.
    """

    def __init__(self, args):
        from prometheus_client import Gauge, start_http_server

        self._Gauge = Gauge
        self._gauges: dict = {}
        self._run_name = (
            getattr(args, "prometheus_run_name", None) or getattr(args, "wandb_group", None) or "miles_training"
        )
        self._base_label_keys = ("run_name",)
        self._base_label_vals = (self._run_name,)

        port = args.prometheus_port
        start_http_server(port)

        bind_ip = get_current_node_ip()
        logger.info("Prometheus metrics server started on %s:%d, run_name=%s", bind_ip, port, self._run_name)

    def _ensure_gauge(self, safe_key: str, description: str, label_keys: tuple):
        """Get or create a Gauge with the given label set.

        If a gauge already exists under ``safe_key`` but with **different**
        label keys (e.g. first call had only ``run_name``, later call adds
        ``engine_id``), we create a separate gauge with a ``__`` suffix to
        avoid a ValueError from prometheus_client.
        """
        cache_key = (safe_key, label_keys)
        if cache_key not in self._gauges:
            try:
                self._gauges[cache_key] = self._Gauge(safe_key, description, label_keys)
            except ValueError:
                suffixed = safe_key + "__" + "_".join(label_keys)
                self._gauges[cache_key] = self._Gauge(suffixed, description, label_keys)
        return self._gauges[cache_key]

    def update(self, metrics: dict):
        """Set gauge values for all numeric metrics (aggregate, no engine_id)."""
        label_keys = self._base_label_keys
        label_vals = self._base_label_vals
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            safe_key = _METRIC_PREFIX + (key.replace("/", "_").replace("-", "_").replace("@", "_at_"))
            gauge = self._ensure_gauge(safe_key, key, label_keys)
            gauge.labels(*label_vals).set(value)

    def update_with_labels(self, metrics: dict, extra_labels: dict):
        """Set gauge values tagged with extra labels (e.g. engine_id).

        ``extra_labels`` is merged with the base labels (run_name) so each
        unique combination produces a separate Prometheus time-series.
        """
        label_keys = self._base_label_keys + tuple(sorted(extra_labels.keys()))
        label_vals = self._base_label_vals + tuple(extra_labels[k] for k in sorted(extra_labels.keys()))
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            safe_key = _METRIC_PREFIX + (key.replace("/", "_").replace("-", "_").replace("@", "_at_"))
            gauge = self._ensure_gauge(safe_key, key, label_keys)
            gauge.labels(*label_vals).set(value)

    def ping(self):
        return True
