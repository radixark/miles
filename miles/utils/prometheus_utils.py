import logging
import os
import tempfile

from miles.utils.misc import SingletonMeta

logger = logging.getLogger(__name__)

_PROMETHEUS_MULTIPROC_DIR = os.path.join(tempfile.gettempdir(), "prometheus_multiproc")
os.makedirs(_PROMETHEUS_MULTIPROC_DIR, exist_ok=True)
os.environ["PROMETHEUS_MULTIPROC_DIR"] = _PROMETHEUS_MULTIPROC_DIR

_METRIC_PREFIX = "miles_metric_"


class PrometheusAdapter(metaclass=SingletonMeta):
    """Prometheus adapter that works across multiple Ray actor processes.

    Uses prometheus_client's multiprocess mode: each process writes gauge
    values to files in a shared directory, and the HTTP server (started only
    by the primary process) aggregates them when Prometheus scrapes /metrics.
    """

    def __init__(self, args, start_server: bool = False):
        try:
            from prometheus_client import Gauge

            self._Gauge = Gauge
            self._gauges: dict = {}
            self._run_name = (
                getattr(args, "prometheus_run_name", None) or getattr(args, "wandb_group", None) or "miles_training"
            )
            self._label_keys = ["run_name"]
            self._label_vals = [self._run_name]

            if start_server:
                from prometheus_client import CollectorRegistry, multiprocess, start_http_server

                # The HTTP server needs a registry with `MultiProcessCollector`
                # to aggregate gauge files from all processes on scrape.
                registry = CollectorRegistry()
                multiprocess.MultiProcessCollector(registry)
                port = args.prometheus_port
                start_http_server(port, registry=registry)
                logger.info(f"Prometheus metrics server started on port {port}, " f"run_name={self._run_name}")
        except ImportError:
            logger.error("prometheus_client not installed. " "Run: pip install prometheus-client")
            raise
        except OSError as e:
            logger.warning(f"Prometheus HTTP server not started: {e}")

    def update(self, metrics: dict):
        """Set gauge values for all numeric metrics.

        Every gauge carries a ``run_name`` label so different training
        runs on the same or different Pods can be distinguished in Grafana.
        Prometheus will scrape these at its configured interval.
        """
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            safe_key = _METRIC_PREFIX + (key.replace("/", "_").replace("-", "_").replace("@", "_at_"))
            if safe_key not in self._gauges:
                self._gauges[safe_key] = self._Gauge(
                    safe_key,
                    key,
                    self._label_keys,
                )
            self._gauges[safe_key].labels(*self._label_vals).set(value)
