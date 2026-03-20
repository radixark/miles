import logging
import os
import tempfile

import wandb
from miles.utils.tensorboard_utils import _TensorboardAdapter

from . import wandb_utils

logger = logging.getLogger(__name__)

# Shared directory for prometheus_client multiprocess mode.
# Must be set BEFORE importing prometheus_client so the CollectorRegistry
# picks it up via the PROMETHEUS_MULTIPROC_DIR env var.
_PROMETHEUS_MULTIPROC_DIR = os.path.join(tempfile.gettempdir(), "prometheus_multiproc")
os.makedirs(_PROMETHEUS_MULTIPROC_DIR, exist_ok=True)
os.environ["PROMETHEUS_MULTIPROC_DIR"] = _PROMETHEUS_MULTIPROC_DIR


class _PrometheusAdapter:
    """Prometheus adapter that works across multiple Ray actor processes.

    Uses prometheus_client's multiprocess mode: each process writes gauge
    values to files in a shared directory, and the HTTP server (started only
    by the primary process) aggregates them when Prometheus scrapes /metrics.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, args, start_server: bool = False):
        if self._initialized:
            return

        try:
            self._gauges = {}
            self._run_name = (
                getattr(args, "prometheus_run_name", None)
                or getattr(args, "wandb_group", None)
                or "miles_training"
            )
            self._label_keys = ["run_name"]
            self._label_vals = [self._run_name]
            self._initialized = True

            if start_server:
                from prometheus_client import (
                    CollectorRegistry,
                    multiprocess,
                    start_http_server,
                )

                # The HTTP server needs a registry with MultiProcessCollector
                # to aggregate gauge files from all processes on scrape.
                registry = CollectorRegistry()
                multiprocess.MultiProcessCollector(registry)
                port = args.prometheus_port
                start_http_server(port, registry=registry)
                logger.info(
                    f"Prometheus metrics server started on port {port}, "
                    f"run_name={self._run_name}"
                )
        except ImportError:
            logger.error(
                "prometheus_client not installed. Run: pip install prometheus-client"
            )
            raise
        except OSError as e:
            logger.warning(f"Prometheus HTTP server not started: {e}")
            # Still mark as initialized — gauges can still be written to the
            # shared dir even if this process didn't start the server.

    def update(self, metrics: dict):
        """Set gauge values for all numeric metrics.

        Every gauge carries a ``run_name`` label so different training
        runs on the same or different Pods can be distinguished in Grafana.
        Prometheus will scrape these at its configured interval.
        """
        if not self._initialized:
            return
        from prometheus_client import Gauge

        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            safe_key = (
                key.replace("/", "_").replace("-", "_").replace("@", "_at_")
            )
            if safe_key not in self._gauges:
                self._gauges[safe_key] = Gauge(
                    safe_key, key,
                    self._label_keys,
                )
            self._gauges[safe_key].labels(*self._label_vals).set(value)


def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)

    if getattr(args, "use_prometheus", False):
        _PrometheusAdapter(args, start_server=primary)


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str):
    if args.use_wandb:
        wandb.log(metrics)

    if args.use_tensorboard:
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])

    if getattr(args, "use_prometheus", False):
        _PrometheusAdapter(args).update(metrics)
