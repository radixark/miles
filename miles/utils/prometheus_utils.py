"""Backwards-compatible shim. Prefer ``miles.utils.tracking.prometheus_utils``."""
from miles.utils.tracking.prometheus_utils import (  # noqa: F401
    _PrometheusCollector,
    get_prometheus,
    init_prometheus,
)
