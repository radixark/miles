"""GPU-specific hardware fault detection: GPU loss and non-auto-recoverable XID errors."""

from __future__ import annotations

import logging

import polars as pl

from miles.utils.ft.controller.metrics.metric_names import GPU_AVAILABLE, XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL
from miles.utils.ft.controller.types import TimeSeriesQueryProtocol, NodeFault

logger = logging.getLogger(__name__)


def check_gpu_faults(
    metric_store: TimeSeriesQueryProtocol,
) -> list[NodeFault]:
    return [
        *_check_gpu_lost(metric_store),
        *_check_non_auto_recoverable_xid(metric_store),
    ]


def _check_gpu_lost(metric_store: TimeSeriesQueryProtocol) -> list[NodeFault]:
    df = metric_store.query_latest(GPU_AVAILABLE)
    if df is None or df.is_empty():
        return []

    bad = df.filter(pl.col("value") == 0.0)
    if bad.is_empty():
        return []

    return [
        NodeFault(node_id=node_id, reason=f"GPU unavailable on {node_id}")
        for node_id in bad["node_id"].unique().to_list()
    ]


def _check_non_auto_recoverable_xid(
    metric_store: TimeSeriesQueryProtocol,
) -> list[NodeFault]:
    df = metric_store.query_latest(XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL)
    if df is None or df.is_empty():
        return []

    bad = df.filter(pl.col("value") > 0.0)
    if bad.is_empty():
        return []

    return [
        NodeFault(
            node_id=node_id,
            reason=f"non-auto-recoverable XID detected on {node_id}",
        )
        for node_id in bad["node_id"].unique().to_list()
    ]
