"""Shared hardware fault detection logic.

Used by both HighConfidenceHardwareDetector (detector chain) and
AlertChecker (recovery orchestrator) to avoid duplicating the same
metric queries and threshold constants.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import polars as pl

from miles.utils.ft.controller.detectors.gpu.checks import check_gpu_faults
from miles.utils.ft.models.fault import NodeFault
from miles.utils.ft.models.metric_names import NODE_FILESYSTEM_AVAIL_BYTES, NODE_NETWORK_UP
from miles.utils.ft.protocols.metrics import MetricQueryProtocol

logger = logging.getLogger(__name__)

DISK_AVAILABLE_THRESHOLD_BYTES: float = 1e9  # 1 GB


def check_all_hardware_faults(
    metric_store: MetricQueryProtocol,
) -> list[NodeFault]:
    return [
        *check_gpu_faults(metric_store),
        *_check_majority_nic_down(metric_store),
    ]


def _check_disk_fault(
    metric_store: MetricQueryProtocol,
    disk_available_threshold_bytes: float = DISK_AVAILABLE_THRESHOLD_BYTES,
) -> list[NodeFault]:
    df = metric_store.query_latest(NODE_FILESYSTEM_AVAIL_BYTES)
    if df is None or df.is_empty():
        return []

    bad = df.filter(pl.col("value") < disk_available_threshold_bytes)
    if bad.is_empty():
        return []

    return [
        NodeFault(
            node_id=row["node_id"],
            reason=f"disk space low on {row['node_id']} ({row['value']:.0f} bytes)",
        )
        for row in bad.iter_rows(named=True)
    ]


def check_nic_down_in_window(
    metric_store: MetricQueryProtocol,
    window: timedelta,
    threshold: int,
) -> list[NodeFault]:
    """Count NIC-down samples per node over *window*; fault nodes at or above *threshold*."""
    df = metric_store.query_range(NODE_NETWORK_UP, window=window)
    if df is None or df.is_empty():
        return []

    down_samples = df.filter(pl.col("value") == 0.0)
    if down_samples.is_empty():
        return []

    node_down_counts: dict[str, int] = {}
    for row in down_samples.iter_rows(named=True):
        node_id = row["node_id"]
        node_down_counts[node_id] = node_down_counts.get(node_id, 0) + 1

    return [
        NodeFault(
            node_id=node_id,
            reason=f"NIC down {count} times on {node_id} in {window}",
            ephemeral=True,
        )
        for node_id, count in sorted(node_down_counts.items())
        if count >= threshold
    ]


def _check_majority_nic_down(metric_store: MetricQueryProtocol) -> list[NodeFault]:
    df = metric_store.query_latest(NODE_NETWORK_UP)
    if df is None or df.is_empty():
        return []

    stats = (
        df.group_by("node_id")
        .agg(
            total_count=pl.len(),
            down_count=(pl.col("value") == 0.0).sum(),
        )
        .filter(pl.col("down_count") > pl.col("total_count") / 2)
    )
    return [
        NodeFault(
            node_id=row["node_id"],
            reason=f"majority NIC down on {row['node_id']} ({row['down_count']}/{row['total_count']})",
            ephemeral=True,
        )
        for row in stats.iter_rows(named=True)
    ]
