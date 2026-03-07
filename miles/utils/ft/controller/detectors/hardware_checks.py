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
    """Count NIC up→down transitions per node over *window*; fault nodes at or above *threshold*.

    Counts state transitions (up→down), not raw down-sample counts.
    This decouples the threshold from the scrape interval.
    """
    df = metric_store.query_range(NODE_NETWORK_UP, window=window)
    if df is None or df.is_empty():
        return []

    # Sort chronologically so shift(1) gives the temporally previous sample
    df = df.sort("timestamp")

    # Detect up→down transitions per (node_id, device).
    # A transition = previous sample was up (>0) and current sample is down (==0).
    # This counts flap events, NOT the number of samples where value==0.
    transitions = (
        df.with_columns(
            prev_value=pl.col("value").shift(1).over("node_id", "device")
        )
        .filter(
            (pl.col("prev_value") > 0) & (pl.col("value") == 0.0)
        )
    )

    if transitions.is_empty():
        return []

    node_counts = (
        transitions.group_by("node_id")
        .agg(count=pl.len())
    )

    return [
        NodeFault(
            node_id=row["node_id"],
            reason=f"NIC went down {row['count']} time(s) on {row['node_id']} in {window}",
            ephemeral=True,
        )
        for row in node_counts.iter_rows(named=True)
        if row["count"] >= threshold
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
            ephemeral=False,
        )
        for row in stats.iter_rows(named=True)
    ]
