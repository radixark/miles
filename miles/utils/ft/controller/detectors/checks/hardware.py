"""Shared hardware fault detection logic.

Used by GpuFaultDetector, NicMajorityDownDetector, DiskSpaceLowDetector,
and NetworkAlertDetector for metric queries and threshold constants.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import polars as pl

from miles.utils.ft.utils.metric_names import NODE_FILESYSTEM_AVAIL_BYTES, NODE_NETWORK_UP
from miles.utils.ft.controller.types import TimeSeriesQueryProtocol, NodeFault

logger = logging.getLogger(__name__)

DISK_AVAILABLE_THRESHOLD_BYTES: float = 1e9  # 1 GB


def check_disk_fault(
    metric_store: TimeSeriesQueryProtocol,
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
    metric_store: TimeSeriesQueryProtocol,
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
    transitions = df.with_columns(prev_value=pl.col("value").shift(1).over("node_id", "device")).filter(
        (pl.col("prev_value") > 0) & (pl.col("value") == 0.0)
    )

    if transitions.is_empty():
        return []

    node_counts = transitions.group_by("node_id").agg(count=pl.len())

    return [
        NodeFault(
            node_id=row["node_id"],
            reason=f"NIC went down {row['count']} time(s) on {row['node_id']} in {window}",
        )
        for row in node_counts.iter_rows(named=True)
        if row["count"] >= threshold
    ]


def check_nic_persistent_down(
    metric_store: TimeSeriesQueryProtocol,
    window: timedelta,
) -> list[NodeFault]:
    """Detect NICs that crashed permanently: went down and never recovered.

    A permanent NIC crash produces only a single down event (no flapping),
    so it cannot be caught by `check_nic_down_in_window` which counts
    up→down transitions.  This function catches NICs whose latest sample
    within the window is down (value==0) and that had at least one prior
    up sample (value>0) — confirming a transition from working to broken.
    """
    df = metric_store.query_range(NODE_NETWORK_UP, window=window)
    if df is None or df.is_empty():
        return []

    df = df.sort("timestamp")

    per_device = df.group_by("node_id", "device").agg(
        first_value=pl.col("value").first(),
        last_value=pl.col("value").last(),
        had_up=(pl.col("value") > 0).any(),
        sample_count=pl.len(),
    )

    persistent_down = per_device.filter(
        (pl.col("last_value") == 0.0)
        & (pl.col("had_up"))
        & (pl.col("sample_count") >= 2)
    )

    if persistent_down.is_empty():
        return []

    node_faults: dict[str, NodeFault] = {}
    for row in persistent_down.iter_rows(named=True):
        nid = row["node_id"]
        if nid not in node_faults:
            node_faults[nid] = NodeFault(
                node_id=nid,
                reason=f"NIC {row['device']} persistently down on {nid}",
            )
    return list(node_faults.values())


def check_majority_nic_down(metric_store: TimeSeriesQueryProtocol) -> list[NodeFault]:
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
