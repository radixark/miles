"""Shared hardware fault detection logic.

Used by both HighConfidenceHardwareDetector (detector chain) and
AlertChecker (recovery orchestrator) to avoid duplicating the same
metric queries and threshold constants.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import polars as pl

from miles.utils.ft.protocols.metrics import MetricQueryProtocol
from miles.utils.ft.models.metric_names import GPU_AVAILABLE, NODE_FILESYSTEM_AVAIL_BYTES, NODE_NETWORK_UP, XID_CODE_RECENT
from miles.utils.ft.models.fault import NodeFault

logger = logging.getLogger(__name__)

CRITICAL_XID_CODES: frozenset[int] = frozenset({48, 62, 64, 79})
DISK_AVAILABLE_THRESHOLD_BYTES: float = 1e9  # 1 GB


def check_all_hardware_faults(
    metric_store: MetricQueryProtocol,
    critical_xid_codes: frozenset[int] = CRITICAL_XID_CODES,
    disk_available_threshold_bytes: float = DISK_AVAILABLE_THRESHOLD_BYTES,
) -> list[NodeFault]:
    return [
        *_check_gpu_lost(metric_store),
        *_check_critical_xid(metric_store, critical_xid_codes=critical_xid_codes),
        *_check_disk_fault(metric_store, disk_available_threshold_bytes=disk_available_threshold_bytes),
        *_check_majority_nic_down(metric_store),
    ]


def _check_gpu_lost(metric_store: MetricQueryProtocol) -> list[NodeFault]:
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


def _check_critical_xid(
    metric_store: MetricQueryProtocol,
    critical_xid_codes: frozenset[int] = CRITICAL_XID_CODES,
) -> list[NodeFault]:
    df = metric_store.query_latest(XID_CODE_RECENT)
    if df is None or df.is_empty():
        return []

    return [
        f for row in df.iter_rows(named=True)
        if (f := _parse_xid_row(row, critical_xid_codes)) is not None
    ]


def _parse_xid_row(
    row: dict[str, object],
    critical_xid_codes: frozenset[int],
) -> NodeFault | None:
    try:
        xid_code = int(row.get("xid", -1))  # type: ignore[arg-type]
        node_id = row.get("node_id")
    except (ValueError, TypeError):
        logger.warning("_check_critical_xid: unparseable row %s", row, exc_info=True)
        return None

    if node_id is None:
        return None

    if xid_code in critical_xid_codes:
        return NodeFault(node_id=node_id, reason=f"critical XID {xid_code} on {node_id}")
    return None


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
        )
        for row in stats.iter_rows(named=True)
    ]
