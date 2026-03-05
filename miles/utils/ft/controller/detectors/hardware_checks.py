"""Shared hardware fault detection logic.

Used by both HighConfidenceHardwareDetector (detector chain) and
AlertChecker (recovery orchestrator) to avoid duplicating the same
metric queries and threshold constants.
"""

from __future__ import annotations

import logging

import polars as pl

from miles.utils.ft.protocols.metrics import MetricQueryProtocol
from miles.utils.ft.metric_names import GPU_AVAILABLE, NODE_FILESYSTEM_AVAIL_BYTES, NODE_NETWORK_UP, XID_CODE_RECENT
from miles.utils.ft.models import NodeFault

logger = logging.getLogger(__name__)

CRITICAL_XID_CODES: frozenset[int] = frozenset({48, 62, 64, 79})
DISK_AVAILABLE_THRESHOLD_BYTES: float = 1e9  # 1 GB


def check_gpu_lost(metric_store: MetricQueryProtocol) -> list[NodeFault]:
    df = metric_store.query_latest(GPU_AVAILABLE)
    if df.is_empty():
        return []

    bad = df.filter(pl.col("value") == 0.0)
    if bad.is_empty():
        return []

    return [
        NodeFault(node_id=node_id, reason=f"GPU unavailable on {node_id}")
        for node_id in bad["node_id"].unique().to_list()
    ]


def check_critical_xid(
    metric_store: MetricQueryProtocol,
    critical_xid_codes: frozenset[int] = CRITICAL_XID_CODES,
) -> list[NodeFault]:
    df = metric_store.query_latest(XID_CODE_RECENT)
    if df.is_empty():
        return []

    faults: list[NodeFault] = []
    for row in df.iter_rows(named=True):
        try:
            xid_code = int(row.get("xid", -1))
            node_id = row.get("node_id")
        except (ValueError, TypeError):
            logger.warning("check_critical_xid: unparseable row %s", row, exc_info=True)
            continue

        if node_id is None:
            continue

        if xid_code in critical_xid_codes:
            faults.append(NodeFault(node_id=node_id, reason=f"critical XID {xid_code} on {node_id}"))
    return faults


def check_disk_fault(
    metric_store: MetricQueryProtocol,
    disk_available_threshold_bytes: float = DISK_AVAILABLE_THRESHOLD_BYTES,
) -> list[NodeFault]:
    df = metric_store.query_latest(NODE_FILESYSTEM_AVAIL_BYTES)
    if df.is_empty():
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


def check_majority_nic_down(metric_store: MetricQueryProtocol) -> list[NodeFault]:
    df = metric_store.query_latest(NODE_NETWORK_UP)
    if df.is_empty():
        return []

    node_stats: dict[str, tuple[int, int]] = {}
    for row in df.iter_rows(named=True):
        node_id = row["node_id"]
        down_count, total_count = node_stats.get(node_id, (0, 0))
        total_count += 1
        if row["value"] == 0.0:
            down_count += 1
        node_stats[node_id] = (down_count, total_count)

    return [
        NodeFault(node_id=node_id, reason=f"majority NIC down on {node_id} ({down_count}/{total_count})")
        for node_id, (down_count, total_count) in node_stats.items()
        if total_count > 0 and down_count > total_count / 2
    ]


def check_all_hardware_faults(
    metric_store: MetricQueryProtocol,
    critical_xid_codes: frozenset[int] = CRITICAL_XID_CODES,
    disk_available_threshold_bytes: float = DISK_AVAILABLE_THRESHOLD_BYTES,
) -> list[NodeFault]:
    return [
        *check_gpu_lost(metric_store),
        *check_critical_xid(metric_store, critical_xid_codes=critical_xid_codes),
        *check_disk_fault(metric_store, disk_available_threshold_bytes=disk_available_threshold_bytes),
        *check_majority_nic_down(metric_store),
    ]
