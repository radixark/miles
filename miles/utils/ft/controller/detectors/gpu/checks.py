"""GPU-specific hardware fault detection: GPU loss and critical XID errors."""

from __future__ import annotations

import logging

import polars as pl

from miles.utils.ft.models.fault import NodeFault
from miles.utils.ft.models.metric_names import GPU_AVAILABLE, XID_CODE_RECENT
from miles.utils.ft.protocols.metrics import MetricQueryProtocol

logger = logging.getLogger(__name__)

CRITICAL_XID_CODES: frozenset[int] = frozenset({48, 62, 64, 79})


def check_gpu_faults(
    metric_store: MetricQueryProtocol,
    critical_xid_codes: frozenset[int] = CRITICAL_XID_CODES,
) -> list[NodeFault]:
    return [
        *_check_gpu_lost(metric_store),
        *_check_critical_xid(metric_store, critical_xid_codes=critical_xid_codes),
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
