from miles.utils.ft.controller.detectors._metric_names import (
    NODE_DISK_AVAILABLE_BYTES,
    NODE_GPU_AVAILABLE,
    NODE_NIC_UP,
    NODE_XID_CODE_RECENT,
)
from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.controller.mini_prometheus.protocol import MetricStoreProtocol
from miles.utils.ft.controller.mini_wandb import MiniWandb
from miles.utils.ft.models import ActionType, Decision

_CRITICAL_XID_CODES: frozenset[int] = frozenset({48, 62, 64, 79})
_DISK_AVAILABLE_THRESHOLD_BYTES: float = 1e9  # 1 GB


class HighConfidenceHardwareDetector(BaseFaultDetector):
    def __init__(
        self,
        critical_xid_codes: frozenset[int] = _CRITICAL_XID_CODES,
        disk_available_threshold_bytes: float = _DISK_AVAILABLE_THRESHOLD_BYTES,
    ) -> None:
        self._critical_xid_codes = critical_xid_codes
        self._disk_available_threshold_bytes = disk_available_threshold_bytes

    def evaluate(
        self,
        metric_store: MetricStoreProtocol,
        mini_wandb: MiniWandb,
        rank_placement: dict[int, str],
    ) -> Decision:
        bad_nodes: set[str] = set()
        reasons: list[str] = []

        self._check_gpu_lost(metric_store, bad_nodes, reasons)
        self._check_critical_xid(metric_store, bad_nodes, reasons)
        self._check_disk_fault(metric_store, bad_nodes, reasons)
        self._check_majority_nic_down(metric_store, bad_nodes, reasons)

        if bad_nodes:
            return Decision(
                action=ActionType.MARK_BAD_AND_RESTART,
                bad_node_ids=sorted(bad_nodes),
                reason="; ".join(reasons),
            )

        return Decision(action=ActionType.NONE, reason="no high-confidence hardware faults")

    def _check_gpu_lost(
        self,
        metric_store: MetricStoreProtocol,
        bad_nodes: set[str],
        reasons: list[str],
    ) -> None:
        df = metric_store.instant_query(f"{NODE_GPU_AVAILABLE} == 0")
        if df.is_empty():
            return

        for node_id in df["node_id"].unique().to_list():
            bad_nodes.add(node_id)
            reasons.append(f"GPU unavailable on {node_id}")

    def _check_critical_xid(
        self,
        metric_store: MetricStoreProtocol,
        bad_nodes: set[str],
        reasons: list[str],
    ) -> None:
        df = metric_store.instant_query(NODE_XID_CODE_RECENT)
        if df.is_empty():
            return

        for row in df.iter_rows(named=True):
            xid_code = int(row.get("xid", -1))
            if xid_code in self._critical_xid_codes:
                node_id = row["node_id"]
                bad_nodes.add(node_id)
                reasons.append(f"critical XID {xid_code} on {node_id}")

    def _check_disk_fault(
        self,
        metric_store: MetricStoreProtocol,
        bad_nodes: set[str],
        reasons: list[str],
    ) -> None:
        df = metric_store.instant_query(
            f"{NODE_DISK_AVAILABLE_BYTES} < {self._disk_available_threshold_bytes}"
        )
        if df.is_empty():
            return

        for row in df.iter_rows(named=True):
            node_id = row["node_id"]
            bad_nodes.add(node_id)
            reasons.append(f"disk space low on {node_id} ({row['value']:.0f} bytes)")

    def _check_majority_nic_down(
        self,
        metric_store: MetricStoreProtocol,
        bad_nodes: set[str],
        reasons: list[str],
    ) -> None:
        df = metric_store.instant_query(NODE_NIC_UP)
        if df.is_empty():
            return

        node_stats: dict[str, tuple[int, int]] = {}  # node_id -> (down_count, total_count)

        for row in df.iter_rows(named=True):
            node_id = row["node_id"]
            down_count, total_count = node_stats.get(node_id, (0, 0))
            total_count += 1
            if row["value"] == 0.0:
                down_count += 1
            node_stats[node_id] = (down_count, total_count)

        for node_id, (down_count, total_count) in node_stats.items():
            if total_count > 0 and down_count >= total_count / 2:
                bad_nodes.add(node_id)
                reasons.append(f"majority NIC down on {node_id} ({down_count}/{total_count})")
