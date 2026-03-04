from datetime import datetime, timedelta, timezone

from miles.utils.ft.controller.detectors._metric_names import NODE_NIC_UP
from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.controller.mini_prometheus.protocol import MetricStoreProtocol
from miles.utils.ft.controller.mini_wandb import MiniWandb
from miles.utils.ft.models import ActionType, Decision

_DEFAULT_ALERT_WINDOW = timedelta(minutes=5)
_DEFAULT_ALERT_THRESHOLD = 2
_DEFAULT_SCRAPE_STEP = timedelta(seconds=10)


class NetworkAlertDetector(BaseFaultDetector):
    def __init__(
        self,
        alert_window: timedelta = _DEFAULT_ALERT_WINDOW,
        alert_threshold: int = _DEFAULT_ALERT_THRESHOLD,
        scrape_step: timedelta = _DEFAULT_SCRAPE_STEP,
    ) -> None:
        self._alert_window = alert_window
        self._alert_threshold = alert_threshold
        self._scrape_step = scrape_step

    def evaluate(
        self,
        metric_store: MetricStoreProtocol,
        mini_wandb: MiniWandb,
        rank_placement: dict[int, str],
    ) -> Decision:
        now = datetime.now(timezone.utc)
        start = now - self._alert_window

        df = metric_store.range_query(
            query=f"{NODE_NIC_UP} == 0",
            start=start,
            end=now,
            step=self._scrape_step,
        )

        if df.is_empty():
            return Decision(action=ActionType.NONE, reason="no NIC alerts in window")

        # Count down events per node_id
        node_down_counts: dict[str, int] = {}
        for row in df.iter_rows(named=True):
            node_id = row["node_id"]
            node_down_counts[node_id] = node_down_counts.get(node_id, 0) + 1

        bad_nodes: list[str] = []
        reasons: list[str] = []
        for node_id, count in sorted(node_down_counts.items()):
            if count >= self._alert_threshold:
                bad_nodes.append(node_id)
                reasons.append(f"NIC down {count} times on {node_id} in {self._alert_window}")

        if bad_nodes:
            return Decision(
                action=ActionType.MARK_BAD_AND_RESTART,
                bad_node_ids=bad_nodes,
                reason="; ".join(reasons),
            )

        return Decision(action=ActionType.NONE, reason="NIC alerts below threshold")
