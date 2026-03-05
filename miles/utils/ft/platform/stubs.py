import logging
from uuid import uuid4

from miles.utils.ft.platform.protocols import JobStatus

logger = logging.getLogger(__name__)


class StubNodeManager:
    """Logs operations but does not call real K8s API."""

    async def mark_node_bad(self, node_id: str, reason: str) -> None:
        logger.info("stub_mark_node_bad node_id=%s reason=%s", node_id, reason)

    async def unmark_node_bad(self, node_id: str) -> None:
        logger.info("stub_unmark_node_bad node_id=%s", node_id)

    async def get_bad_nodes(self) -> list[str]:
        return []


class StubTrainingJob:
    """Logs operations but does not call real Ray Job API."""

    async def stop_training(self, timeout_seconds: int = 300) -> None:
        logger.info("stub_stop_training timeout_seconds=%d", timeout_seconds)

    async def submit_training(
        self, excluded_node_ids: list[str] | None = None,
    ) -> str:
        run_id = uuid4().hex[:8]
        logger.info("stub_submit_training run_id=%s excluded_node_ids=%s", run_id, excluded_node_ids)
        return run_id

    async def get_training_status(self) -> JobStatus:
        return JobStatus.RUNNING


class StubNotifier:
    """Logs notifications but does not send them anywhere."""

    async def send(self, title: str, content: str, severity: str) -> None:
        logger.info(
            "stub_send_notification title=%s severity=%s content=%s",
            title, severity, content,
        )

    async def aclose(self) -> None:
        pass
