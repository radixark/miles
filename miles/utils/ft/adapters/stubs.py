import logging
from uuid import uuid4

from miles.utils.ft.adapters.types import (
    STOP_TRAINING_TIMEOUT_SECONDS,
    AgentMetadataProvider,
    JobStatus,
    MainJobProtocol,
    NodeManagerProtocol,
    NotifierProtocol,
)

logger = logging.getLogger(__name__)


class NullMetadataProvider(AgentMetadataProvider):
    """Returns empty metadata dict."""

    def get_metadata(self) -> dict[str, str]:
        return {}


class StubNodeManager(NodeManagerProtocol):
    """Logs operations but does not call real K8s API."""

    async def mark_node_bad(self, node_id: str, reason: str, node_metadata: dict[str, str] | None = None) -> None:
        logger.info("stub_mark_node_bad node_id=%s reason=%s", node_id, reason)

    async def unmark_node_bad(self, node_id: str) -> None:
        logger.info("stub_unmark_node_bad node_id=%s", node_id)

    async def get_bad_nodes(self) -> list[str]:
        return []


class StubMainJob(MainJobProtocol):
    """Logs operations but does not call real Ray Job API."""

    async def stop_job(self, timeout_seconds: int = STOP_TRAINING_TIMEOUT_SECONDS) -> None:
        logger.info("stub_stop_job timeout_seconds=%d", timeout_seconds)

    async def submit_job(self) -> str:
        run_id = uuid4().hex[:8]
        logger.info("stub_submit_job run_id=%s", run_id)
        return run_id

    async def get_job_status(self) -> JobStatus:
        return JobStatus.RUNNING


class StubNotifier(NotifierProtocol):
    """Logs notifications but does not send them anywhere."""

    async def send(self, title: str, content: str, severity: str) -> None:
        logger.info(
            "stub_send_notification title=%s severity=%s content=%s",
            title,
            severity,
            content,
        )

    async def aclose(self) -> None:
        pass
