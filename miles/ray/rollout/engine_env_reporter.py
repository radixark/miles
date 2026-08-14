import asyncio
import logging
import math
import time
from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import EngineEnvReportEvent
from miles.utils.env_report.redaction import redact_server_info

logger = logging.getLogger(__name__)
SERVER_INFO_TIMEOUT_SECONDS = 10.0
RETRY_INTERVAL_SECONDS = 60.0


class EngineEnvReporter:
    def __init__(self, *, interval_seconds: float) -> None:
        self._interval_seconds = interval_seconds
        self._next_due: float | None = None

    async def report_if_due(self, *, cell_id: str, server_url: str, api_client: SGLangApiClient) -> None:
        now = time.monotonic()
        if self._next_due is not None and now < self._next_due:
            return

        reported = await self._report(cell_id=cell_id, server_url=server_url, api_client=api_client)
        self._next_due = now + self._delay_after(reported=reported)

    def _delay_after(self, *, reported: bool) -> float:
        if not reported:
            return RETRY_INTERVAL_SECONDS
        return self._interval_seconds if self._interval_seconds > 0 else math.inf

    async def _report(self, *, cell_id: str, server_url: str, api_client: SGLangApiClient) -> bool:
        try:
            server_info = await asyncio.wait_for(api_client.get_server_info(), timeout=SERVER_INFO_TIMEOUT_SECONDS)
            if is_event_logger_initialized():
                get_event_logger().log(
                    EngineEnvReportEvent,
                    {"cell_id": cell_id, "server_url": server_url, "server_info": redact_server_info(server_info)},
                    print_log=False,
                )
            return True
        except Exception:
            logger.warning("Failed to record the engine env of cell %s", cell_id, exc_info=True)
            return False
