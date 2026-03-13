import logging

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.agents.types import MetricSample

logger = logging.getLogger(__name__)


class StubCollector(BaseCollector):
    def _collect_sync(self) -> list[MetricSample]:
        logger.debug("collector: stub collect returning empty list")
        return []
