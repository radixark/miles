import abc
import logging
from typing import Any

import httpx

from miles.utils.ft.adapters.types import NotifierProtocol
from miles.utils.ft.utils.retry import retry_async_or_raise

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_INITIAL_BACKOFF_SECONDS = 1.0


class BaseWebhookNotifier(NotifierProtocol):
    """Base class for webhook-based notifiers with retry and exponential backoff."""

    def __init__(
        self,
        webhook_url: str,
        timeout_seconds: float = 10.0,
        max_retries: int = _MAX_RETRIES,
        initial_backoff_seconds: float = _INITIAL_BACKOFF_SECONDS,
    ) -> None:
        self._webhook_url = webhook_url
        self._client = httpx.AsyncClient(timeout=timeout_seconds)
        self._max_retries = max_retries
        self._initial_backoff_seconds = initial_backoff_seconds

    async def aclose(self) -> None:
        await self._client.aclose()

    async def send(self, title: str, content: str, severity: str) -> None:
        logger.info(f"webhook send: {title=} {content=} {severity=}")
        payload = self._build_payload(title=title, content=content, severity=severity)

        async def _post() -> None:
            response = await self._client.post(self._webhook_url, json=payload)
            response.raise_for_status()

        await retry_async_or_raise(
            func=_post,
            description=f"webhook_send({self._webhook_url})",
            max_retries=self._max_retries,
            backoff_base=self._initial_backoff_seconds,
        )

    @abc.abstractmethod
    def _build_payload(self, title: str, content: str, severity: str) -> dict[str, Any]: ...
