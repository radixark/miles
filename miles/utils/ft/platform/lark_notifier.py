import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_INITIAL_BACKOFF_SECONDS = 1.0


class LarkWebhookNotifier:
    """Sends notifications via Lark custom bot webhook (interactive card)."""

    def __init__(self, webhook_url: str) -> None:
        self._webhook_url = webhook_url
        self._client = httpx.AsyncClient(timeout=10.0)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def send(self, title: str, content: str, severity: str) -> None:
        payload = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": f"[{severity}] {title}",
                    }
                },
                "elements": [{"tag": "markdown", "content": content}],
            },
        }

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._client.post(self._webhook_url, json=payload)
                response.raise_for_status()
                return
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt < _MAX_RETRIES - 1:
                    backoff = _INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                    logger.warning(
                        "lark_webhook_send_failed attempt=%d/%d url=%s, retrying in %.1fs",
                        attempt + 1, _MAX_RETRIES, self._webhook_url, backoff,
                        exc_info=True,
                    )
                    await asyncio.sleep(backoff)

        logger.error(
            "lark_webhook_send_failed_all_retries url=%s",
            self._webhook_url,
            exc_info=last_error,
        )
        if last_error is not None:
            raise last_error
