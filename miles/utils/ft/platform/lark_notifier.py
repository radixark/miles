import logging

import httpx

logger = logging.getLogger(__name__)


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

        try:
            response = await self._client.post(self._webhook_url, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(
                "lark_webhook_send_failed url=%s error=%s",
                self._webhook_url,
                exc,
            )
