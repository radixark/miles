import logging
from typing import Any

from miles.utils.ft.adapters.impl.notifiers.webhook_notifier import BaseWebhookNotifier

logger = logging.getLogger(__name__)


class LarkWebhookNotifier(BaseWebhookNotifier):
    """Sends notifications via Lark custom bot webhook (interactive card)."""

    def _build_payload(self, title: str, content: str, severity: str) -> dict[str, Any]:
        logger.debug("notifier: building lark payload title=%s, severity=%s", title, severity)
        return {
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
