import logging
from typing import Any

from miles.utils.ft.adapters.impl.notifiers.webhook_notifier import BaseWebhookNotifier

logger = logging.getLogger(__name__)

_SEVERITY_COLOR = {
    "critical": 0xE74C3C,
    "warning": 0xF1C40F,
    "info": 0x3498DB,
}


class DiscordWebhookNotifier(BaseWebhookNotifier):
    """Sends notifications via Discord webhook (embed)."""

    def _build_payload(self, title: str, content: str, severity: str) -> dict[str, Any]:
        color = _SEVERITY_COLOR.get(severity, 0x95A5A6)
        logger.debug("notifier: building discord payload title=%s, severity=%s", title, severity)
        return {
            "embeds": [
                {
                    "title": f"[{severity}] {title}",
                    "description": content,
                    "color": color,
                }
            ],
        }
