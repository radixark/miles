from typing import Any

from miles.utils.ft.platform.notifiers.webhook_notifier import BaseWebhookNotifier

_SEVERITY_COLOR = {
    "critical": 0xE74C3C,
    "warning": 0xF1C40F,
    "info": 0x3498DB,
}


class DiscordWebhookNotifier(BaseWebhookNotifier):
    """Sends notifications via Discord webhook (embed)."""

    def _build_payload(self, title: str, content: str, severity: str) -> dict[str, Any]:
        color = _SEVERITY_COLOR.get(severity, 0x95A5A6)
        return {
            "embeds": [
                {
                    "title": f"[{severity}] {title}",
                    "description": content,
                    "color": color,
                }
            ],
        }
