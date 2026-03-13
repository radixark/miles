import logging
from typing import Any

from miles.utils.ft.adapters.impl.notifiers.webhook_notifier import BaseWebhookNotifier

logger = logging.getLogger(__name__)

_SEVERITY_EMOJI = {
    "critical": ":red_circle:",
    "warning": ":large_yellow_circle:",
    "info": ":large_blue_circle:",
}


class SlackWebhookNotifier(BaseWebhookNotifier):
    """Sends notifications via Slack Incoming Webhook (Block Kit)."""

    def _build_payload(self, title: str, content: str, severity: str) -> dict[str, Any]:
        emoji = _SEVERITY_EMOJI.get(severity, ":white_circle:")
        logger.debug("notifier: building slack payload title=%s, severity=%s", title, severity)
        return {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{emoji} [{severity}] {title}",
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": content,
                    },
                },
            ],
        }
