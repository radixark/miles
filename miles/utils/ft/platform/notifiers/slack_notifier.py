from typing import Any

from miles.utils.ft.platform.notifiers.webhook_notifier import BaseWebhookNotifier

_SEVERITY_EMOJI = {
    "critical": ":red_circle:",
    "warning": ":large_yellow_circle:",
    "info": ":large_blue_circle:",
}


class SlackWebhookNotifier(BaseWebhookNotifier):
    """Sends notifications via Slack Incoming Webhook (Block Kit)."""

    def _build_payload(self, title: str, content: str, severity: str) -> dict[str, Any]:
        emoji = _SEVERITY_EMOJI.get(severity, ":white_circle:")
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
