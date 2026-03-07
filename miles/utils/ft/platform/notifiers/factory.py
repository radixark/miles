"""Notifier factory: builds the appropriate WebhookNotifier subclass
from explicit configuration parameters.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from miles.utils.ft.platform.stubs import StubNotifier

if TYPE_CHECKING:
    from miles.utils.ft.platform.notifiers.webhook_notifier import WebhookNotifier

logger = logging.getLogger(__name__)


def build_notifier(
    platform: str,
    notify_webhook_url: str = "",
    notify_platform: str = "",
) -> WebhookNotifier | StubNotifier | None:
    webhook_url = notify_webhook_url.strip()
    notify_platform = notify_platform.strip().lower() or "lark"

    if webhook_url:
        cls = _get_notifier_class(notify_platform)
        return cls(webhook_url=webhook_url)

    if platform == "stub":
        return StubNotifier()

    logger.warning(
        "No notifier configured for platform=%s "
        "(--notify-webhook-url not set). "
        "Recovery alerts will not be delivered.",
        platform,
    )
    return None


def _get_notifier_class(notify_platform: str) -> type[WebhookNotifier]:
    from miles.utils.ft.platform.notifiers.discord_notifier import DiscordWebhookNotifier
    from miles.utils.ft.platform.notifiers.lark_notifier import LarkWebhookNotifier
    from miles.utils.ft.platform.notifiers.slack_notifier import SlackWebhookNotifier

    registry: dict[str, type[WebhookNotifier]] = {
        "lark": LarkWebhookNotifier,
        "slack": SlackWebhookNotifier,
        "discord": DiscordWebhookNotifier,
    }
    cls = registry.get(notify_platform)
    if cls is None:
        raise ValueError(
            f"Unknown notify platform: {notify_platform!r}. "
            f"Supported: {sorted(registry)}"
        )
    return cls
