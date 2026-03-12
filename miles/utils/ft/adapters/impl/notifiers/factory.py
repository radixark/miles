"""Notifier factory: builds the appropriate BaseWebhookNotifier subclass
from explicit configuration parameters.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from miles.utils.ft.adapters.stubs import StubNotifier

if TYPE_CHECKING:
    from miles.utils.ft.adapters.impl.notifiers.webhook_notifier import BaseWebhookNotifier

logger = logging.getLogger(__name__)


def build_notifier(
    platform: str,
    notify_webhook_url: str = "",
    notify_platform: str = "",
    notify_timeout_seconds: float = 10.0,
    notify_max_retries: int = 3,
    notify_initial_backoff_seconds: float = 1.0,
) -> BaseWebhookNotifier | StubNotifier | None:
    webhook_url = notify_webhook_url.strip()
    notify_platform = notify_platform.strip().lower() or "lark"

    if webhook_url:
        cls = _get_notifier_class(notify_platform)
        return cls(
            webhook_url=webhook_url,
            timeout_seconds=notify_timeout_seconds,
            max_retries=notify_max_retries,
            initial_backoff_seconds=notify_initial_backoff_seconds,
        )

    if platform == "stub":
        return StubNotifier()

    logger.warning(
        "No notifier configured for platform=%s "
        "(--notify-webhook-url not set). "
        "Recovery alerts will not be delivered.",
        platform,
    )
    return None


def _get_notifier_class(notify_platform: str) -> type[BaseWebhookNotifier]:
    from miles.utils.ft.adapters.impl.notifiers.discord_notifier import DiscordWebhookNotifier
    from miles.utils.ft.adapters.impl.notifiers.lark_notifier import LarkWebhookNotifier
    from miles.utils.ft.adapters.impl.notifiers.slack_notifier import SlackWebhookNotifier

    registry: dict[str, type[BaseWebhookNotifier]] = {
        "lark": LarkWebhookNotifier,
        "slack": SlackWebhookNotifier,
        "discord": DiscordWebhookNotifier,
    }
    cls = registry.get(notify_platform)
    if cls is None:
        raise ValueError(f"Unknown notify platform: {notify_platform!r}. " f"Supported: {sorted(registry)}")
    return cls
