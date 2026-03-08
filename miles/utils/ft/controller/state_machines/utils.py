from __future__ import annotations

import logging

from miles.utils.ft.protocols.platform import NotifierProtocol

logger = logging.getLogger(__name__)


async def safe_notify(
    notifier: NotifierProtocol | None,
    title: str,
    content: str,
    severity: str = "critical",
) -> None:
    if notifier is None:
        return
    try:
        await notifier.send(title=title, content=content, severity=severity)
    except Exception:
        logger.error("safe_notify_failed title=%s", title, exc_info=True)
