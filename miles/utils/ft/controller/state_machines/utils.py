from __future__ import annotations

import logging

from miles.utils.ft.adapters.types import NotifierProtocol
from miles.utils.ft.utils.retry import retry_async

logger = logging.getLogger(__name__)

_NOTIFY_MAX_RETRIES: int = 3
_NOTIFY_BACKOFF_BASE: float = 0.5
_NOTIFY_MAX_BACKOFF: float = 2.0


async def safe_notify(
    notifier: NotifierProtocol | None,
    title: str,
    content: str,
    severity: str = "critical",
) -> None:
    if notifier is None:
        logger.debug("safe_notify: notifier is None, skipping title=%s", title)
        return

    logger.debug("safe_notify: sending title=%s, severity=%s", title, severity)
    result = await retry_async(
        func=lambda: notifier.send(title=title, content=content, severity=severity),
        description=f"safe_notify({title})",
        max_retries=_NOTIFY_MAX_RETRIES,
        backoff_base=_NOTIFY_BACKOFF_BASE,
        max_backoff=_NOTIFY_MAX_BACKOFF,
    )
    if not result.ok:
        logger.error(
            "safe_notify_failed title=%s after %d retries",
            title,
            _NOTIFY_MAX_RETRIES,
            exc_info=True,
        )
