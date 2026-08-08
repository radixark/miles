"""Role-and-content-only matcher: every other message field is ignored."""

from __future__ import annotations

from typing import Any

from miles.utils.chat_template_utils.message_matcher_hub.utils import _normalize_value


def role_content_only_message_matches(stored: dict[str, Any], replayed: dict[str, Any]) -> bool:
    """Compare only role and content using the strict matcher's empty-value rule.

    High-risk field projection: every other message field — including whole
    ``tool_calls`` — is deliberately ignored, so the stored prefix wins for
    anything a template might read from those fields.
    """
    return all(_normalize_value(stored.get(key)) == _normalize_value(replayed.get(key)) for key in ("role", "content"))
