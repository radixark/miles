"""Resolve a ``--session-message-matcher`` selector to a matcher callable."""

from __future__ import annotations

from miles.utils.chat_template_utils.message_matcher_hub.loose_tool_call import loose_tool_call_message_matches
from miles.utils.chat_template_utils.message_matcher_hub.role_content_only import role_content_only_message_matches
from miles.utils.chat_template_utils.message_matcher_hub.strict import strict_message_matches
from miles.utils.chat_template_utils.message_matcher_hub.utils import SessionMessageMatcher

_BUILTIN_MESSAGE_MATCHERS: dict[str, SessionMessageMatcher] = {
    "strict": strict_message_matches,
    "loose_tool_call": loose_tool_call_message_matches,
    "role_content_only": role_content_only_message_matches,
}


def resolve_session_message_matcher(selector: str) -> SessionMessageMatcher:
    """Resolve an exact built-in alias or a synchronous dotted import path."""
    if selector in _BUILTIN_MESSAGE_MATCHERS:
        return _BUILTIN_MESSAGE_MATCHERS[selector]
    aliases = ", ".join(_BUILTIN_MESSAGE_MATCHERS)
    if not isinstance(selector, str) or not selector or "." not in selector:
        raise ValueError(
            f"invalid --session-message-matcher {selector!r}; use one of {aliases}, "
            f"or a dotted import path such as package.module.matcher"
        )
    try:
        from miles.utils.misc import load_function

        return load_function(selector, sync_required=True)
    except Exception as exc:
        raise ValueError(
            f"failed to resolve --session-message-matcher {selector!r}; use one of {aliases}, "
            f"or a dotted import path such as package.module.matcher: {exc}"
        ) from exc
