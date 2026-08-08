"""Session message matching policies and selector resolution.

Owns every message-level equivalence policy the session server can select
via ``--session-message-matcher``, plus the strict append-only validation
that shares ``_TEMPLATE_RELEVANT_KEYS`` with the matchers.  Template
loading and rendering stay in ``template.py``; this package must depend
only on the standard library (``load_function`` is imported lazily inside
the resolver so plain matcher use never pulls in Ray via
``miles.utils.misc``).

Layout: ``utils`` holds the shared type alias, constants and value
normalization; each matcher, the selector resolver, and the append-only
validation live in their own modules.
"""

from miles.utils.chat_template_utils.message_matcher_hub.append_only import (
    assert_messages_append_only_with_allowed_role,
)
from miles.utils.chat_template_utils.message_matcher_hub.loose_tool_call import loose_tool_call_message_matches
from miles.utils.chat_template_utils.message_matcher_hub.resolver import resolve_session_message_matcher
from miles.utils.chat_template_utils.message_matcher_hub.role_content_only import role_content_only_message_matches
from miles.utils.chat_template_utils.message_matcher_hub.strict import strict_message_matches
from miles.utils.chat_template_utils.message_matcher_hub.utils import SessionMessageMatcher

__all__ = [
    "SessionMessageMatcher",
    "assert_messages_append_only_with_allowed_role",
    "loose_tool_call_message_matches",
    "resolve_session_message_matcher",
    "role_content_only_message_matches",
    "strict_message_matches",
]
