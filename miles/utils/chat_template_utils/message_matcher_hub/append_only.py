"""Strict append-only history validation shared by TITO and the sessions."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from miles.utils.chat_template_utils.message_matcher_hub.strict import strict_message_matches
from miles.utils.chat_template_utils.message_matcher_hub.utils import _TEMPLATE_RELEVANT_KEYS, SessionMessageMatcher


def assert_messages_append_only_with_allowed_role(
    stored_messages: list[dict[str, Any]],
    new_messages: list[dict[str, Any]],
    allowed_append_roles: Collection[str],
    *,
    message_matcher: SessionMessageMatcher | None = None,
) -> None:
    """Assert *new_messages* is an append-only extension of *stored_messages*.

    The stored prefix must match pairwise under *message_matcher* (defaults
    to the strict template-relevant comparison), and any appended messages
    must have a role in *allowed_append_roles*.
    """
    if not stored_messages:
        return

    matcher = message_matcher if message_matcher is not None else strict_message_matches

    if len(new_messages) < len(stored_messages):
        raise ValueError(
            f"new messages ({len(new_messages)}) are fewer than stored messages ({len(stored_messages)})",
            new_messages,
            stored_messages,
        )

    for i, stored_msg in enumerate(stored_messages):
        if not matcher(stored_msg, new_messages[i]):
            diffs = {
                key: {"stored": repr(stored_msg.get(key))[:200], "new": repr(new_messages[i].get(key))[:200]}
                for key in _TEMPLATE_RELEVANT_KEYS
                if stored_msg.get(key) != new_messages[i].get(key)
            }
            raise ValueError(
                f"message mismatch at index {i} "
                f"(role: stored={stored_msg.get('role')}, new={new_messages[i].get('role')}). "
                f"Diffs: {diffs}"
            )

    for j, msg in enumerate(new_messages[len(stored_messages) :]):
        if msg.get("role") not in allowed_append_roles:
            raise ValueError(
                f"appended message at index {len(stored_messages) + j} "
                f"has role={msg.get('role')!r}, allowed={allowed_append_roles}"
            )
