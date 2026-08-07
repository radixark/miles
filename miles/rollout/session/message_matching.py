"""Request-scoped matching and authoritative replay history construction."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from miles.rollout.session.errors import SessionMessageMatcherError
from miles.utils.chat_template_utils.message_matcher_hub import SessionMessageMatcher


def _values_exactly_equal(stored: Any, replayed: Any) -> bool:
    if type(stored) is not type(replayed):
        return False
    if isinstance(stored, list):
        return len(stored) == len(replayed) and all(
            _values_exactly_equal(left, right) for left, right in zip(stored, replayed, strict=True)
        )
    if isinstance(stored, dict):
        return stored.keys() == replayed.keys() and all(
            _values_exactly_equal(stored[key], replayed[key]) for key in stored
        )
    return stored == replayed


@dataclass
class MessageMatchCache:
    """Evaluate each stored/replayed message pair at most once per request."""

    matcher: SessionMessageMatcher
    _results: dict[tuple[int, int], bool] = field(default_factory=dict)

    def matches(self, stored: dict[str, Any], replayed: dict[str, Any]) -> bool:
        key = (id(stored), id(replayed))
        if key not in self._results:
            try:
                result = self.matcher(stored, replayed)
            except Exception as exc:
                raise SessionMessageMatcherError("session message matcher raised an exception") from exc
            if type(result) is not bool:
                raise SessionMessageMatcherError(
                    f"session message matcher must return bool, got {type(result).__name__}"
                )
            self._results[key] = result
        return self._results[key]


@dataclass(frozen=True)
class AuthoritativeMessageHistory:
    effective_messages: list[dict[str, Any]]
    replayed_messages: list[dict[str, Any]] | None
    accepted_replay_indices: tuple[int, ...]


def build_authoritative_message_history(
    stored_messages: list[dict[str, Any]],
    replayed_messages: list[dict[str, Any]],
    *,
    reuse_prefix_len: int,
) -> AuthoritativeMessageHistory:
    """Use the reusable stored prefix and preserve the untouched replay suffix."""
    effective_messages = copy.deepcopy(stored_messages[:reuse_prefix_len] + replayed_messages[reuse_prefix_len:])
    accepted_replay_indices = tuple(
        index
        for index in range(reuse_prefix_len)
        if not _values_exactly_equal(stored_messages[index], replayed_messages[index])
    )
    replay_audit = copy.deepcopy(replayed_messages) if accepted_replay_indices else None
    return AuthoritativeMessageHistory(
        effective_messages=effective_messages,
        replayed_messages=replay_audit,
        accepted_replay_indices=accepted_replay_indices,
    )
