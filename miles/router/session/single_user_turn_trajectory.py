import logging
import threading
import uuid
from typing import Any

from pydantic import BaseModel, Field

from miles.router.session.session_types import SessionRecord
from miles.utils.chat_template_utils import apply_chat_template
from miles.utils.chat_template_utils.token_seq_comparator import TokenSeqComparator

logger = logging.getLogger(__name__)

_TEMPLATE_RELEVANT_KEYS = ("role", "content", "reasoning_content", "tool_calls")


def _normalize_value(value: Any) -> Any:
    """Normalize falsy sentinels that produce identical Jinja2 output.

    None, "" and [] are all falsy in Jinja2 and render the same way,
    but client libraries may interchange them (e.g. content: null vs ""
    for tool-call-only responses, or tool_calls: null vs []).
    """
    if value is None or value == "" or value == []:
        return None
    return value


def _message_matches(stored: dict[str, Any], new: dict[str, Any]) -> bool:
    """Compare only the fields that affect chat-template tokenization.

    External client libraries (e.g. litellm) may inject extra keys like
    ``provider_specific_fields`` into messages.  These have no effect on
    the Jinja2 chat template output, so we only compare the keys that
    templates actually read: role, content, reasoning_content, tool_calls.
    """
    for key in _TEMPLATE_RELEVANT_KEYS:
        if _normalize_value(stored.get(key)) != _normalize_value(new.get(key)):
            return False
    return True


class SingleUserTurnTrajectory(BaseModel):
    """State for a single-user-turn trajectory.

    Tracks the full message history and accumulated token IDs for one session.
    The message sequence is: [system?, user, assistant, tool, assistant, tool, …].
    """

    messages: list[dict[str, Any]] = Field(default_factory=list)
    records: list[SessionRecord] = Field(default_factory=list)
    # Accumulated token IDs from the latest trajectory
    # (prompt_token_ids + completion_token_ids).
    # Sent as pretokenized_token_ids on the next turn so SGLang can skip
    # re-tokenizing the prefix.
    token_ids: list[int] = Field(default_factory=list)

    def append_session_record(self, record: SessionRecord):
        self.records.append(record)


class SingleUserTurnTrajectoryManager:
    """Trajectory manager for single-user-turn sessions.

    Assumes a conversation where no user message appears after the first
    assistant message.  The typical sequence is:
    [system?, user?, assistant, tool, assistant, tool, …]
    with optional system messages injected mid-conversation (e.g. retry
    prompts).

    Each new request's messages must be a strict append-only extension of
    the previous request's messages (only tool/system messages appended).
    This allows reusing pretokenized_token_ids across turns.
    """

    def __init__(self, args, tokenizer: Any, *, tito_tokenizer=None):
        self.sessions: dict[str, SingleUserTurnTrajectory] = {}
        self.args = args
        self.tokenizer = tokenizer
        self._lock = threading.RLock()
        self._comparator = TokenSeqComparator(tokenizer) if tokenizer else None
        self._trim_trailing_ids = (tito_tokenizer.get_trim_trailing_ids() or None) if tito_tokenizer else None

    def create_session(self) -> str:
        with self._lock:
            session_id = uuid.uuid4().hex
            self.sessions[session_id] = SingleUserTurnTrajectory()
            return session_id

    def get_session_records_by_id(self, session_id: str) -> list[SessionRecord] | None:
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                return None
            return session.records

    def compute_session_metadata(self, session_id: str) -> dict:
        if self._comparator is None:
            return {}
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None or not session.token_ids:
                return {}
            try:
                tools = session.records[-1].request.get("tools")
                expected_ids = apply_chat_template(
                    session.messages,
                    tokenizer=self.tokenizer,
                    tools=tools,
                    add_generation_prompt=False,
                    tokenize=True,
                )
                mismatches = self._comparator.compare_sequences(
                    expected_ids,
                    session.token_ids,
                    trim_trailing_ids=self._trim_trailing_ids,
                )
                return {"tito_session_mismatch": [m.to_dict() for m in mismatches]}
            except Exception:
                logger.exception("Failed to compute tito_session_mismatch for session %s", session_id)
                return {}

    def delete_session_by_id(self, session_id: str) -> bool | None:
        with self._lock:
            session = self.sessions.pop(session_id, None)
            if session is None:
                return None
            return True

    def append_session_record(self, session_id: str, record: SessionRecord) -> bool | None:
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                return None
            session.append_session_record(record)
            return True

    def try_prepare_pretokenized(
        self, session_id: str, request_messages: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Check if we can use pretokenized token input for this request.

        Returns a dict with pretokenized_token_ids and pretokenized_num_message
        if the request is eligible, or None if it's not.

        Eligibility requires:
        1. The session has prior turns (token_ids is non-empty).
        2. No user message appears after the first assistant message.
        3. The new messages are append-only relative to stored messages —
           only tool or system messages have been appended after the
           previous assistant message.
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                previews = [
                    f"[{i}] role={m.get('role')}, content={(m.get('content') or '')[:100]!r}"
                    for i, m in enumerate(request_messages)
                ]
                raise ValueError(
                    f"session not found: session_id={session_id}, "
                    f"num_messages={len(request_messages)}\n"
                    + "\n".join(previews)
                    + "\nThis usually means a stale agent environment from a previous "
                    "training run is still sending requests after the router restarted. "
                    "Ensure all agent containers are fully stopped before restarting training."
                )

            if not session.token_ids:
                # empty trajectory, no pretokenized input available
                return None

            self._assert_no_user_after_assistant(request_messages)
            self._assert_append_only(session.messages, request_messages)

            return {
                "pretokenized_token_ids": session.token_ids,
                "pretokenized_num_message": len(session.messages),
            }

    def update_pretokenized_state(
        self,
        session_id: str,
        request_messages: list[dict[str, Any]],
        assistant_message: dict[str, Any],
        prompt_token_ids: list[int],
        completion_token_ids: list[int],
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        """Update the session's pretokenized state after a successful response.

        1. **Validate prefix**: assert the previously stored token_ids are a
           prefix of ``prompt_token_ids + completion_token_ids``, confirming
           SGLang actually reused our pretokenized input.
        2. **Store**: save ``prompt_token_ids + completion_token_ids`` as the
           new token_ids for next turn's pretokenized reuse.

        Args:
            session_id: The session ID.
            request_messages: The full message list from the request.
            assistant_message: The assistant message from the response.
            prompt_token_ids: Prompt token IDs from the response.
            completion_token_ids: Output token IDs from the response.
            tools: Tool definitions from the request (OpenAI format).
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                return

            all_token_ids = prompt_token_ids + completion_token_ids
            session.messages = list(request_messages) + [assistant_message]

            # Validate that SGLang reused our pretokenized prefix.
            prev = session.token_ids
            if prev:
                assert all_token_ids[: len(prev)] == prev, (
                    f"pretokenized prefix mismatch: "
                    f"stored {len(prev)} tokens are not a prefix of "
                    f"prompt_token_ids + completion_token_ids "
                    f"({len(all_token_ids)} tokens)"
                )

            # Store actual response tokens for next turn's pretokenized reuse.
            session.token_ids = all_token_ids

    @staticmethod
    def _assert_no_user_after_assistant(messages: list[dict[str, Any]]) -> None:
        """Assert no user message appears after the first assistant message."""
        seen_assistant = False
        for i, msg in enumerate(messages):
            role = msg.get("role")
            if role == "assistant":
                seen_assistant = True
            elif role == "user" and seen_assistant:
                raise ValueError(
                    f"invalid message structure: user message at index {i} "
                    f"appears after the first assistant message"
                )

    @staticmethod
    def _assert_append_only(
        stored_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
    ) -> None:
        """Assert new_messages is append-only vs stored_messages.

        The stored prefix must match exactly, and any new messages appended
        after the stored prefix must have role 'tool' or 'system'.
        """
        if not stored_messages:
            return

        if len(new_messages) < len(stored_messages):
            raise ValueError(
                f"new messages ({len(new_messages)}) are fewer than " f"stored messages ({len(stored_messages)})"
            )

        for i, stored_msg in enumerate(stored_messages):
            if not _message_matches(stored_msg, new_messages[i]):
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

        ALLOWED_APPEND_ROLES = {"tool", "system"}
        for j, msg in enumerate(new_messages[len(stored_messages) :]):
            if msg.get("role") not in ALLOWED_APPEND_ROLES:
                raise ValueError(
                    f"appended message at index {len(stored_messages) + j} "
                    f"has role={msg.get('role')!r}, allowed={ALLOWED_APPEND_ROLES}"
                )
