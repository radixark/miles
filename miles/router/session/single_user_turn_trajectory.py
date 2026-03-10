import logging
import threading
import uuid
from typing import Any

from pydantic import BaseModel, Field

from miles.router.session.session_types import SessionRecord
from miles.utils.chat_template_utils import apply_chat_template

logger = logging.getLogger(__name__)


class SingleUserTurnTrajectory(BaseModel):
    """State for a single-user-turn trajectory.

    Tracks the full message history and accumulated token IDs for one session.
    The message sequence is: [system?, user, assistant, tool, assistant, tool, …].
    """

    messages: list[dict[str, Any]] = Field(default_factory=list)
    records: list[SessionRecord] = Field(default_factory=list)
    # Accumulated token IDs from the latest response
    # (prompt_token_ids + completion_token_ids).
    # Sent as pretokenized_token_ids on the next turn so SGLang can skip
    # re-tokenizing the prefix.
    token_ids: list[int] = Field(default_factory=list)

    def append_session_record(self, record: SessionRecord):
        self.records.append(record)


class SingleUserTurnTrajectoryManager:
    """Trajectory manager for single-user-turn sessions.

    Assumes a conversation with exactly one user message, optionally preceded
    by system messages, followed by multi-turn tool-call steps
    (assistant → tool → assistant → tool → …).  Additional system messages
    may be injected mid-conversation (e.g. retry prompts).

    The chat template rendering after the last user message must satisfy the
    append-only invariant: each new request's messages are a strict extension
    of the previous request's messages, with only tool/system messages
    appended. This allows reusing pretokenized_token_ids across turns.
    """

    def __init__(self, args, tokenizer: Any):
        self.sessions: dict[str, SingleUserTurnTrajectory] = {}
        self.args = args
        self.tokenizer = tokenizer
        self._lock = threading.RLock()

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
        2. The prompt contains exactly one user message.
        3. The new messages are append-only relative to stored messages —
           only tool or system messages have been appended after the
           previous assistant message.
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise ValueError("session not found, register it first")

            if not session.token_ids:
                # emptry trajectory, no pretokenized input available
                return None

            if not self._validate_message_structure(request_messages):
                raise ValueError("invalid message structure: must contain exactly one user message")

            if not self._is_append_only(session.messages, request_messages):
                # new messages are not append-only, includes new tool messages
                raise ValueError(
                    "new messages are not append-only: only tool and system "
                    "messages may be appended after the stored prefix"
                )

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

        Steps:
        1. **Validate prefix**: assert the previously stored token_ids are a
           prefix of ``prompt_token_ids + completion_token_ids``, confirming
           SGLang actually reused our pretokenized input.
        2. **Sanity check (text-only)**: validate prefix relationships on
           decoded token text. We intentionally avoid reconstructing assistant
           text from structured messages/tool_calls because parser output can
           be lossy for assistant messages.
        3. **Store**: save ``prompt_token_ids + completion_token_ids`` as the
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

            # Step 1: Validate that SGLang reused our pretokenized prefix.
            prev = session.token_ids
            if prev:
                assert all_token_ids[: len(prev)] == prev, (
                    f"pretokenized prefix mismatch: "
                    f"stored {len(prev)} tokens are not a prefix of "
                    f"prompt_token_ids + completion_token_ids "
                    f"({len(all_token_ids)} tokens)"
                )

            # Step 2: Sanity check — decoded tokens must match chat template output.
            decoded_text = self.tokenizer.decode(all_token_ids)
            applied_text = apply_chat_template(
                session.messages,
                tokenizer=self.tokenizer,
                tools=tools,
                add_generation_prompt=False,
                tokenize=False,
            )
            assert decoded_text == applied_text, f"decoded_text != applied_text: {decoded_text} != {applied_text}"

            # Step 3: Store actual response tokens for next turn's pretokenized reuse.
            session.token_ids = all_token_ids

    @staticmethod
    def _validate_message_structure(messages: list[dict[str, Any]]) -> bool:
        """Check that messages contain exactly one user message."""
        user_count = sum(1 for msg in messages if msg.get("role") == "user")
        return user_count == 1

    @staticmethod
    def _is_append_only(
        stored_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
    ) -> bool:
        """Check new_messages is append-only vs stored_messages.

        The stored prefix must match exactly, and any new messages appended
        after the stored prefix must have role 'tool' or 'system'.
        """
        if not stored_messages:
            return True

        if len(new_messages) < len(stored_messages):
            return False

        # Check that the stored prefix is unchanged.
        for i, stored_msg in enumerate(stored_messages):
            if new_messages[i] != stored_msg:
                return False

        ALLOWED_APPEND_ROLES = {"tool", "system"}
        for msg in new_messages[len(stored_messages) :]:
            if msg.get("role") not in ALLOWED_APPEND_ROLES:
                return False

        return True
