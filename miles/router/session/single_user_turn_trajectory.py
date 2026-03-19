import logging
import threading
import uuid
from typing import Any

from pydantic import BaseModel, Field, computed_field

from miles.router.session.session_types import SessionRecord
from miles.utils.chat_template_utils import apply_chat_template, assert_messages_append_only, message_matches
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer
from miles.utils.chat_template_utils.token_seq_comparator import TokenSeqComparator

logger = logging.getLogger(__name__)


class SingleUserTurnTrajectory(BaseModel):
    """State for a single-user-turn trajectory.

    Tracks the full message history and accumulated token IDs for one session.
    The message sequence is: [system?, user, assistant, tool, assistant, tool, …].

    Supports rollback to a previous assistant checkpoint: when the agent
    framework retries by sending a prefix of the stored messages,
    ``trajectory_token_ids`` is truncated to the matching checkpoint and
    the conversation continues from there.
    """

    messages: list[dict[str, Any]] = Field(default_factory=list)
    records: list[SessionRecord] = Field(default_factory=list)
    trajectory_token_ids: list[list[int]] = Field(default_factory=list)
    num_assistant: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def token_ids(self) -> list[int]:
        """Current token IDs — the latest assistant checkpoint."""
        return self.trajectory_token_ids[-1] if self.trajectory_token_ids else []

    def append_session_record(self, record: SessionRecord):
        self.records.append(record)

    def _detect_and_rollback(
        self,
        request_messages: list[dict[str, Any]],
    ) -> None:
        """Detect if *request_messages* requires a rollback and perform it.

        A rollback is triggered when *request_messages* diverges from
        ``self.messages`` before reaching the end of stored messages.
        The divergence point must fall after an assistant message (checkpoint
        boundary).  The session state is truncated to that checkpoint.
        """
        stored = self.messages
        if not stored or not self.trajectory_token_ids:
            return

        match_len = 0
        for i in range(min(len(request_messages), len(stored))):
            if message_matches(stored[i], request_messages[i]):
                match_len = i + 1
            else:
                break

        if match_len >= len(stored):
            return

        # Find the last assistant message within the matched prefix.
        rollback_msg_end = None
        checkpoint_index = -1
        assistant_count = 0
        for i in range(match_len):
            if stored[i].get("role") == "assistant":
                rollback_msg_end = i + 1
                checkpoint_index = assistant_count
                assistant_count += 1

        if checkpoint_index < 0:
            raise ValueError(
                f"rollback failed: no assistant message found in the first "
                f"{match_len} matched messages (stored has {len(stored)} messages, "
                f"request has {len(request_messages)} messages)"
            )

        logger.info(
            "Rolling back session: stored %d messages / %d checkpoints -> " "checkpoint %d (messages[:%d])",
            len(stored),
            self.num_assistant,
            checkpoint_index,
            rollback_msg_end,
        )

        self.messages = stored[:rollback_msg_end]
        self.trajectory_token_ids = self.trajectory_token_ids[: checkpoint_index + 1]
        self.records = self.records[: checkpoint_index + 1]
        self.num_assistant = checkpoint_index + 1


class SingleUserTurnTrajectoryManager:
    """Lightweight session manager for single-user-turn trajectories.

    Handles session CRUD, message-level validation (append-only, no user
    after assistant), and token ID read/store.  All tokenization computation
    is delegated to ``TITOTokenizer``.

    The typical message sequence is:
    ``[system?, user?, assistant, tool, assistant, tool, …]``
    with optional system messages injected mid-conversation.

    Supports rollback: if the agent sends messages that are a prefix of the
    stored messages (ending at an assistant boundary), the session state is
    rolled back to that checkpoint before proceeding.
    """

    def __init__(self, args, tokenizer: Any, *, tito_tokenizer: TITOTokenizer | None = None):
        self.sessions: dict[str, SingleUserTurnTrajectory] = {}
        self.args = args
        self.tokenizer = tokenizer
        self._lock = threading.RLock()
        self._comparator = TokenSeqComparator(tokenizer)
        self._tito_tokenizer = tito_tokenizer

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

    def get_session_token_ids(self, session_id: str) -> list[int]:
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                return []
            return session.token_ids

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
                trim_ids = self._tito_tokenizer.get_comparator_ignore_trailing_ids() if self._tito_tokenizer else None
                mismatches = self._comparator.compare_sequences(
                    expected_ids,
                    session.token_ids,
                    trim_trailing_ids=trim_ids,
                )
                result = {"tito_session_mismatch": [m.to_dict() for m in mismatches]}
                return result
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
        self,
        session_id: str,
        request_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """Compute a merged prompt via ``TITOTokenizer.merge_tokens`` and
        return it as ``input_ids`` for SGLang.

        Returns ``None`` on the first turn (no stored token_ids yet).

        If *request_messages* is a prefix of the stored messages (ending at
        an assistant boundary), the session state is rolled back to that
        checkpoint before the append-only validation proceeds.
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
                return None

            self._assert_no_user_after_assistant(request_messages)
            session._detect_and_rollback(request_messages)
            assert_messages_append_only(session.messages, request_messages)

            if self._tito_tokenizer is None:
                return None

            merged = self._tito_tokenizer.merge_tokens(
                old_messages=session.messages,
                new_messages=request_messages,
                pretokenized_token_ids=session.token_ids,
                tools=tools,
            )
            return {
                "input_ids": merged,
            }

    def update_pretokenized_state(
        self,
        session_id: str,
        request_messages: list[dict[str, Any]],
        assistant_message: dict[str, Any],
        prompt_token_ids: list[int],
        completion_token_ids: list[int],
    ) -> None:
        """Store raw token IDs after a successful response (no stripping).

        Appends a new checkpoint to ``trajectory_token_ids``.  Validates that
        the previously stored token_ids are a prefix of the new
        ``prompt_token_ids + completion_token_ids``, confirming SGLang
        actually reused our pretokenized input.
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                return

            all_token_ids = prompt_token_ids + completion_token_ids
            session.messages = list(request_messages) + [assistant_message]

            max_trim = self._tito_tokenizer.max_trim_tokens if self._tito_tokenizer else 0
            prev = session.token_ids
            if prev:
                check_len = len(prev) - max_trim
                if check_len > 0 and all_token_ids[:check_len] != prev[:check_len]:
                    first_mismatch = next(
                        (
                            i
                            for i, (a, b) in enumerate(zip(all_token_ids[:check_len], prev[:check_len], strict=True))
                            if a != b
                        ),
                        min(len(all_token_ids), check_len),
                    )
                    raise ValueError(
                        f"pretokenized prefix mismatch: "
                        f"stored {len(prev)} tokens (checking first {check_len}, "
                        f"allowing {max_trim} trailing) are not a prefix of "
                        f"prompt_token_ids + completion_token_ids "
                        f"({len(all_token_ids)} tokens), "
                        f"first mismatch at index {first_mismatch}, "
                        f"matched {first_mismatch}/{check_len} prefix tokens\n"
                        f"request_messages={request_messages}\n"
                        f"assistant_message={assistant_message}"
                    )

            session.trajectory_token_ids.append(all_token_ids)
            session.num_assistant += 1

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
