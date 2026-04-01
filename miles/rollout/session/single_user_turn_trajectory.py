import logging
import threading
import uuid
from typing import Any

from pydantic import BaseModel, Field, computed_field

from miles.rollout.session.session_errors import MessageValidationError, SessionNotFoundError, TokenizationError
from miles.rollout.session.session_types import SessionRecord
from miles.utils.chat_template_utils import apply_chat_template, assert_messages_append_only, message_matches
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer

logger = logging.getLogger(__name__)


class SingleUserTurnTrajectory(BaseModel):
    """State for a single-user-turn trajectory.

    Tracks the full message history and accumulated token IDs for one session.
    The typical message sequence is: [system?, user, assistant, tool, assistant, tool, …],
    but the agent may retry from an earlier point (e.g. re-running a tool call),
    in which case the session is rolled back to the last matching assistant
    checkpoint and re-extended from there.
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

    def _try_detect_and_rollback_to_assistant_checkpoint(
        self,
        request_messages: list[dict[str, Any]],
    ) -> None:
        """Detect if *request_messages* diverges from stored history and roll back.

        In agentic workflows the agent may retry from an earlier point — for
        example, re-running a tool call with different arguments.  When that
        happens the new request shares a common prefix with the stored messages
        but diverges before the end.  This method truncates session state back
        to the last assistant checkpoint within the matching prefix.

        Example — agent retries after the first tool call::

            stored:  [sys, user, assistant₁, tool₁, assistant₂]
                      ───────────────────── ▲
                      checkpoint 0 (assistant₁)   checkpoint 1 (assistant₂)

            request: [sys, user, assistant₁, tool₁_different, ...]
                                             ↑ diverges here (index 3)

            match_len = 3  (sys, user, assistant₁ all match)
            Last assistant in matched prefix → assistant₁ (checkpoint 0)

            After rollback:
              messages           = [sys, user, assistant₁]
              trajectory_token_ids = [checkpoint_0_ids]
              records              = [record_0]
              num_assistant        = 1

        No rollback occurs when:
        - The stored history is empty.
        - *request_messages* is a strict extension of stored messages
          (``match_len >= len(stored)``).
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
            raise MessageValidationError(
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
    """

    def __init__(self, args, tokenizer: Any, *, tito_tokenizer: TITOTokenizer):
        self.sessions: dict[str, SingleUserTurnTrajectory] = {}
        self.args = args
        self.tokenizer = tokenizer
        self._lock = threading.RLock()
        self.tito_tokenizer = tito_tokenizer
        self.comparator = tito_tokenizer.create_comparator()

    def create_session(self) -> str:
        with self._lock:
            session_id = uuid.uuid4().hex
            self.sessions[session_id] = SingleUserTurnTrajectory()
            return session_id

    def get_session_records_by_id(self, session_id: str) -> list[SessionRecord]:
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")
            return session.records

    def get_session_token_ids(self, session_id: str) -> list[int]:
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")
            return session.token_ids

    def compute_session_mismatch(self, session_id: str) -> list[dict] | None:
        """Compare accumulated token IDs against canonical chat template output.

        Returns a list of mismatch dicts from ``TokenSeqComparator.compare_sequences``,
        each containing ``{position, expected_token, actual_token, context}``,
        or ``None`` if the session has no token IDs yet.
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")
            if not session.token_ids:
                return None
            try:
                tools = session.records[-1].request.get("tools") if session.records else None
                expected_ids = apply_chat_template(
                    session.messages,
                    tokenizer=self.tokenizer,
                    tools=tools,
                    add_generation_prompt=False,
                    tokenize=True,
                )
                mismatches = self.comparator.compare_sequences(expected_ids, session.token_ids)
                return [m.to_dict() for m in mismatches]
            except Exception as e:
                raise TokenizationError(
                    f"failed to compute tito_session_mismatch for session {session_id}: {e}"
                ) from e

    def delete_session_by_id(self, session_id: str) -> bool:
        with self._lock:
            session = self.sessions.pop(session_id, None)
            if session is None:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")
            return True

    def append_session_record(self, session_id: str, record: SessionRecord) -> bool:
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")
            session.append_session_record(record)
            return True

    def prepare_pretokenized(
        self,
        session_id: str,
        request_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """Compute a merged prompt via ``TITOTokenizer.merge_tokens`` and
        return it as ``input_ids`` for SGLang.

        Returns ``None`` on the first turn (no stored token_ids yet).
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                previews = [
                    f"[{i}] role={m.get('role')}, content={(m.get('content') or '')[:100]!r}"
                    for i, m in enumerate(request_messages)
                ]
                raise SessionNotFoundError(
                    f"session not found: session_id={session_id}, "
                    f"num_messages={len(request_messages)}\n"
                    + "\n".join(previews)
                    + "\nThis usually means a stale agent environment from a previous "
                    "training run is still sending requests after the router restarted. "
                    "Ensure all agent containers are fully stopped before restarting training."
                )

            if not session.token_ids:
                return None

            # Validate and reconcile request_messages against stored session state:
            # 1. Reject multi-turn (user after assistant) — single-user-turn only.
            self._assert_no_user_after_assistant(request_messages)
            # 2. Detect agent retries and roll back to the last matching checkpoint.
            session._try_detect_and_rollback_to_assistant_checkpoint(request_messages)
            # 3. Confirm the (possibly rolled-back) stored messages are a prefix of request.
            try:
                assert_messages_append_only(session.messages, request_messages)
            except ValueError as e:
                raise MessageValidationError(str(e)) from e

            merged = self.tito_tokenizer.merge_tokens(
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
        """Store raw token IDs after a successful response.

        Appends ``prompt_token_ids + completion_token_ids`` as-is (no
        stripping or modification) as a new checkpoint in
        ``trajectory_token_ids``.  Validates that the previously stored
        token_ids are a prefix of the new checkpoint, tolerating up to
        ``max_trim_tokens`` trailing tokens that may differ due to
        chat-template boundary re-tokenization.  This confirms SGLang
        actually reused our pretokenized input.
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(f"update_pretokenized_state: session not found: session_id={session_id}")

            all_token_ids = prompt_token_ids + completion_token_ids
            session.messages = list(request_messages) + [assistant_message]

            max_trim = self.tito_tokenizer.max_trim_tokens
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
                    raise TokenizationError(
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
                raise MessageValidationError(
                    f"invalid message structure: user message at index {i} "
                    f"appears after the first assistant message"
                )
