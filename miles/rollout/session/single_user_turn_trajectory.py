import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from miles.rollout.session.session_errors import MessageValidationError, SessionNotFoundError, TokenizationError
from miles.rollout.session.session_types import SessionRecord
from miles.utils.chat_template_utils import apply_chat_template, assert_messages_append_only, message_matches
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer

logger = logging.getLogger(__name__)


# TODO: hardcoded to 1 for now; if multi-step rollback is actually needed,
#  raise this limit or make it configurable and remove the restriction.
MAX_ASSISTANT_ROLLBACK_STEPS = 1


def _assert_no_user_after_assistant(messages: list[dict[str, Any]]) -> None:
    """Assert no user message appears after the first assistant message."""
    seen_assistant = False
    for i, msg in enumerate(messages):
        role = msg.get("role")
        if role == "assistant":
            seen_assistant = True
        elif role == "user" and seen_assistant:
            raise MessageValidationError(
                f"invalid message structure: user message at index {i} " f"appears after the first assistant message"
            )


@dataclass
class SingleUserTurnTrajectory:
    """State for a single-user-turn trajectory.

    Tracks the full message history and accumulated token IDs for one session.
    The typical message sequence is: [system?, user, assistant, tool, assistant, tool, …],
    but the agent may retry from an earlier point (e.g. re-running a tool call),
    in which case the session is rolled back at most one assistant step.

    Concurrency contract: all mutating methods must be called under ``self.lock``.
    """

    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)
    closing: bool = field(default=False, repr=False, compare=False)
    messages: list[dict[str, Any]] = field(default_factory=list)
    records: list[SessionRecord] = field(default_factory=list)
    trajectory_token_ids: list[list[int]] = field(default_factory=list)
    num_assistant: int = 0

    @property
    def token_ids(self) -> list[int]:
        """Current token IDs — the latest assistant checkpoint."""
        return self.trajectory_token_ids[-1] if self.trajectory_token_ids else []

    def append_record(self, record: SessionRecord) -> None:
        self.records.append(record)

    def prepare_pretokenized(
        self,
        request_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        tito_tokenizer: TITOTokenizer,
    ) -> dict[str, Any] | None:
        """Validate messages, rollback if needed, and compute merged input_ids.

        Returns ``None`` on the first turn (no stored token_ids yet).
        Must be called under ``self.lock``.
        """
        if not self.token_ids:
            return None

        # 1. Reject multi-turn (user after assistant) — single-user-turn only.
        _assert_no_user_after_assistant(request_messages)
        # 2. Detect agent retries and roll back (at most one assistant step).
        self._try_detect_and_rollback_to_assistant_checkpoint(request_messages)
        # 3. Confirm the (possibly rolled-back) stored messages are a prefix of request.
        try:
            assert_messages_append_only(self.messages, request_messages)
        except ValueError as e:
            raise MessageValidationError(str(e)) from e

        merged = tito_tokenizer.merge_tokens(
            old_messages=self.messages,
            new_messages=request_messages,
            pretokenized_token_ids=self.token_ids,
            tools=tools,
        )
        return {"input_ids": merged}

    def update_pretokenized_state(
        self,
        request_messages: list[dict[str, Any]],
        assistant_message: dict[str, Any],
        prompt_token_ids: list[int],
        completion_token_ids: list[int],
        max_trim_tokens: int,
    ) -> None:
        """Store raw token IDs after a successful response.

        Appends ``prompt_token_ids + completion_token_ids`` as a new checkpoint.
        Validates that the previously stored token_ids are a prefix of the new
        checkpoint (tolerating up to ``max_trim_tokens`` trailing differences).
        Must be called under ``self.lock``.
        """
        all_token_ids = prompt_token_ids + completion_token_ids

        prev = self.token_ids
        if prev:
            check_len = len(prev) - max_trim_tokens
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
                    f"allowing {max_trim_tokens} trailing) are not a prefix of "
                    f"prompt_token_ids + completion_token_ids "
                    f"({len(all_token_ids)} tokens), "
                    f"first mismatch at index {first_mismatch}, "
                    f"matched {first_mismatch}/{check_len} prefix tokens\n"
                    f"request_messages={request_messages}\n"
                    f"assistant_message={assistant_message}"
                )

        self.messages = list(request_messages) + [assistant_message]
        self.trajectory_token_ids.append(all_token_ids)
        self.num_assistant += 1

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

        Only a single-step rollback is allowed (controlled by
        ``MAX_ASSISTANT_ROLLBACK_STEPS``).  Discarding exactly one assistant
        message means the agent is retrying from the preceding checkpoint —
        the request shares the stored prefix up to that assistant and then
        continues with whatever the agent chooses (same or different tool
        result, additional messages, etc.).  Any request that would need to
        discard more than one assistant (i.e. jump back across multiple
        turns) is rejected with ``MessageValidationError`` and no state is
        modified.

        Example — agent retries after the first tool call::

            stored:  [sys, user, assistant₁, tool₁, assistant₂]
                      ───────────────────── ▲
                      checkpoint 0 (assistant₁)   checkpoint 1 (assistant₂)

            request: [sys, user, assistant₁, tool₁_different, ...]
                                             ↑ diverges here (index 3)

            match_len = 3  (sys, user, assistant₁ all match)
            Last assistant in matched prefix → assistant₁ (checkpoint 0)
            discard_count = 2 - 1 = 1  (≤ MAX_ASSISTANT_ROLLBACK_STEPS)

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

        discard_count = self.num_assistant - (checkpoint_index + 1)
        if discard_count > MAX_ASSISTANT_ROLLBACK_STEPS:
            raise MessageValidationError(
                f"rollback failed: discard_count={discard_count} exceeds "
                f"max_assistant_rollback_steps={MAX_ASSISTANT_ROLLBACK_STEPS} "
                f"(stored has {len(stored)} messages, "
                f"request has {len(request_messages)} messages)"
            )

        logger.info(
            "Rolling back session: stored %d messages / %d checkpoints -> "
            "checkpoint %d (messages[:%d]), discarding %d assistant(s)",
            len(stored),
            self.num_assistant,
            checkpoint_index,
            rollback_msg_end,
            discard_count,
        )

        self.messages = stored[:rollback_msg_end]
        self.trajectory_token_ids = self.trajectory_token_ids[: checkpoint_index + 1]
        self.records = self.records[: checkpoint_index + 1]
        self.num_assistant = checkpoint_index + 1


class SessionRegistry:
    """Session ID -> trajectory mapping with shared tokenizer resources.

    Pure CRUD plus read-only computation (compute_session_mismatch).
    Does NOT mutate session state - all mutations are methods on
    SingleUserTurnTrajectory, called by the route handler under session.lock.
    """

    def __init__(self, args, tokenizer: Any, *, tito_tokenizer: TITOTokenizer):
        self.sessions: dict[str, SingleUserTurnTrajectory] = {}
        self.args = args
        self.tokenizer = tokenizer
        self.tito_tokenizer = tito_tokenizer
        self.comparator = tito_tokenizer.create_comparator()

    def create_session(self) -> str:
        session_id = uuid.uuid4().hex
        self.sessions[session_id] = SingleUserTurnTrajectory()
        return session_id

    def get_session(self, session_id: str) -> SingleUserTurnTrajectory:
        session = self.sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")
        return session

    def remove_session(self, session_id: str) -> None:
        if self.sessions.pop(session_id, None) is None:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")

    def compute_session_mismatch(self, session: SingleUserTurnTrajectory) -> list[dict] | None:
        """Compare accumulated token IDs against canonical chat template output.

        Read-only: does not mutate session state.
        """
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
            raise TokenizationError(f"failed to compute tito_session_mismatch: {e}") from e
