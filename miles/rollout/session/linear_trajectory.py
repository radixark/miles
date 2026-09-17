import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from miles.rollout.session.config import SessionServerConfig
from miles.rollout.session.errors import MessageValidationError, SessionNotFoundError, TokenizationError
from miles.rollout.session.request_args import PreparedChatRequest, prepare_chat_request
from miles.rollout.session.types import SessionRecord
from miles.utils.chat_template_utils.message_matcher_hub import (
    SessionMessageMatcher,
    assert_messages_append_only_with_allowed_role,
    strict_message_matches,
)
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer

logger = logging.getLogger(__name__)


# TODO: hardcoded to 1 for now; if multi-step rollback is actually needed,
#  raise this limit or make it configurable and remove the restriction.
MAX_ASSISTANT_ROLLBACK_STEPS = 1


def assert_pretokenized_prefix(
    prev: list[int],
    all_token_ids: list[int],
    *,
    max_trim_tokens: int,
    request_messages: list[dict[str, Any]],
    assistant_message: dict[str, Any],
) -> None:
    """Stored token_ids must be a prefix of the new checkpoint, tolerating up
    to *max_trim_tokens* trailing differences. Pure token-level check, shared
    verbatim by the v1 checkpoint update and the v2 commit."""
    if not prev:
        return
    check_len = len(prev) - max_trim_tokens
    if check_len > 0 and all_token_ids[:check_len] != prev[:check_len]:
        first_mismatch = next(
            (i for i, (a, b) in enumerate(zip(all_token_ids[:check_len], prev[:check_len], strict=True)) if a != b),
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


@dataclass
class LinearTrajectory:
    """State for a linear trajectory.

    Tracks the full message history and accumulated token IDs for one session.

    Session-generated assistant responses create checkpoints; client-injected assistant messages remain prompt history.

    Rollback uses ``generated_checkpoint_message_ends`` instead of inferring checkpoints from message roles.

    The typical message sequence is: [system?, user, assistant, tool, assistant, tool, …],
    but the agent may retry from an earlier point (e.g. re-running a tool call),
    in which case the session is rolled back at most one assistant step.

    ``prepare_token_ids_and_request_args`` resolves arguments and renders from
    the selected checkpoint before applying rollback.

    Concurrency contract: all mutating methods must be called under ``self.lock``.
    """

    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)
    closing: bool = field(default=False, repr=False, compare=False)
    messages: list[dict[str, Any]] = field(default_factory=list)
    records: list[SessionRecord] = field(default_factory=list)
    trajectory_token_ids: list[list[int]] = field(default_factory=list)
    generated_checkpoint_message_ends: list[int] = field(default_factory=list)
    num_assistant: int = 0

    @property
    def token_ids(self) -> list[int]:
        """Current token IDs — the latest assistant checkpoint."""
        return self.trajectory_token_ids[-1] if self.trajectory_token_ids else []

    def append_record(self, record: SessionRecord) -> None:
        self.records.append(record)

    def prepare_token_ids_and_request_args(
        self,
        client_args: dict[str, Any],
        *,
        config: SessionServerConfig,
        tito_tokenizer: TITOTokenizer,
        message_matcher: SessionMessageMatcher | None = None,
    ) -> PreparedChatRequest:
        """Return the prepared request with prompt token IDs in `body["input_ids"]`.

        Select the checkpoint that the request continues, resolve arguments,
        and render its prompt before discarding any history. A validation or
        rendering error leaves session state unchanged.

        The caller must hold `self.lock`.
        """
        matcher = message_matcher if message_matcher is not None else strict_message_matches
        request_messages = client_args.get("messages", [])
        checkpoint_index = self._find_rollback_checkpoint(request_messages, matcher)
        prepared = prepare_chat_request(client_args, tito_tokenizer, config=config, turn_args=None)
        prepared.body["input_ids"] = self._render_token_ids(
            request_messages,
            checkpoint_index=checkpoint_index,
            template_args=prepared.template_args,
            tito_tokenizer=tito_tokenizer,
            message_matcher=matcher,
        )
        self._rollback_to_checkpoint(checkpoint_index)
        return prepared

    def _render_token_ids(
        self,
        request_messages: list[dict[str, Any]],
        *,
        checkpoint_index: int,
        template_args: dict[str, Any],
        tito_tokenizer: TITOTokenizer,
        message_matcher: SessionMessageMatcher | None = None,
    ) -> list[int]:
        """Render from the selected checkpoint without modifying session state.

        Validate the appended roles and reuse that checkpoint's stored messages
        and token prefix. An empty checkpoint renders the request from scratch.
        Must be called under ``self.lock``.
        """
        matcher = message_matcher if message_matcher is not None else strict_message_matches

        token_ids = self.trajectory_token_ids[checkpoint_index] if checkpoint_index >= 0 else []
        if not token_ids:
            return tito_tokenizer.apply_chat_template(
                request_messages,
                add_generation_prompt=True,
                tokenize=True,
                template_args=template_args,
            )

        message_end = self.generated_checkpoint_message_ends[checkpoint_index]
        stored_messages = self.messages[:message_end]
        try:
            assert_messages_append_only_with_allowed_role(
                stored_messages, request_messages, tito_tokenizer.allowed_append_roles, message_matcher=matcher
            )
        except ValueError as e:
            raise MessageValidationError(
                f"{e}; the selected TITO fixed template does not support appending this role"
            ) from e

        effective_messages = stored_messages + request_messages[len(stored_messages) :]
        return tito_tokenizer.merge_tokens(
            old_messages=stored_messages,
            new_messages=effective_messages,
            pretokenized_token_ids=token_ids,
            template_args=template_args,
        )

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
        assert_pretokenized_prefix(
            self.token_ids,
            all_token_ids,
            max_trim_tokens=max_trim_tokens,
            request_messages=request_messages,
            assistant_message=assistant_message,
        )

        # Commit the same effective history the tokens were built from (stored
        # spellings for the reused prefix, the replay only for the new tail):
        # committing the raw replay would let an accepted-but-reserialized
        # prefix rewrite stored history, so the next canonical replay could
        # no longer match its own session.
        self.messages = self.messages + request_messages[len(self.messages) :] + [assistant_message]
        self.trajectory_token_ids.append(all_token_ids)
        self.generated_checkpoint_message_ends.append(len(request_messages) + 1)
        self.num_assistant = len(self.generated_checkpoint_message_ends)

    def _find_rollback_checkpoint(
        self,
        request_messages: list[dict[str, Any]],
        message_matcher: SessionMessageMatcher,
    ) -> int:
        """Find the last generated checkpoint in the matching prefix, without mutation.

        Return the current tip for an append-only request, or -1 when the
        request starts from scratch. Client-injected assistant messages do not
        create checkpoints. Reject rollback beyond ``MAX_ASSISTANT_ROLLBACK_STEPS``.
        """
        stored = self.messages
        if not stored or not self.trajectory_token_ids:
            return -1

        match_len = 0
        for i in range(min(len(request_messages), len(stored))):
            if message_matcher(stored[i], request_messages[i]):
                match_len = i + 1
            else:
                break

        if match_len >= len(stored):
            return self.num_assistant - 1

        checkpoint_index = -1
        for i in reversed(range(len(self.generated_checkpoint_message_ends))):
            if self.generated_checkpoint_message_ends[i] <= match_len:
                checkpoint_index = i
                break

        discard_count = self.num_assistant - (checkpoint_index + 1)
        if discard_count > MAX_ASSISTANT_ROLLBACK_STEPS:
            raise MessageValidationError(
                f"rollback failed: discard_count={discard_count} exceeds "
                f"max_assistant_rollback_steps={MAX_ASSISTANT_ROLLBACK_STEPS} "
                f"(stored has {len(stored)} messages, "
                f"request has {len(request_messages)} messages)"
            )

        return checkpoint_index

    def _rollback_to_checkpoint(self, checkpoint_index: int) -> None:
        """Discard history after the selected checkpoint once preparation succeeds."""
        discard_count = self.num_assistant - (checkpoint_index + 1)
        if discard_count == 0:
            return
        rollback_msg_end = self.generated_checkpoint_message_ends[checkpoint_index] if checkpoint_index >= 0 else 0
        logger.info(
            "Rolling back session: stored %d messages / %d checkpoints -> "
            "checkpoint %d (messages[:%d]), discarding %d generated checkpoint(s)",
            len(self.messages),
            self.num_assistant,
            checkpoint_index,
            rollback_msg_end,
            discard_count,
        )

        self.messages = self.messages[:rollback_msg_end]
        self.trajectory_token_ids = self.trajectory_token_ids[: checkpoint_index + 1]
        self.records = self.records[: checkpoint_index + 1]
        self.generated_checkpoint_message_ends = self.generated_checkpoint_message_ends[: checkpoint_index + 1]
        self.num_assistant = len(self.generated_checkpoint_message_ends)


class SessionRegistry:
    """Session ID -> trajectory mapping with shared tokenizer resources.

    Pure CRUD plus read-only computation (compute_session_mismatch).
    Does NOT mutate session state - all mutations are methods on
    LinearTrajectory; called by the route handler under session.lock.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        tito_tokenizer: TITOTokenizer,
        message_matcher: SessionMessageMatcher | None = None,
    ):
        self.sessions: dict[str, LinearTrajectory] = {}
        self.tokenizer = tokenizer
        self.tito_tokenizer = tito_tokenizer
        self.comparator = tito_tokenizer.create_comparator()
        self.message_matcher: SessionMessageMatcher = (
            message_matcher if message_matcher is not None else strict_message_matches
        )

    def create_session(self) -> str:
        session_id = uuid.uuid4().hex
        self.sessions[session_id] = LinearTrajectory()
        return session_id

    def get_session(self, session_id: str) -> LinearTrajectory:
        session = self.sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")
        return session

    def remove_session(self, session_id: str) -> None:
        if self.sessions.pop(session_id, None) is None:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")

    def compute_session_mismatch(self, session: LinearTrajectory) -> list[dict] | None:
        """Compare accumulated token IDs against canonical chat template output.

        Read-only: does not mutate session state.
        """
        if not session.token_ids:
            return None
        try:
            tools = session.records[-1].request.get("tools") if session.records else None
            expected_ids = self.tito_tokenizer.apply_chat_template(
                session.messages,
                add_generation_prompt=False,
                tokenize=True,
                template_args=self.tito_tokenizer.default_template_args(tools),
            )
            mismatches = self.comparator.compare_sequences(expected_ids, session.token_ids)
            return [m.to_dict() for m in mismatches]
        except Exception as e:
            raise TokenizationError(f"failed to compute tito_session_mismatch: {e}") from e
