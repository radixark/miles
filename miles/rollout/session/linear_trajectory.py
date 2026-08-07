import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from miles.rollout.session.errors import MessageValidationError, SessionNotFoundError, TokenizationError
from miles.rollout.session.message_matching import MessageMatchCache, build_authoritative_message_history
from miles.rollout.session.types import SessionRecord
from miles.utils.chat_template_utils.message_matcher_hub import (
    SessionMessageMatcher,
    assert_messages_append_only_with_allowed_role,
    message_matches,
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


@dataclass(frozen=True)
class PreparedLinearRequest:
    """Pure v1 request plan, ready to apply after TITO succeeds."""

    effective_messages: list[dict[str, Any]]
    replayed_messages: list[dict[str, Any]] | None
    accepted_replay_indices: tuple[int, ...]
    common_match_len: int
    reuse_checkpoint_index: int
    reuse_prefix_len: int
    prompt_token_ids: list[int]


@dataclass
class LinearTrajectory:
    """State for a linear trajectory.

    Tracks the full message history and accumulated token IDs for one session.

    Session-generated assistant responses create checkpoints; client-injected assistant messages remain prompt history.

    Rollback uses ``generated_checkpoint_message_ends`` instead of inferring checkpoints from message roles.

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
    generated_checkpoint_message_ends: list[int] = field(default_factory=list)
    num_assistant: int = 0

    @property
    def token_ids(self) -> list[int]:
        """Current token IDs — the latest assistant checkpoint."""
        return self.trajectory_token_ids[-1] if self.trajectory_token_ids else []

    def append_record(self, record: SessionRecord) -> None:
        self.records.append(record)

    def plan_pretokenized(
        self,
        request_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        tito_tokenizer: TITOTokenizer,
        message_matcher: SessionMessageMatcher = message_matches,
    ) -> PreparedLinearRequest:
        """Plan rollback, authoritative history, and prompt tokens without mutation."""
        match_cache = MessageMatchCache(message_matcher)
        common_match_len = 0
        for index in range(min(len(request_messages), len(self.messages))):
            if not match_cache.matches(self.messages[index], request_messages[index]):
                break
            common_match_len = index + 1

        checkpoint_index = len(self.generated_checkpoint_message_ends) - 1
        reuse_prefix_len = len(self.messages) if self.token_ids else 0
        if self.token_ids and common_match_len < len(self.messages):
            checkpoint_index = -1
            for index in reversed(range(len(self.generated_checkpoint_message_ends))):
                if self.generated_checkpoint_message_ends[index] <= common_match_len:
                    checkpoint_index = index
                    break
            reuse_prefix_len = self.generated_checkpoint_message_ends[checkpoint_index] if checkpoint_index >= 0 else 0
            discard_count = self.num_assistant - (checkpoint_index + 1)
            if discard_count > MAX_ASSISTANT_ROLLBACK_STEPS:
                raise MessageValidationError(
                    f"rollback failed: discard_count={discard_count} exceeds "
                    f"max_assistant_rollback_steps={MAX_ASSISTANT_ROLLBACK_STEPS} "
                    f"(stored has {len(self.messages)} messages, "
                    f"request has {len(request_messages)} messages)"
                )

        history = build_authoritative_message_history(
            self.messages,
            request_messages,
            reuse_prefix_len=reuse_prefix_len,
        )
        if reuse_prefix_len == 0:
            prompt_token_ids = tito_tokenizer.apply_chat_template(
                history.effective_messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
            )
        else:
            stored_prefix = history.effective_messages[:reuse_prefix_len]
            try:
                assert_messages_append_only_with_allowed_role(
                    stored_prefix,
                    history.effective_messages,
                    tito_tokenizer.allowed_append_roles,
                )
            except ValueError as exc:
                raise MessageValidationError(
                    f"{exc}; the selected TITO fixed template does not support " "appending this role"
                ) from exc
            prompt_token_ids = tito_tokenizer.merge_tokens(
                old_messages=stored_prefix,
                new_messages=history.effective_messages,
                pretokenized_token_ids=self.trajectory_token_ids[checkpoint_index],
                tools=tools,
            )

        return PreparedLinearRequest(
            effective_messages=history.effective_messages,
            replayed_messages=history.replayed_messages,
            accepted_replay_indices=history.accepted_replay_indices,
            common_match_len=common_match_len,
            reuse_checkpoint_index=checkpoint_index,
            reuse_prefix_len=reuse_prefix_len,
            prompt_token_ids=prompt_token_ids,
        )

    def apply_prepared_request(self, prepared: PreparedLinearRequest) -> None:
        """Apply only the rollback selected by a fully validated request plan."""
        if prepared.reuse_prefix_len >= len(self.messages):
            return
        discard_count = self.num_assistant - (prepared.reuse_checkpoint_index + 1)
        logger.info(
            "Rolling back session: stored %d messages / %d checkpoints -> "
            "checkpoint %d (messages[:%d]), discarding %d generated checkpoint(s)",
            len(self.messages),
            self.num_assistant,
            prepared.reuse_checkpoint_index,
            prepared.reuse_prefix_len,
            discard_count,
        )
        self.messages = list(prepared.effective_messages[: prepared.reuse_prefix_len])
        end = prepared.reuse_checkpoint_index + 1
        self.trajectory_token_ids = self.trajectory_token_ids[:end]
        self.records = self.records[:end]
        self.generated_checkpoint_message_ends = self.generated_checkpoint_message_ends[:end]
        self.num_assistant = len(self.generated_checkpoint_message_ends)

    def prepare_pretokenized(
        self,
        request_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        tito_tokenizer: TITOTokenizer,
        message_matcher: SessionMessageMatcher = message_matches,
    ) -> list[int]:
        """Compatibility surface: plan, atomically apply rollback, return tokens."""
        prepared = self.plan_pretokenized(
            request_messages,
            tools,
            tito_tokenizer=tito_tokenizer,
            message_matcher=message_matcher,
        )
        self.apply_prepared_request(prepared)
        return prepared.prompt_token_ids

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

        self.messages = list(request_messages) + [assistant_message]
        self.trajectory_token_ids.append(all_token_ids)
        self.generated_checkpoint_message_ends.append(len(request_messages) + 1)
        self.num_assistant = len(self.generated_checkpoint_message_ends)


class SessionRegistry:
    """Session ID -> trajectory mapping with shared tokenizer resources.

    Pure CRUD plus read-only computation (compute_session_mismatch).
    Does NOT mutate session state - all mutations are methods on
    LinearTrajectory; called by the route handler under session.lock.
    """

    def __init__(
        self,
        args,
        tokenizer: Any,
        *,
        tito_tokenizer: TITOTokenizer,
        message_matcher: SessionMessageMatcher = message_matches,
        message_matcher_selector: str = "strict",
    ):
        self.sessions: dict[str, LinearTrajectory] = {}
        self.args = args
        self.tokenizer = tokenizer
        self.tito_tokenizer = tito_tokenizer
        self.message_matcher = message_matcher
        self.message_matcher_selector = message_matcher_selector
        self.comparator = tito_tokenizer.create_comparator()

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
                tools=tools,
                add_generation_prompt=False,
                tokenize=True,
            )
            mismatches = self.comparator.compare_sequences(expected_ids, session.token_ids)
            return [m.to_dict() for m in mismatches]
        except Exception as e:
            raise TokenizationError(f"failed to compute tito_session_mismatch: {e}") from e
