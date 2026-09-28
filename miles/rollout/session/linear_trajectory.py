import asyncio
import logging
import uuid
from copy import deepcopy
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
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer, extract_template_args

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

    ``turn_args_history`` stores each successful generation's full resolved request.
    Rollback slices it with the token checkpoints. Model rules decide which of
    these fields a continuation inherits or must keep unchanged.

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
    turn_args_history: list[dict[str, Any]] = field(default_factory=list)
    evaluation: bool = False
    sampling_defaults: dict[str, Any] = field(default_factory=dict)
    sampling_support_replay: bool = False

    @property
    def turn_args(self) -> dict[str, Any]:
        """The full resolved request at the current tip; empty before the first checkpoint."""
        return self.turn_args_history[-1] if self.turn_args_history else {}

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

        In order: select the checkpoint *client_args*'s messages continue
        (``_find_rollback_checkpoint``); decide the request args
        (``request_args.prepare_chat_request``, which may raise a 400); render
        the prompt with those template args; then roll back to the checkpoint
        (``_rollback_to_checkpoint``). The args come before the render because
        they change the token ids. Both steps precede rollback so a validation
        or rendering error leaves session state unchanged.

        The caller must hold `self.lock`.
        """
        matcher = message_matcher if message_matcher is not None else strict_message_matches
        request_messages = client_args.get("messages", [])
        checkpoint_index = self._find_rollback_checkpoint(request_messages, matcher)
        turn_args = self.turn_args_history[checkpoint_index] if checkpoint_index >= 0 else None
        prepared = prepare_chat_request(
            client_args,
            tito_tokenizer,
            config=config,
            turn_args=turn_args,
            evaluation=self.evaluation,
            sampling_defaults=self.sampling_defaults,
            sampling_support_replay=self.sampling_support_replay,
        )
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
        """Build prompt token IDs while reusing the selected checkpoint's token prefix.

        Use `template_args` for newly rendered tokens, without modifying session state.

        Validates that *request_messages* extends the selected checkpoint's history
        under *message_matcher* (defaults to the strict matcher) and reuses its stored
        token_ids as the pretokenized prefix. When no stored checkpoint is selected
        to build on — the first turn, or a retry of the first turn that selects
        the empty checkpoint — renders *request_messages* from scratch via
        the chat template instead.

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
        # Confirm the selected checkpoint's stored messages are a prefix of request,
        #    and that each appended message role is in tito_tokenizer.allowed_append_roles.
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
        turn_args: dict[str, Any] | None = None,
    ) -> None:
        """Store raw token IDs after a successful response.

        Appends ``prompt_token_ids + completion_token_ids`` as a new checkpoint,
        recording the full resolved request in ``turn_args`` alongside.
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
        self.turn_args_history.append(deepcopy(turn_args or {}))
        self.generated_checkpoint_message_ends.append(len(request_messages) + 1)
        self.num_assistant = len(self.generated_checkpoint_message_ends)

    def _find_rollback_checkpoint(
        self,
        request_messages: list[dict[str, Any]],
        message_matcher: SessionMessageMatcher,
    ) -> int:
        """Find the last generated checkpoint in the matching prefix, without mutation.

        In agentic workflows the agent may retry from an earlier point — for
        example, re-running a tool call with different arguments. When that
        happens the new request shares a common prefix with the stored messages
        but diverges before the end. This method selects the last generated
        assistant checkpoint within the matching prefix, or the empty checkpoint
        when the matching prefix holds no generated checkpoint at all. The caller
        applies rollback only after argument resolution and rendering succeed.

        Only a single-step rollback is allowed (controlled by
        ``MAX_ASSISTANT_ROLLBACK_STEPS``). Discarding exactly one generated
        checkpoint means the agent is retrying from the preceding checkpoint —
        the request shares the stored prefix up to that generated response and
        then continues with whatever the agent chooses (same or different tool
        result, additional messages, etc.). Any request that would need to
        discard more than one generated checkpoint (i.e. jump back across
        multiple turns) is rejected with ``MessageValidationError`` and no
        state is modified.

        Example — agent retries after the first tool call::

            stored:  [sys, user, assistant₁, tool₁, assistant₂]
                      ───────────────────── ▲
                      checkpoint 0 (assistant₁)   checkpoint 1 (assistant₂)

            request: [sys, user, assistant₁, tool₁_different, ...]
                                             ↑ diverges here (index 3)

            match_len = 3  (sys, user, assistant₁ all match)
            Last generated checkpoint in matched prefix → assistant₁ (checkpoint 0)
            discard_count = 2 - 1 = 1  (≤ MAX_ASSISTANT_ROLLBACK_STEPS)

            After preparation succeeds and the caller applies rollback:
              messages           = [sys, user, assistant₁]
              trajectory_token_ids = [checkpoint_0_ids]
              records              = [record_0]
              num_assistant        = 1

        Example — agent retries the very first turn::

            stored:  [user, assistant₁]
            request: [user]
                           ↑ stored continues past the request (index 1)

            match_len = 1  (user matches), no generated checkpoint in the matched prefix
            Rollback target → the empty checkpoint (index -1)
            discard_count = 1 - 0 = 1  (≤ MAX_ASSISTANT_ROLLBACK_STEPS)

            The caller re-renders the prompt from scratch before applying rollback.
            After successful preparation and rollback, the session is empty,
            so turn 1 regenerates like any later turn.

        No rollback is needed when:
        - The stored history is empty (return -1).
        - *request_messages* is a strict extension of stored messages
          (``match_len >= len(stored)``; return the current checkpoint index).
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

        # Only responses generated by this session create checkpoints.
        # Client-injected assistant messages won't create new checkpoints.
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
        # No generated checkpoint in the matched prefix means the agent is retrying the
        # first turn, so roll back to the empty checkpoint and retain no messages.
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
        self.turn_args_history = self.turn_args_history[: checkpoint_index + 1]
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

    def create_session(
        self,
        *,
        evaluation: bool = False,
        sampling_defaults: dict[str, Any] | None = None,
        sampling_support_replay: bool = False,
    ) -> str:
        session_id = uuid.uuid4().hex
        self.sessions[session_id] = LinearTrajectory(
            evaluation=evaluation,
            sampling_defaults=dict(sampling_defaults or {}),
            sampling_support_replay=sampling_support_replay,
        )
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
            # No new request is available here; use the committed turn's renderer fields.
            expected_ids = self.tito_tokenizer.apply_chat_template(
                session.messages,
                add_generation_prompt=False,
                tokenize=True,
                template_args=extract_template_args(session.turn_args),
            )
            mismatches = self.comparator.compare_sequences(expected_ids, session.token_ids)
            return [m.to_dict() for m in mismatches]
        except Exception as e:
            raise TokenizationError(f"failed to compute tito_session_mismatch: {e}") from e
