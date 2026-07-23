"""Session state over the trajectory tree: always-branch serving.

The tree (``trajectory_tree``) is the storage. Default serving never
destroys and never rejects a mismatch: ``position_for_request`` finds the
deepest attach point (whole forest), the suffix becomes the new branch's
delta, and every non-extension simply grows a sibling or a new root —
retry semantics lives entirely in the samples-op picker now. The strict
mode (``--session-strict-append-only``) is the single-chain invariant
guard: anything that would create a sibling or a second root fails loud
with diagnostics.

Concurrency contract: positioning and commits run under
``SessionState.lock``; the proxy call happens outside the lock. Commits are
append-only node creations under an explicitly captured parent, so
concurrent generations become sibling nodes — under strict mode a commit
whose view moved during flight is dropped instead (identity guard in
``core``).
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from miles.rollout.session.errors import MessageValidationError, SessionError, SessionNotFoundError, TokenizationError
from miles.rollout.session.types import SessionRecord
from miles.rollout.session.v2.trajectory_tree import SessionTree, TrajectoryNode
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer

logger = logging.getLogger(__name__)


class TruncatedGenerationError(SessionError):
    """Raised when a request extends a generation that ended with
    finish_reason='length': truncation closes that path for good — branch
    before the cut instead. v2-only (v1 never branches), hence defined here;
    the shared SessionError handler maps it to its status code."""

    status_code: int = 409


@dataclass
class SessionState:
    """Per-session concurrency container plus the trajectory forest.

    ``active_leaf`` is the head of the single-chain view: the path root ->
    active_leaf is what GET /sessions, judgment, and sample assembly see.
    ``None`` means no committed generation yet (empty view, first-turn
    semantics — a failed first turn leaves the session fully retryable).
    """

    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)
    closing: bool = field(default=False, repr=False, compare=False)
    tree: SessionTree = field(default_factory=SessionTree)
    active_leaf: TrajectoryNode | None = None

    def active_path(self) -> list[TrajectoryNode]:
        return self.active_leaf.path_nodes() if self.active_leaf is not None else []

    def active_messages(self) -> list[dict[str, Any]]:
        return self.active_leaf.path_messages() if self.active_leaf is not None else []

    def active_records(self) -> list[SessionRecord]:
        return [node.record for node in self.active_path()]

    def active_token_ids(self) -> list[int]:
        return self.active_leaf.token_ids if self.active_leaf is not None else []


def position_for_request(state: SessionState, request_messages: list[dict[str, Any]], *, strict: bool) -> None:
    """Position the view for *request_messages*.

    Default (always-branch): the attach point is the deepest node whose full
    path is a message-prefix of the request, anywhere in the forest; the view
    moves there and the suffix becomes the new branch's delta. Nothing is
    ever destroyed or rejected for mismatching — except extending a
    length-truncated generation (409: truncation closes that path for good).

    Strict (``--session-strict-append-only``): the forest must stay a single
    chain. Anything that would create a sibling or a second root — attaching
    to an internal node, a degenerate resend of a non-leaf path, any
    divergence — fails loud with attach diagnostics. A failed first turn
    leaves no node, so any new first request is still accepted.
    """
    attach = state.tree.find_attach_point(request_messages)

    if attach.node is not None and attach.node.truncated:
        raise TruncatedGenerationError(
            "truncated generation cannot be extended: the matched node ended with "
            "finish_reason='length' and truncation closes that path for good; "
            "branch before the cut instead"
        )

    if strict:
        current_leaf = state.active_leaf
        creates_root = attach.node is None and state.tree.nodes
        creates_sibling = attach.node is not None and (attach.node is not current_leaf or attach.node.children)
        if creates_root or creates_sibling:
            attach_at = attach.node.seq if attach.node is not None else None
            raise MessageValidationError(
                f"session is append-only (--session-strict-append-only): request must "
                f"strictly extend the current chain (attach node seq={attach_at}, "
                f"matched {attach.matched_messages} messages, best overlap "
                f"{attach.best_overlap} of {len(request_messages)} request messages)"
            )

    if attach.node is not state.active_leaf:
        logger.info(
            "Branching: request(%d msgs) attaches at node seq=%s "
            "(matched %d msgs, best overlap %d), tree has %d nodes",
            len(request_messages),
            attach.node.seq if attach.node is not None else "<new root>",
            attach.matched_messages,
            attach.best_overlap,
            len(state.tree.nodes),
        )
    state.active_leaf = attach.node


def prepare_input(
    state: SessionState,
    request_messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    tito_tokenizer: TITOTokenizer,
    strict: bool = False,
) -> list[int]:
    """Pretokenized input_ids for the positioned view.

    New roots render from scratch. Branch/extension suffixes inherit the
    attach node's snapshot: the canonical render of the matched prefix is
    spliced with the snapshot (preserving the original sampled ids and the
    family's boundary tokens) and the canonical render of the full request
    supplies the suffix — which may contain client-carried assistants
    (compaction), rendered with reasoning preserved (preserve-think kwargs
    are family constants). If the snapshot diverges from the canonical
    prefix render, fall back to the full canonical render: correctness
    first, inheritance is an optimization (the mismatch metric will show it).

    Non-assistant suffix roles stay gated by ``--tito-allowed-append-roles``;
    client-carried assistants are always allowed (they are prompt), except on
    families that cannot re-render a mid-path assistant (DeepSeek-V3.2).
    """
    parent = state.active_leaf
    if parent is None:
        return tito_tokenizer.apply_chat_template(
            request_messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
        )

    stored = state.active_messages()
    suffix = request_messages[len(stored) :]
    _validate_suffix_roles(suffix, tito_tokenizer, assistant_exempt=not strict)

    if not any(m.get("role") == "assistant" for m in suffix):
        return tito_tokenizer.merge_tokens(
            old_messages=stored,
            new_messages=request_messages,
            pretokenized_token_ids=parent.token_ids,
            tools=tools,
        )

    full = tito_tokenizer.apply_chat_template(request_messages, tools=tools, add_generation_prompt=True, tokenize=True)
    prefix = tito_tokenizer.apply_chat_template(stored, tools=tools, add_generation_prompt=False, tokenize=True)
    snapshot = parent.token_ids
    if full[: len(prefix)] == prefix and len(prefix) >= len(snapshot) and prefix[: len(snapshot)] == snapshot:
        return snapshot + prefix[len(snapshot) :] + full[len(prefix) :]
    # The canonical render diverges from the stored snapshot (template quirks
    # like Qwen3's empty-think prompt block make this legal). Inheritance
    # would break the commit prefix check and the per-leaf record alignment,
    # so the branch degrades to a self-contained NEW ROOT: full canonical
    # prompt, zero inheritance — the carried history is prompt material.
    logger.warning(
        "Branch inheritance fell back to a zero-inheritance root: the stored "
        "snapshot (%d tokens) is not a prefix of the canonical render (%d prefix / %d full tokens)",
        len(snapshot),
        len(prefix),
        len(full),
    )
    state.active_leaf = None
    return full


def _validate_suffix_roles(
    suffix: list[dict[str, Any]],
    tito_tokenizer: TITOTokenizer,
    *,
    assistant_exempt: bool,
) -> None:
    """Carried assistants are prompt material for branches (always exempt in
    branch mode); under strict append-only they fall back to the role gate —
    an assistant the model never generated appearing on the main chain is
    exactly the bug strict mode exists to catch."""
    allowed = set(tito_tokenizer.allowed_append_roles)
    if assistant_exempt:
        allowed |= {"assistant"}
    for message in suffix:
        role = message.get("role")
        if role not in allowed:
            raise MessageValidationError(
                f"appended message role={role!r} not allowed "
                f"(allowed={sorted(set(tito_tokenizer.allowed_append_roles))}); "
                f"to allow more roles use --tito-allowed-append-roles"
            )
    if not tito_tokenizer.supports_midpath_assistant_rerender:
        assistant_positions = [i for i, m in enumerate(suffix) if m.get("role") == "assistant"]
        if assistant_positions and any(m.get("role") == "user" for m in suffix[assistant_positions[0] + 1 :]):
            raise MessageValidationError(
                "this template family cannot re-render a carried assistant that precedes a "
                "later user turn without cutting its reasoning (deepseek-v3.2 hardcodes the "
                "think gate); rewrite the branch without the carried assistant or use a "
                "different template family"
            )


def commit_generation(
    state: SessionState,
    *,
    parent: TrajectoryNode | None,
    request_messages: list[dict[str, Any]],
    assistant_message: dict[str, Any],
    prompt_token_ids: list[int],
    completion_token_ids: list[int],
    max_trim_tokens: int,
    record: SessionRecord,
    response_id: str,
    finish_reason: str,
) -> TrajectoryNode:
    """Validate and append one generation under *parent* (captured at
    positioning time — append-only, so concurrent generations become sibling
    nodes), then advance the view to the new node. Prefix validation is
    byte-identical to the pre-tree checkpoint check."""
    all_token_ids = prompt_token_ids + completion_token_ids
    prev = parent.token_ids if parent is not None else []
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

    parent_messages = parent.path_messages() if parent is not None else []
    delta = list(request_messages[len(parent_messages) :]) + [assistant_message]
    node = state.tree.create_node(
        parent,
        delta_messages=delta,
        token_ids=all_token_ids,
        completion_span=(len(prompt_token_ids), len(all_token_ids)),
        committed_at=record.timestamp,
        response_id=response_id,
        record=record,
        finish_reason=finish_reason,
    )
    state.active_leaf = node
    return node


class SessionRegistry:
    """Session ID -> session state mapping with shared tokenizer resources.

    Pure CRUD plus read-only computation (compute_session_mismatch); all
    session mutations go through the module-level serving functions, called
    by the route handler under ``SessionState.lock``.
    """

    def __init__(self, args, tokenizer: Any, *, tito_tokenizer: TITOTokenizer):
        self.sessions: dict[str, SessionState] = {}
        self.args = args
        self.tokenizer = tokenizer
        self.tito_tokenizer = tito_tokenizer
        self.comparator = tito_tokenizer.create_comparator()

    def create_session(self) -> str:
        session_id = uuid.uuid4().hex
        self.sessions[session_id] = SessionState()
        return session_id

    def get_session(self, session_id: str) -> SessionState:
        session = self.sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")
        return session

    def remove_session(self, session_id: str) -> None:
        if self.sessions.pop(session_id, None) is None:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")

    def compute_mismatch(self, messages: list[dict[str, Any]], token_ids: list[int], tools: Any) -> list[dict] | None:
        """Compare accumulated token IDs against canonical chat template
        output for one path. Read-only."""
        if not token_ids:
            return None
        try:
            expected_ids = self.tito_tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=False,
                tokenize=True,
            )
            mismatches = self.comparator.compare_sequences(expected_ids, token_ids)
            return [m.to_dict() for m in mismatches]
        except Exception as e:
            raise TokenizationError(f"failed to compute tito_session_mismatch: {e}") from e

    def compute_session_mismatch(self, state: SessionState) -> list[dict] | None:
        """The active-path view of ``compute_mismatch``."""
        if state.active_leaf is None:
            return None
        records = state.active_records()
        tools = records[-1].request.get("tools") if records else None
        return self.compute_mismatch(state.active_messages(), state.active_token_ids(), tools)
