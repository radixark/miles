"""Session state over the trajectory tree: always-branch serving.

This module is the serving policy on top. A request is never rejected for mismatching stored history and never
destroys anything: it attaches at the deepest matching node, its unmatched
suffix becomes the new branch's delta, and whatever is not a clean
extension just grows a sibling or a new root. Whether a branch was a retry
is decided later by the sample_picker, not here.

Concurrency contract: single lock on the whole tree. A commit only appends a new
node under the parent captured at positioning time, so concurrent
generations from the same spot become sibling nodes instead of a conflict.
"""

import asyncio
import logging
import uuid
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any

from miles.rollout.session.errors import MessageValidationError, TokenizationError, TruncatedGenerationError
from miles.rollout.session.linear_trajectory import SessionRegistry, assert_pretokenized_prefix
from miles.rollout.session.request_contract import validate_session_render_fingerprint
from miles.rollout.session.types import SessionRecord
from miles.rollout.session.v2.tree_trajectory import SessionTree, TrajectoryNode
from miles.utils.chat_template_utils.message_matcher_hub import SessionMessageMatcher
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer

logger = logging.getLogger(__name__)


@dataclass
class SessionStateV2:
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
    render_fingerprint: Hashable | None = None

    def active_path(self) -> list[TrajectoryNode]:
        return self.active_leaf.path_nodes() if self.active_leaf is not None else []

    def active_messages(self) -> list[dict[str, Any]]:
        return self.active_leaf.path_messages() if self.active_leaf is not None else []

    def active_records(self) -> list[SessionRecord]:
        return [node.record for node in self.active_path()]

    def active_token_ids(self) -> list[int]:
        return self.active_leaf.token_ids if self.active_leaf is not None else []


def position_for_request(
    state: SessionStateV2,
    request_messages: list[dict[str, Any]],
    *,
    message_matcher: SessionMessageMatcher | None = None,
) -> None:
    """Move the view (``active_leaf``) to the attach point for *request_messages*."""
    attach = state.tree.find_attach_point(request_messages, message_matcher=message_matcher)

    if attach.node is not None and attach.node.truncated:
        raise TruncatedGenerationError(
            "truncated generation cannot be extended: the matched node ended with "
            "finish_reason='length' and truncation closes that path for good; "
            "branch before the cut instead"
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


def prepare_pretokenized(
    state: SessionStateV2,
    request_messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    tito_tokenizer: TITOTokenizer,
    render_fingerprint: Hashable | None = None,
) -> list[int]:
    """Pretokenized input_ids for the positioned view.

    - No attach node: render the whole request from scratch.
    - Otherwise: reuse the parent's token snapshot as-is and tokenize only
      the new suffix on top — the shared prefix is never re-rendered.
    """
    validate_session_render_fingerprint(state.render_fingerprint, render_fingerprint)
    parent = state.active_leaf
    if parent is None:
        prompt_token_ids = tito_tokenizer.apply_chat_template(
            request_messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
        )
    else:
        stored = state.active_messages()
        _validate_suffix_roles(request_messages[len(stored) :], tito_tokenizer)
        effective_messages = stored + request_messages[len(stored) :]
        prompt_token_ids = tito_tokenizer.merge_tokens(
            old_messages=stored,
            new_messages=effective_messages,
            pretokenized_token_ids=parent.token_ids,
            tools=tools,
        )

    if state.render_fingerprint is None:
        state.render_fingerprint = render_fingerprint
    return prompt_token_ids


def _validate_suffix_roles(
    suffix: list[dict[str, Any]],
    tito_tokenizer: TITOTokenizer,
) -> None:
    allowed = set(tito_tokenizer.allowed_append_roles)
    for message in suffix:
        role = message.get("role")
        if role not in allowed:
            raise MessageValidationError(
                f"appended message role={role!r} not allowed "
                f"(allowed={sorted(set(tito_tokenizer.allowed_append_roles))}); "
                "the selected TITO fixed template does not support appending this role"
            )


def commit_generation(
    state: SessionStateV2,
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
    positioning time), then advance the view to the new node. Prefix
    validation is byte-identical to the pre-tree checkpoint check."""
    all_token_ids = prompt_token_ids + completion_token_ids
    assert_pretokenized_prefix(
        parent.token_ids if parent is not None else [],
        all_token_ids,
        max_trim_tokens=max_trim_tokens,
        request_messages=request_messages,
        assistant_message=assistant_message,
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


class SessionRegistryV2(SessionRegistry):
    """Session ID -> session state mapping with shared tokenizer resources.

    The v1 registry shell (CRUD + tokenizer resources) with the session type
    swapped to ``SessionStateV2``; all session mutations go through the
    module-level serving functions, called by the route handler under
    ``SessionStateV2.lock``.
    """

    sessions: dict[str, SessionStateV2]

    def create_session(self) -> str:
        session_id = uuid.uuid4().hex
        self.sessions[session_id] = SessionStateV2()
        return session_id

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

    def compute_session_mismatch(self, state: SessionStateV2) -> list[dict] | None:
        """The active-path view of ``compute_mismatch``."""
        if state.active_leaf is None:
            return None
        records = state.active_records()
        tools = records[-1].request.get("tools") if records else None
        return self.compute_mismatch(state.active_messages(), state.active_token_ids(), tools)
