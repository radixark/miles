"""Session-as-forest data model: trajectory nodes and attach-point search.

The forest is append-only and ``seq`` (commit order) is the only ordering
key. Everything here is synchronous pure data — serving policy,
concurrency, and tokenization live one layer up in ``session_state``.
"""

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from miles.rollout.session.errors import TokenizationError
from miles.rollout.session.types import SessionRecord
from miles.utils.chat_template_utils.message_matcher_hub import SessionMessageMatcher, strict_message_matches

MAX_NODES = 1024

_INT32 = np.iinfo(np.int32)


@dataclass
class TrajectoryNode:
    """End of one model generation (SessionRecord is 1:1 with the node)."""

    delta_messages: list[dict[str, Any]]
    # This node's tokens past the reused parent prefix, as a read-only int32 array. Storing deltas keeps a
    # session's token storage O(final context) instead of one full snapshot per turn. Left out of __eq__:
    # comparing arrays is elementwise, so a generated __eq__ would raise instead of returning a bool.
    token_delta: np.ndarray = field(compare=False)
    # Parent tokens this node's snapshot reuses verbatim: the parent's full length, unless a
    # max_trim_tokens > 0 template rewrote the parent's trailing tokens at the turn boundary. The rewritten
    # positions are in this node's delta and replace the parent's tail in this node's snapshot.
    parent_reuse: int
    completion_span: tuple[int, int]  # this node's sampled completion within token_ids
    seq: int  # per-session logical commit order — THE ordering key
    committed_at: float  # wall clock, decoration only (NTP-unsafe; never order by this)
    response_id: str  # upstream response id: the agent-branch <-> leaf join key
    record: SessionRecord
    finish_reason: str
    # Full resolved request for this generation except its input_ids (this node's prompt tokens),
    # held by reference: read-only once committed.
    turn_args: dict[str, Any] = field(default_factory=dict)
    parent: "TrajectoryNode | None" = None
    children: list["TrajectoryNode"] = field(default_factory=list, repr=False)

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"

    @property
    def num_tokens(self) -> int:
        """Length of the root->node snapshot, without materializing it."""
        return self.parent_reuse + self.token_delta.size

    @property
    def token_ids(self) -> list[int]:
        """The full root->node snapshot, materialized from the path's deltas.

        Each access builds a new list in O(path tokens), so read it once per use. Each delta starts at or
        below its parent's length and runs to its node's own length, so after writing the deltas root
        first, every position below a node's length holds that node's token: a boundary rewrite replaces
        the tail it rewrote, and an ancestor's tail past a shorter descendant is cut off by the slice.
        """
        nodes = self.path_nodes()
        snapshot = np.empty(max(node.num_tokens for node in nodes), dtype=np.int32)
        for node in nodes:
            snapshot[node.parent_reuse : node.num_tokens] = node.token_delta
        return snapshot[: self.num_tokens].tolist()

    def path_nodes(self) -> list["TrajectoryNode"]:
        nodes: list[TrajectoryNode] = []
        node: TrajectoryNode | None = self
        while node is not None:
            nodes.append(node)
            node = node.parent
        nodes.reverse()
        return nodes

    def path_messages(self) -> list[dict[str, Any]]:
        return [message for node in self.path_nodes() for message in node.delta_messages]


@dataclass(frozen=True)
class AttachPoint:
    """Result of matching a request against the tree."""

    node: "TrajectoryNode | None"  # None = new root
    matched_messages: int  # request messages consumed by the attach node's path
    best_overlap: int  # diagnostics only: deepest overlap seen, incl. partial deltas


class SessionTree:
    """Forest of trajectory nodes plus the append-only commit surface."""

    def __init__(self) -> None:
        self.roots: list[TrajectoryNode] = []
        self.nodes: list[TrajectoryNode] = []  # creation (seq) order

    def leaves(self) -> list[TrajectoryNode]:
        return [node for node in self.nodes if not node.children]

    def snapshot(self) -> "SessionTree":
        """Copy the tree structure without copying committed messages, tokens or records."""
        tree = SessionTree()
        for node in self.nodes:
            parent = tree.nodes[node.parent.seq] if node.parent is not None else None
            copied = replace(node, parent=parent, children=[])
            tree.nodes.append(copied)
            if parent is None:
                tree.roots.append(copied)
            else:
                parent.children.append(copied)
        return tree

    def create_node(
        self,
        parent: TrajectoryNode | None,
        *,
        delta_messages: list[dict[str, Any]],
        token_ids: list[int],
        completion_span: tuple[int, int],
        committed_at: float,
        response_id: str,
        record: SessionRecord,
        finish_reason: str,
        turn_args: dict[str, Any] | None = None,
        parent_token_ids: list[int] | None = None,
        verified_prefix_length: int = 0,
    ) -> TrajectoryNode:
        """Append a node for the full root->node snapshot ``token_ids``; only its delta is stored.

        A caller that already holds the parent's snapshot passes it as ``parent_token_ids``, together
        with the prefix length it has checked equal, so only the rest of the parent is compared again.
        """
        if len(self.nodes) >= MAX_NODES:
            raise ValueError(
                f"node cap reached ({MAX_NODES}): the session cannot branch or extend "
                f"further — this almost always means the harness is not replaying "
                f"history verbatim"
            )
        parent_reuse = (
            0
            if parent is None
            else _reused_parent_prefix(
                parent,
                token_ids,
                parent_token_ids=parent.token_ids if parent_token_ids is None else parent_token_ids,
                verified_prefix_length=verified_prefix_length,
            )
        )
        node = TrajectoryNode(
            delta_messages=list(delta_messages),
            token_delta=_int32_token_delta(token_ids[parent_reuse:]),
            parent_reuse=parent_reuse,
            completion_span=completion_span,
            seq=len(self.nodes),
            committed_at=committed_at,
            response_id=response_id,
            record=record,
            finish_reason=finish_reason,
            turn_args=turn_args or {},
            parent=parent,
        )
        self.nodes.append(node)
        if parent is None:
            self.roots.append(node)
        else:
            parent.children.append(node)
        return node

    def find_attach_point(
        self,
        request_messages: list[dict[str, Any]],
        *,
        message_matcher: SessionMessageMatcher | None = None,
    ) -> AttachPoint:
        """Deepest node whose full path messages are a prefix of the request.

        Message equivalence is decided by *message_matcher* (defaults to the
        strict matcher).  A node is only entered after its parent's delta is
        fully consumed; ties on depth (twins whose deltas both match) go to
        the latest ``seq``.  Pure judgment — never mutates the forest.
        """
        matcher = message_matcher if message_matcher is not None else strict_message_matches
        best: TrajectoryNode | None = None
        best_matched = -1
        best_overlap = 0

        stack = [(root, 0) for root in reversed(self.roots)]
        while stack:
            node, offset = stack.pop()
            delta = node.delta_messages
            i = 0
            while (
                i < len(delta)
                and offset + i < len(request_messages)
                and matcher(delta[i], request_messages[offset + i])
            ):
                i += 1
            best_overlap = max(best_overlap, offset + i)
            if i < len(delta):
                continue  # partial delta: this node (and its subtree) is not a candidate
            matched = offset + len(delta)
            if matched > best_matched or (matched == best_matched and best is not None and node.seq > best.seq):
                best, best_matched = node, matched
            stack.extend((child, matched) for child in reversed(node.children))

        if best is None:
            return AttachPoint(node=None, matched_messages=0, best_overlap=best_overlap)
        return AttachPoint(node=best, matched_messages=best_matched, best_overlap=best_overlap)


def _reused_parent_prefix(
    parent: TrajectoryNode,
    token_ids: list[int],
    *,
    parent_token_ids: list[int],
    verified_prefix_length: int,
) -> int:
    """How many of the parent's tokens ``token_ids`` repeats verbatim.

    Assembly and ``GET /sessions`` read every record's prompt as a prefix of its leaf's snapshot.
    That is exact only while no boundary rewrite reaches the prompt of any turn on the path, so such
    a rewrite is refused.
    """
    limit = min(len(token_ids), parent.num_tokens)
    reuse = min(verified_prefix_length, limit)
    while reuse < limit and token_ids[reuse] == parent_token_ids[reuse]:
        reuse += 1
    prompt_length = max(node.completion_span[0] for node in parent.path_nodes())
    if reuse < prompt_length:
        raise TokenizationError(
            f"boundary rewrite reaches into an earlier turn's prompt: reuses {reuse} tokens, "
            f"the longest earlier prompt has {prompt_length}"
        )
    return reuse


def _int32_token_delta(token_ids: list[int]) -> np.ndarray:
    """Store token ids as a read-only int32 array, refusing ids outside int32 instead of wrapping them."""
    wide = np.asarray(token_ids, dtype=np.int64)
    if wide.size and (wide.min() < _INT32.min or wide.max() > _INT32.max):
        raise TokenizationError("token id outside int32 storage range")
    delta = wide.astype(np.int32)
    delta.setflags(write=False)
    return delta
