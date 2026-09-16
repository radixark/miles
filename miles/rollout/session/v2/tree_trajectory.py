"""Session-as-forest data model: trajectory nodes and attach-point search.

The forest is append-only and ``seq`` (commit order) is the only ordering
key. Everything here is synchronous pure data — serving policy,
concurrency, and tokenization live one layer up in ``session_state``.
"""

from dataclasses import dataclass, field
from typing import Any

from miles.rollout.session.types import SessionRecord
from miles.utils.chat_template_utils.message_matcher_hub import SessionMessageMatcher, strict_message_matches

MAX_NODES = 1024


@dataclass
class TrajectoryNode:
    """End of one model generation (SessionRecord is 1:1 with the node)."""

    delta_messages: list[dict[str, Any]]
    token_ids: list[int]  # full root->node snapshot
    completion_span: tuple[int, int]  # this node's sampled completion within token_ids
    seq: int  # per-session logical commit order — THE ordering key
    committed_at: float  # wall clock, decoration only (NTP-unsafe; never order by this)
    response_id: str  # upstream response id: the agent-branch <-> leaf join key
    record: SessionRecord
    finish_reason: str
    parent: "TrajectoryNode | None" = None
    children: list["TrajectoryNode"] = field(default_factory=list, repr=False)

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"

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
    ) -> TrajectoryNode:
        if len(self.nodes) >= MAX_NODES:
            raise ValueError(
                f"node cap reached ({MAX_NODES}): the session cannot branch or extend "
                f"further — this almost always means the harness is not replaying "
                f"history verbatim"
            )
        node = TrajectoryNode(
            delta_messages=list(delta_messages),
            token_ids=list(token_ids),
            completion_span=completion_span,
            seq=len(self.nodes),
            committed_at=committed_at,
            response_id=response_id,
            record=record,
            finish_reason=finish_reason,
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
