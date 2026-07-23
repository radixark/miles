"""Trajectory tree: the v5 session data model (MULTI_LINEAGE_DESIGN.md).

A session is a forest. A node is the end of ONE model generation: its
``delta_messages`` are the messages that generation's request added beyond
the parent (env messages plus any client-carried foreign assistants) plus the
sampled assistant; its ``token_ids`` are the FULL root->node snapshot (v1
stores no deltas), so every node is directly injectable as a pretokenized
prefix and every leaf is its own assembled token sequence.

Attach-point search is pure and never mutates the forest; node creation is
append-only and happens under the owning ``SessionState.lock`` — concurrent
commits become sibling nodes, never conflicts.
"""

from dataclasses import dataclass, field
from typing import Any

from miles.utils.chat_template_utils import message_matches

# Branching backstop: fail loud on runaway branching (e.g. replay drift)
# instead of growing forever; normal sessions stay far below this. The
# authoritative check is at commit time (create_node); any dispatch-time
# check is fast-fail only.
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
    record: Any
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
    """Result of matching a request against the forest.

    ``node is None`` means no node's full path is a prefix of the request:
    the request starts a new root. ``matched_messages`` counts the request
    messages consumed by the attach node's path (0 for a new root); the
    request suffix ``request_messages[matched_messages:]`` becomes the new
    branch's delta (empty suffix = degenerate resend: generate a new child of
    the attach node). ``best_overlap`` is diagnostics only: the deepest
    message-level overlap seen anywhere, including partial matches inside a
    node's delta.
    """

    node: "TrajectoryNode | None"
    matched_messages: int
    best_overlap: int


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
        record: Any,
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
            token_ids=token_ids,
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

    def find_attach_point(self, request_messages: list[dict[str, Any]]) -> AttachPoint:
        """Deepest node whose full path messages are a prefix of the request.

        A node is only entered after its parent's delta is fully consumed;
        ties on depth (twins whose deltas both match) go to the latest ``seq``.
        Pure judgment — never mutates the forest.
        """
        best: TrajectoryNode | None = None
        best_matched = -1
        best_overlap = 0

        def visit(node: TrajectoryNode, offset: int) -> None:
            nonlocal best, best_matched, best_overlap
            delta = node.delta_messages
            i = 0
            while (
                i < len(delta)
                and offset + i < len(request_messages)
                and message_matches(delta[i], request_messages[offset + i])
            ):
                i += 1
            best_overlap = max(best_overlap, offset + i)
            if i < len(delta):
                return  # partial delta: this node (and its subtree) is not a candidate
            matched = offset + len(delta)
            if matched > best_matched or (matched == best_matched and best is not None and node.seq > best.seq):
                best, best_matched = node, matched
            for child in node.children:
                visit(child, matched)

        for root in self.roots:
            visit(root, 0)

        if best is None:
            return AttachPoint(node=None, matched_messages=0, best_overlap=best_overlap)
        return AttachPoint(node=best, matched_messages=best_matched, best_overlap=best_overlap)
