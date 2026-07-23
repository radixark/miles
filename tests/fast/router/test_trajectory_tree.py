"""Unit tests for the v5 trajectory tree: data model + attach-point search.

Pure model level (no serving wiring) — the case numbers reference the
serving matrix in MULTI_LINEAGE_DESIGN.md.
"""

import pytest

from miles.rollout.session.v2 import trajectory_tree
from miles.rollout.session.v2.trajectory_tree import SessionTree

SYS = {"role": "system", "content": "sys"}
U1 = {"role": "user", "content": "q1"}
A1 = {"role": "assistant", "content": "a1"}
T1 = {"role": "tool", "content": "t1", "tool_call_id": "c1"}
A2 = {"role": "assistant", "content": "a2"}
T2 = {"role": "tool", "content": "t2", "tool_call_id": "c2"}
A3 = {"role": "assistant", "content": "a3"}


def _commit(tree, parent, delta, *, finish_reason="stop", committed_at=None):
    seq = len(tree.nodes)
    prefix = parent.token_ids if parent is not None else []
    tokens = prefix + [100 + seq]
    return tree.create_node(
        parent,
        delta_messages=delta,
        token_ids=tokens,
        completion_span=(len(tokens) - 1, len(tokens)),
        committed_at=committed_at if committed_at is not None else float(seq),
        response_id=f"resp-{seq}",
        record=object(),
        finish_reason=finish_reason,
    )


@pytest.fixture
def linear_tree():
    """root n0=[sys,u1,a1] -> n1=[t1,a2] -> n2=[t2,a3]"""
    tree = SessionTree()
    n0 = _commit(tree, None, [SYS, U1, A1])
    n1 = _commit(tree, n0, [T1, A2])
    n2 = _commit(tree, n1, [T2, A3])
    return tree, n0, n1, n2


class TestAttachPoint:
    def test_case1_strict_leaf_extension(self, linear_tree):
        tree, _, _, n2 = linear_tree
        ap = tree.find_attach_point([SYS, U1, A1, T1, A2, T2, A3, {"role": "user", "content": "next"}])
        assert ap.node is n2
        assert ap.matched_messages == 7

    def test_case2_degenerate_resend_of_full_path(self, linear_tree):
        tree, _, _, n2 = linear_tree
        ap = tree.find_attach_point([SYS, U1, A1, T1, A2, T2, A3])
        assert ap.node is n2
        assert ap.matched_messages == 7  # empty suffix: generate a new child of n2

    def test_case3_internal_node_divergence(self, linear_tree):
        tree, _, n1, _ = linear_tree
        t2_diff = {"role": "tool", "content": "t2-DIFF", "tool_call_id": "c2"}
        ap = tree.find_attach_point([SYS, U1, A1, T1, A2, t2_diff])
        assert ap.node is n1
        assert ap.matched_messages == 5
        assert ap.best_overlap == 5

    def test_case4_divergence_inside_child_delta_attaches_parent(self, linear_tree):
        tree, n0, _, _ = linear_tree
        t1_diff = {"role": "tool", "content": "t1-DIFF", "tool_call_id": "c1"}
        ap = tree.find_attach_point([SYS, U1, A1, t1_diff])
        assert ap.node is n0
        assert ap.matched_messages == 3

    def test_case4b_divergence_inside_root_delta_is_new_root(self, linear_tree):
        tree, _, _, _ = linear_tree
        ap = tree.find_attach_point([SYS, {"role": "user", "content": "other"}])
        assert ap.node is None
        assert ap.matched_messages == 0
        assert ap.best_overlap == 1  # sys matched before the divergence

    def test_case4b_pure_prefix_of_root_delta_is_new_root(self, linear_tree):
        """Today's no-anchor shape: resend [sys, u1] against stored [sys, u1, a1, ...]."""
        tree, _, _, _ = linear_tree
        ap = tree.find_attach_point([SYS, U1])
        assert ap.node is None
        assert ap.best_overlap == 2

    def test_case5_foreign_assistant_divergence_attaches_parent(self, linear_tree):
        """Client rewrote a2 (compaction): the rewritten assistant lands in the
        new branch's delta; matching just stops before it."""
        tree, n0, _, _ = linear_tree
        a2_rewritten = {"role": "assistant", "content": "a2-compacted"}
        ap = tree.find_attach_point([SYS, U1, A1, T1, a2_rewritten, {"role": "user", "content": "go on"}])
        assert ap.node is n0
        assert ap.matched_messages == 3
        assert ap.best_overlap == 4  # sys,u1,a1 + t1 inside n1's delta

    def test_case6_zero_overlap_is_new_root(self, linear_tree):
        tree, _, _, _ = linear_tree
        ap = tree.find_attach_point([{"role": "system", "content": "subagent"}, {"role": "user", "content": "task"}])
        assert ap.node is None
        assert ap.best_overlap == 0

    def test_case9_twin_tie_goes_to_latest_seq(self):
        tree = SessionTree()
        n0 = _commit(tree, None, [SYS, U1, A1])
        twin_a = _commit(tree, n0, [T1, A2])
        twin_b = _commit(tree, n0, [T1, A2])  # same delta text, later seq
        assert twin_a.seq < twin_b.seq
        ap = tree.find_attach_point([SYS, U1, A1, T1, A2, {"role": "user", "content": "next"}])
        assert ap.node is twin_b

    def test_deepest_match_wins_across_roots(self):
        """Two roots where one's opening is a prefix of the other's history."""
        tree = SessionTree()
        r1 = _commit(tree, None, [SYS, U1, A1])
        n1 = _commit(tree, r1, [T1, A2])
        _commit(tree, None, [SYS, U1, {"role": "assistant", "content": "other-a1"}])
        ap = tree.find_attach_point([SYS, U1, A1, T1, A2, T2])
        assert ap.node is n1

    def test_empty_request_is_new_root(self, linear_tree):
        tree, _, _, _ = linear_tree
        ap = tree.find_attach_point([])
        assert ap.node is None
        assert ap.best_overlap == 0

    def test_search_never_mutates(self, linear_tree):
        tree, _, _, _ = linear_tree
        before = [(n.seq, len(n.children)) for n in tree.nodes]
        tree.find_attach_point([SYS, U1, A1, T1, A2])
        assert [(n.seq, len(n.children)) for n in tree.nodes] == before


class TestModel:
    def test_case10_concurrent_commits_become_siblings(self, linear_tree):
        tree, n0, _, _ = linear_tree
        s1 = _commit(tree, n0, [{"role": "user", "content": "sub A"}, A2])
        s2 = _commit(tree, n0, [{"role": "user", "content": "sub B"}, A3])
        assert s1 in n0.children and s2 in n0.children
        assert s1.parent is n0 and s2.parent is n0

    def test_full_snapshot_and_path_helpers(self, linear_tree):
        _, n0, n1, n2 = linear_tree
        assert n2.token_ids[: len(n1.token_ids)] == n1.token_ids  # snapshot inherits prefix
        assert n2.path_nodes() == [n0, n1, n2]
        assert n2.path_messages() == [SYS, U1, A1, T1, A2, T2, A3]

    def test_case7_truncated_is_derived_from_finish_reason(self):
        tree = SessionTree()
        node = _commit(tree, None, [U1, A1], finish_reason="length")
        assert node.truncated
        assert not _commit(tree, node, [T1, A2]).truncated

    def test_leaves_in_creation_order(self, linear_tree):
        tree, n0, _, n2 = linear_tree
        side = _commit(tree, n0, [T1, {"role": "assistant", "content": "retry"}])
        assert tree.leaves() == [n2, side]

    def test_case12_node_cap_fails_loud(self, monkeypatch):
        monkeypatch.setattr(trajectory_tree, "MAX_NODES", 2)
        tree = SessionTree()
        n0 = _commit(tree, None, [U1, A1])
        _commit(tree, n0, [T1, A2])
        with pytest.raises(ValueError, match="node cap reached"):
            _commit(tree, n0, [T1, A3])

    def test_seq_is_creation_order_not_wall_clock(self):
        """Wall clock is decoration: ordering must survive a clock step backwards."""
        tree = SessionTree()
        n0 = _commit(tree, None, [SYS, U1, A1], committed_at=100.0)
        early_clock = _commit(tree, n0, [T1, A2], committed_at=50.0)
        later = _commit(tree, n0, [T1, A2], committed_at=99.0)
        assert early_clock.seq < later.seq
        ap = tree.find_attach_point([SYS, U1, A1, T1, A2, T2])
        assert ap.node is later
