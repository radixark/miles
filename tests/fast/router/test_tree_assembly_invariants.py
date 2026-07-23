"""Tree-merge legality invariants: randomized forest fuzz + targeted edges.

The oracle re-derives the rules from the design doc independently of the
production code (picker = temporal supersession with the length guardrail;
ownership = earliest surviving leaf) and then checks every emitted sample:

1. tokens == the leaf snapshot (a prefix of it under truncation), per-token
   fields in lockstep;
2. exactly-once: every trainable position lies inside a completion span the
   sample's leaf OWNS, and each owned span is fully trainable (clipped to
   the kept range) — shared spans train nowhere else;
3. the trimmed set matches the picker rule exactly; guardrail-violating
   trees 422 instead of guessing;
4. rewards land by response id; sample order is leaf commit order.
"""

import json
import random
import uuid
from types import SimpleNamespace

import pytest
from tests.fast.rollout.session.test_samples import _make_record

from miles.rollout.session.samples.codec import COMPUTED_FIELDS_V2, decode_samples_reply
from miles.rollout.session.v2.core import SessionCore
from miles.rollout.session.v2.session_state import SessionRegistry
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample

_ARGS = SimpleNamespace(
    miles_router_timeout=30,
    hf_checkpoint="Qwen/Qwen3-0.6B",
    chat_template_path=None,
    apply_chat_template_kwargs={"enable_thinking": False},
    tito_model="default",
    tito_allowed_append_roles=["tool"],
    session_server_instance_id=uuid.uuid4().hex,
)


class _UnusedBackend:
    async def do_proxy(self, *args, **kwargs):
        raise AssertionError("collect_samples must not touch the proxy backend")


@pytest.fixture(scope="module")
def core():
    tokenizer = load_tokenizer(_ARGS.hf_checkpoint, chat_template_path=None, trust_remote_code=True)
    tito_tokenizer = get_tito_tokenizer(
        tokenizer,
        tokenizer_type=_ARGS.tito_model,
        chat_template_kwargs=_ARGS.apply_chat_template_kwargs,
        allowed_append_roles=_ARGS.tito_allowed_append_roles,
    )
    registry = SessionRegistry(_ARGS, tokenizer, tito_tokenizer=tito_tokenizer)
    return SessionCore(_UnusedBackend(), registry, _ARGS, _ARGS.session_server_instance_id)


class _Grower:
    """Grows legal trees out of synthetic records with globally unique token ids."""

    def __init__(self, state):
        self.state = state
        self.next_token = 1

    def _fresh(self, n: int) -> list[int]:
        ids = list(range(self.next_token, self.next_token + n))
        self.next_token += n
        return ids

    def grow(self, parent, *, env_len: int, completion_len: int, finish_reason: str = "stop"):
        prefix = parent.token_ids if parent is not None else []
        prompt = prefix + self._fresh(env_len)
        completion = self._fresh(completion_len)
        record = _make_record(prompt_token_ids=prompt, output_token_ids=completion, finish_reason=finish_reason)
        return self.state.tree.create_node(
            parent,
            delta_messages=[],
            token_ids=prompt + completion,
            completion_span=(len(prompt), len(prompt) + len(completion)),
            committed_at=float(len(self.state.tree.nodes)),
            response_id=f"resp-{len(self.state.tree.nodes)}",
            record=record,
            finish_reason=finish_reason,
        )


async def _fresh_grower(core):
    response = await core.create_session()
    sid = json.loads(response.body)["session_id"]
    state = core.registry.sessions[sid]
    return sid, state, _Grower(state)


def _subtree_leaves(node):
    if not node.children:
        return [node]
    return [leaf for child in node.children for leaf in _subtree_leaves(child)]


def _oracle_pick(state):
    """Independent re-derivation of ruling F: (kept, trimmed, guardrail_violator)."""
    kept, trimmed = [], []
    for leaf in state.tree.leaves():
        if leaf.parent is None:
            kept.append(leaf)
            continue
        later = [s for s in leaf.parent.children if s.seq > leaf.seq]
        if not later:
            kept.append(leaf)
            continue
        survivors_max = max(len(x.token_ids) for s in later for x in _subtree_leaves(s))
        if len(leaf.token_ids) > survivors_max:
            return None, None, leaf
        trimmed.append(leaf)
    return kept, trimmed, None


def _oracle_owner(kept):
    owner = {}
    for leaf in sorted(kept, key=lambda leaf: leaf.seq):
        for node in leaf.path_nodes():
            owner.setdefault(node.seq, leaf.seq)
    return owner


def _assert_sample_legality(sample: Sample, leaf, owner: dict[int, int]) -> None:
    snapshot = leaf.token_ids
    assert sample.tokens == snapshot[: len(sample.tokens)], "sample tokens must be a prefix of the leaf snapshot"
    assert len(sample.loss_mask) == sample.response_length
    assert len(sample.rollout_log_probs) == sample.response_length

    response_start = len(sample.tokens) - sample.response_length
    trained = {response_start + i for i, bit in enumerate(sample.loss_mask) if bit == 1}
    expected = set()
    for node in leaf.path_nodes():
        if owner[node.seq] != leaf.seq:
            continue
        lo, hi = node.completion_span
        expected |= set(range(lo, min(hi, len(sample.tokens))))
    # Mid-path early-stop (a non-COMPLETED turn ends the fold) may keep the
    # sample shorter than the snapshot; trained positions must then be the
    # owned completions clipped to the kept range.
    expected = {p for p in expected if response_start <= p < len(sample.tokens)}
    assert trained == expected, f"exactly-once violated: trained={sorted(trained)} expected={sorted(expected)}"


async def _collect(core, sid, *, max_seq_len=None, agent_metadata=None):
    response = await core.collect_samples(sid, max_seq_len=max_seq_len, agent_metadata=agent_metadata)
    return response.status_code, bytes(response.body)


def _grow_random_tree(grower, rng: random.Random, *, allow_shorter_retries: bool = False):
    """Random legal forest: extensions, retries, deep abandons, new roots.
    By default replacement branches are at least as long as what they
    supersede (retry-shaped); ``allow_shorter_retries`` drops that floor so
    guardrail-violating shapes appear and the 422 lane gets fuzzed too."""
    grower.grow(None, env_len=rng.randint(1, 3), completion_len=rng.randint(1, 4))
    for _ in range(rng.randint(2, 9)):
        state = grower.state
        op = rng.random()
        leaves = state.tree.leaves()
        if op < 0.15:  # new root (subagent)
            grower.grow(None, env_len=rng.randint(1, 3), completion_len=rng.randint(1, 4))
        elif op < 0.45:  # retry: supersede a random childless leaf with a sibling
            leaf = rng.choice(leaves)
            if leaf.parent is None:
                grower.grow(leaf, env_len=rng.randint(1, 2), completion_len=rng.randint(1, 3))
                continue
            abandoned_len = len(leaf.token_ids) - len(leaf.parent.token_ids)
            floor = 1 if allow_shorter_retries else max(1, abandoned_len)
            grower.grow(
                leaf.parent,
                env_len=rng.randint(1, 2),
                completion_len=max(floor, rng.randint(1, abandoned_len + 2)),
            )
        else:  # extend a random leaf
            grower.grow(rng.choice(leaves), env_len=rng.randint(1, 2), completion_len=rng.randint(1, 3))


@pytest.mark.parametrize("seed", range(25))
async def test_fuzz_forest_assembly_invariants(core, seed):
    sid, state, grower = await _fresh_grower(core)
    rng = random.Random(seed)
    _grow_random_tree(grower, rng)
    state.active_leaf = state.tree.leaves()[-1]

    kept, trimmed, violator = _oracle_pick(state)
    rewards = {leaf.response_id: round(rng.random(), 3) for leaf in state.tree.leaves() if rng.random() < 0.5}

    status, payload = await _collect(core, sid, agent_metadata={"rewards": rewards})
    if violator is not None:
        assert status == 422
        assert violator.response_id in payload.decode()
        return

    assert status == 200
    reply = decode_samples_reply(payload, Sample(), fields=COMPUTED_FIELDS_V2)
    assert len(reply.samples) == len(kept)
    owner = _oracle_owner(kept)
    for sample, leaf in zip(reply.samples, kept, strict=True):
        assert sample.metadata["leaf"]["node_id"] == leaf.seq  # order = leaf commit order
        _assert_sample_legality(sample, leaf, owner)
        if leaf.response_id in rewards:
            assert sample.reward == rewards[leaf.response_id]
    emitted = {s.metadata["leaf"]["node_id"] for s in reply.samples}
    assert emitted.isdisjoint({leaf.seq for leaf in trimmed})


@pytest.mark.parametrize("seed", range(50, 70))
async def test_fuzz_guardrail_shapes_fail_loud_or_assemble_legally(core, seed):
    """With shorter replacements allowed, some trees stop being retry-shaped:
    the oracle predicts either a guardrail 422 naming the violating leaf, or a
    legal assembly — never a silent wrong-side outcome."""
    sid, state, grower = await _fresh_grower(core)
    rng = random.Random(seed)
    _grow_random_tree(grower, rng, allow_shorter_retries=True)
    state.active_leaf = state.tree.leaves()[-1]

    kept, _, violator = _oracle_pick(state)
    status, payload = await _collect(core, sid)
    if violator is not None:
        assert status == 422
        assert violator.response_id in payload.decode()
        return
    assert status == 200
    reply = decode_samples_reply(payload, Sample(), fields=COMPUTED_FIELDS_V2)
    assert len(reply.samples) == len(kept)
    owner = _oracle_owner(kept)
    for sample, leaf in zip(reply.samples, kept, strict=True):
        _assert_sample_legality(sample, leaf, owner)


@pytest.mark.parametrize("seed", range(25, 35))
async def test_fuzz_forest_invariants_under_truncation(core, seed):
    sid, state, grower = await _fresh_grower(core)
    rng = random.Random(seed)
    _grow_random_tree(grower, rng)
    state.active_leaf = state.tree.leaves()[-1]

    kept, _, violator = _oracle_pick(state)
    if violator is not None:
        return  # guardrail shapes covered by the other fuzz
    max_seq_len = rng.randint(4, 12)

    status, payload = await _collect(core, sid, max_seq_len=max_seq_len)
    assert status == 200
    reply = decode_samples_reply(payload, Sample(), fields=COMPUTED_FIELDS_V2)
    if not reply.samples:
        assert reply.empty_reason == "all_truncated"
        return
    # Leaves whose every turn truncated away drop from the material; the
    # survivors' ownership is recomputed over what remains (no orphan spans).
    emitted_ids = [s.metadata["leaf"]["node_id"] for s in reply.samples]
    emitted_leaves = [next(leaf for leaf in kept if leaf.seq == node_id) for node_id in emitted_ids]
    owner = _oracle_owner(emitted_leaves)
    for sample, leaf in zip(reply.samples, emitted_leaves, strict=True):
        assert len(sample.tokens) <= max_seq_len
        _assert_sample_legality(sample, leaf, owner)


class TestTargetedEdges:
    async def test_chained_retries_single_survivor(self, core):
        sid, state, grower = await _fresh_grower(core)
        root = grower.grow(None, env_len=2, completion_len=2)
        for _ in range(3):  # three abandoned attempts, each superseded
            grower.grow(root, env_len=1, completion_len=2)
        survivor = grower.grow(root, env_len=1, completion_len=2)
        state.active_leaf = survivor

        status, payload = await _collect(core, sid)
        assert status == 200
        reply = decode_samples_reply(payload, Sample(), fields=COMPUTED_FIELDS_V2)
        (sample,) = reply.samples
        assert sample.metadata["leaf"]["node_id"] == survivor.seq

    async def test_twins_equal_length_earlier_trimmed(self, core):
        sid, state, grower = await _fresh_grower(core)
        root = grower.grow(None, env_len=2, completion_len=2)
        grower.grow(root, env_len=1, completion_len=2)
        twin_b = grower.grow(root, env_len=1, completion_len=2)
        state.active_leaf = twin_b

        status, payload = await _collect(core, sid)
        assert status == 200
        reply = decode_samples_reply(payload, Sample(), fields=COMPUTED_FIELDS_V2)
        (sample,) = reply.samples  # temporal rule: the earlier equal-length twin is trimmed
        assert sample.metadata["leaf"]["node_id"] == twin_b.seq

    async def test_fork_below_fork_ownership_nesting(self, core):
        """Two nested fork points: each shared span trains exactly once, in
        the earliest surviving leaf that contains it."""
        sid, state, grower = await _fresh_grower(core)
        root = grower.grow(None, env_len=2, completion_len=2)
        mid = grower.grow(root, env_len=1, completion_len=2)
        deep_a = grower.grow(mid, env_len=1, completion_len=2)  # earliest leaf: owns root+mid+own
        grower.grow(deep_a, env_len=1, completion_len=2)  # extend: deep_a no longer a leaf
        grower.grow(mid, env_len=1, completion_len=3)  # later sibling below mid
        side = grower.grow(root, env_len=1, completion_len=4)  # sibling below root
        state.active_leaf = side

        kept, trimmed, violator = _oracle_pick(state)
        assert violator is None
        status, payload = await _collect(core, sid)
        assert status == 200
        reply = decode_samples_reply(payload, Sample(), fields=COMPUTED_FIELDS_V2)
        assert len(reply.samples) == len(kept)
        owner = _oracle_owner(kept)
        for sample, leaf in zip(reply.samples, kept, strict=True):
            _assert_sample_legality(sample, leaf, owner)
        # Cross-check: the union of trained spans over all samples covers each
        # surviving-path completion exactly once.
        span_train_count: dict[int, int] = {}
        for sample, leaf in zip(reply.samples, kept, strict=True):
            response_start = len(sample.tokens) - sample.response_length
            trained = {response_start + i for i, bit in enumerate(sample.loss_mask) if bit == 1}
            for node in leaf.path_nodes():
                lo, hi = node.completion_span
                if set(range(lo, hi)) <= trained:
                    span_train_count[node.seq] = span_train_count.get(node.seq, 0) + 1
        surviving_nodes = {node.seq for leaf in kept for node in leaf.path_nodes()}
        assert span_train_count == {seq: 1 for seq in surviving_nodes}

    async def test_midpath_length_turn_early_stops_but_stays_legal(self, core):
        """A non-COMPLETED mid-path turn ends the fold (documented early-stop):
        the sample stops at that turn and every trained position still lies in
        an owned completion span."""
        sid, state, grower = await _fresh_grower(core)
        root = grower.grow(None, env_len=2, completion_len=2)
        cut = grower.grow(root, env_len=1, completion_len=2, finish_reason="length")
        leaf = grower.grow(cut, env_len=1, completion_len=2)
        state.active_leaf = leaf

        status, payload = await _collect(core, sid)
        assert status == 200
        reply = decode_samples_reply(payload, Sample(), fields=COMPUTED_FIELDS_V2)
        (sample,) = reply.samples
        assert len(sample.tokens) == len(cut.token_ids)  # fold stopped at the length turn
        _assert_sample_legality(sample, leaf, _oracle_owner([leaf]))
