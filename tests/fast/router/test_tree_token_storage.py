"""Token storage contract of the v2 trajectory tree.

Each node stores only its tokens past the parent prefix it reuses, as a read-only int32
array, and a committed request keeps no ``input_ids``. A session's token storage is then
linear in its final context instead of one full snapshot per turn, while every snapshot,
prompt, assembled sample and ``GET /sessions`` record reads the same tokens as before.
"""

import json
from contextlib import asynccontextmanager

import numpy as np
import orjson
import pytest
from tests.fast.router.test_session_samples_op_v2 import _ARGS

from miles.rollout.session.errors import TokenizationError
from miles.rollout.session.samples.codec import COMPUTED_FIELDS_V2, decode_samples_and_merge_input_sample
from miles.rollout.session.samples.merge import compute_samples_from_openai_records
from miles.rollout.session.types import SessionRecord
from miles.rollout.session.v2.core import SessionCoreV2
from miles.rollout.session.v2.session_state import SessionRegistryV2
from miles.rollout.session.v2.tree_trajectory import SessionTree
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer
from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample

_TURNS = 5
# Stands in for a trailing token a max_trim_tokens=1 template renders differently at the turn boundary.
_REWRITTEN_ID = 13


class _RecordingBackend:
    """Answers every chat request with the same completion and keeps the bodies it was sent."""

    def __init__(self, completion_ids: list[int]):
        self.completion_ids = completion_ids
        self.requests: list[dict] = []

    async def do_proxy(self, request, path, *, body, headers) -> dict:
        self.requests.append(orjson.loads(body))
        choice = {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
            "meta_info": {
                "output_token_logprobs": [(-0.1, token_id) for token_id in self.completion_ids],
                "completion_tokens": len(self.completion_ids),
            },
        }
        return {
            "request_body": body,
            "response_body": orjson.dumps({"id": f"chat-{len(self.requests)}", "choices": [choice]}),
            "status_code": 200,
            "headers": {"content-type": "application/json"},
        }


class _RewritingTITOTokenizer(TITOTokenizer):
    """Every continued turn rewrites the parent's last token, as a trim-tolerant template may."""

    max_trim_tokens = 1

    def merge_tokens(self, old_messages, new_messages, pretokenized_token_ids, *, template_args=None):
        merged = super().merge_tokens(old_messages, new_messages, pretokenized_token_ids, template_args=template_args)
        merged[len(pretokenized_token_ids) - 1] = _REWRITTEN_ID
        return merged


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(_ARGS.hf_checkpoint, chat_template_path=_ARGS.chat_template_path, trust_remote_code=True)


@asynccontextmanager
async def _serve(tokenizer, *, rewriting: bool):
    if rewriting:
        tito = _RewritingTITOTokenizer(tokenizer, chat_template_kwargs=_ARGS.apply_chat_template_kwargs)
    else:
        tito = get_tito_tokenizer(
            tokenizer, tokenizer_type="default", chat_template_kwargs=_ARGS.apply_chat_template_kwargs
        )
    backend = _RecordingBackend(tokenizer.encode("ok", add_special_tokens=False) + [tokenizer.eos_token_id])
    core = SessionCoreV2(backend, SessionRegistryV2(tokenizer, tito_tokenizer=tito), _ARGS, _ARGS.instance_id)
    try:
        yield core, backend
    finally:
        await core.close()


@pytest.fixture
async def default_session(tokenizer):
    async with _serve(tokenizer, rewriting=False) as served:
        yield served


@pytest.fixture
async def rewriting_session(tokenizer):
    async with _serve(tokenizer, rewriting=True) as served:
        yield served


@pytest.fixture(params=[False, True], ids=["default", "rewriting"])
async def any_session(request, tokenizer):
    async with _serve(tokenizer, rewriting=request.param) as served:
        yield served


async def _run_chain(core: SessionCoreV2, turns: int = _TURNS) -> str:
    sid = json.loads((await core.create_session()).body)["session_id"]
    messages = [{"role": "user", "content": "hi"}]
    for turn in range(turns):
        response = await core.chat_completions(
            sid, method="POST", query="", headers={}, body=json.dumps({"messages": messages}).encode()
        )
        assert response.status_code == 200, response.body
        assistant = json.loads(response.body)["choices"][0]["message"]
        messages = messages + [assistant, {"role": "user", "content": f"turn {turn}"}]
    return sid


def _leaf(core: SessionCoreV2, sid: str):
    return core.registry.get_session(sid).latest()


async def test_token_storage_is_linear_in_final_context(default_session):
    core, backend = default_session
    sid = await _run_chain(core)
    leaf = _leaf(core, sid)
    path = leaf.path_nodes()
    assert len(path) == _TURNS

    # Four bytes per token of the FINAL context, not per turn x context.
    assert sum(node.token_delta.nbytes for node in path) == 4 * leaf.num_tokens
    for parent, child in zip([None, *path], path, strict=False):
        assert child.token_delta.dtype == np.int32 and not child.token_delta.flags.writeable
        assert child.parent_reuse == (parent.num_tokens if parent is not None else 0)

    leaf_token_ids = leaf.token_ids
    for node, sent in zip(path, backend.requests, strict=True):
        # Every snapshot is a prefix of its leaf's, and holds exactly the prompt the backend received.
        assert node.token_ids == leaf_token_ids[: node.num_tokens]
        assert node.token_ids[: node.completion_span[0]] == sent["input_ids"]
        # Neither the record nor turn_args repeats the prompt the snapshot holds.
        assert "input_ids" not in node.record.request
        assert "input_ids" not in node.turn_args


async def test_get_session_serves_each_request_as_sent(default_session):
    core, backend = default_session
    sid = await _run_chain(core)

    payload = json.loads((await core.get_session(sid)).body)
    assert [record["request"] for record in payload["records"]] == backend.requests
    assert payload["metadata"]["accumulated_token_ids"] == _leaf(core, sid).token_ids
    assert payload["metadata"]["tree"]["nodes"][-1]["num_tokens"] == _leaf(core, sid).num_tokens


async def test_collect_assembles_what_records_with_input_ids_assemble(any_session):
    """Prompts sliced from the leaf assemble the same turns as records carrying their own input_ids."""
    core, backend = any_session
    sid = await _run_chain(core)
    leaf = _leaf(core, sid)
    records = [
        record.model_copy(update={"request": {**record.request, "input_ids": sent["input_ids"]}})
        for record, sent in zip((node.record for node in leaf.path_nodes()), backend.requests, strict=True)
    ]
    expected = compute_samples_from_openai_records(
        _ARGS,
        records,
        core.registry.tokenizer,
        accumulated_token_ids=leaf.token_ids,
        max_trim_tokens=core.registry.tito_tokenizer.max_trim_tokens,
    )
    derived = compute_samples_from_openai_records(
        _ARGS,
        [node.record for node in leaf.path_nodes()],
        core.registry.tokenizer,
        accumulated_token_ids=leaf.token_ids,
        max_trim_tokens=core.registry.tito_tokenizer.max_trim_tokens,
        prompt_lengths=[node.completion_span[0] for node in leaf.path_nodes()],
    )
    assert [(s.tokens, s.loss_mask, s.rollout_log_probs) for s in derived] == [
        (s.tokens, s.loss_mask, s.rollout_log_probs) for s in expected
    ]

    response = await core.collect_samples(sid, max_seq_len=None)
    assert response.status_code == 200, response.body
    (sample,) = decode_samples_and_merge_input_sample(response.body, Sample(), fields=COMPUTED_FIELDS_V2).samples
    assert sample.tokens == leaf.token_ids


async def test_boundary_rewrites_live_in_the_child_delta(rewriting_session):
    core, backend = rewriting_session
    sid = await _run_chain(core)
    path = _leaf(core, sid).path_nodes()
    for parent, child in zip(path, path[1:], strict=False):
        assert child.parent_reuse == parent.num_tokens - 1
        assert child.token_ids[parent.num_tokens - 1] == _REWRITTEN_ID != parent.token_ids[-1]
    # One shadowed token per continued turn on top of the final context.
    assert sum(node.token_delta.size for node in path) == path[-1].num_tokens + len(path) - 1
    payload = json.loads((await core.get_session(sid)).body)
    assert [record["request"]["input_ids"] for record in payload["records"]] == [
        sent["input_ids"] for sent in backend.requests
    ]


def _record() -> SessionRecord:
    return SessionRecord(
        timestamp=0.0, method="POST", path="/v1/chat/completions", request={}, response={}, status_code=200
    )


def _create(tree: SessionTree, parent, token_ids: list[int], completion_span: tuple[int, int]):
    return tree.create_node(
        parent,
        delta_messages=[],
        token_ids=token_ids,
        completion_span=completion_span,
        committed_at=0.0,
        response_id="",
        record=_record(),
        finish_reason="stop",
    )


def test_sibling_rewrites_each_keep_their_own_view():
    """Two children rewriting the same parent tail differently: each snapshot keeps its own
    rewrite, and the parent keeps the token it committed."""
    tree = SessionTree()
    parent = _create(tree, None, [1, 2, 3, 9], (3, 4))
    left = _create(tree, parent, [1, 2, 3, 70, 5, 6], (5, 6))
    right = _create(tree, parent, [1, 2, 3, 80, 7, 8], (5, 6))

    assert left.parent_reuse == right.parent_reuse == 3
    assert parent.token_ids == [1, 2, 3, 9]
    assert left.token_ids == [1, 2, 3, 70, 5, 6]
    assert right.token_ids == [1, 2, 3, 80, 7, 8]
    assert [node.token_delta.tolist() for node in tree.nodes] == [[1, 2, 3, 9], [70, 5, 6], [80, 7, 8]]


@pytest.mark.parametrize(
    ("token_ids", "match"),
    [
        ([1, 2, 7, 9, 5], "reaches into the parent's prompt"),
        ([1, 2, 3], "shorter than its parent"),
        ([1, 2, 3, 9, 2**31], "outside int32"),
    ],
)
def test_snapshots_that_break_the_storage_contract_are_refused(token_ids, match):
    tree = SessionTree()
    parent = _create(tree, None, [1, 2, 3, 9], (3, 4))
    with pytest.raises(TokenizationError, match=match):
        _create(tree, parent, token_ids, (len(token_ids) - 1, len(token_ids)))
    assert len(tree.nodes) == 1 and not parent.children
