"""Images through the v2 session server end to end, with a stub engine:
the request's data URLs are stored once and referenced as ``mm://`` in the
session's records, the engine receives the client's data URLs with the
pre-tokenized ids (one placeholder per image), and the collected samples
carry the expanded prompt plus the images in the trainer's PNG format."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import threading
import uuid

import numpy as np
import pytest
import torch
from tests.fast.fixtures.session_fixtures import make_session_server_config
from tests.fast.rollout.session.test_media import FakeProcessor, _data_url, _png

from miles.rollout.session.media import MEDIA_URL_PREFIX
from miles.rollout.session.samples.codec import COMPUTED_FIELDS_V2, decode_samples_and_merge_input_sample
from miles.rollout.session.v2.core import SessionCoreV2
from miles.rollout.session.v2.session_state import SessionRegistryV2
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.types import Sample

MODEL = "Qwen/Qwen3.8-27B"


@pytest.fixture(scope="module")
def tokenizer():
    """The Qwen3.8 tokenizer with the qwen38small fixed template installed, as
    the session server does through ``--chat-template-path``."""
    from transformers import AutoTokenizer

    from miles.utils.chat_template_utils.tito_tokenizer import TEMPLATE_DIR, Qwen38SmallTITOTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(MODEL)
    except Exception as e:  # noqa: BLE001 - no network or no cache
        pytest.skip(f"{MODEL} tokenizer unavailable: {e}")
    tok.chat_template = (TEMPLATE_DIR / Qwen38SmallTITOTokenizer.FIXED_TEMPLATE.template).read_text(encoding="utf-8")
    return tok


GLM_MODEL = os.environ.get("MILES_TEST_GLM53_TOKENIZER", "zai-org/GLM-5.3-Flash")


@pytest.fixture(scope="module")
def glm_tokenizer():
    """The GLM-5.3 tokenizer with its native template (the glm53 family fixes no template)."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(GLM_MODEL, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001 - no network or no cache
        pytest.skip(f"{GLM_MODEL} tokenizer unavailable: {e}")


class ExactTokenEngine:
    """A chat backend that behaves like SGLang's exact-token path: it runs the
    request ids as sent, expands each image placeholder to the processor's
    count, and reports that prompt length."""

    def __init__(self, tokenizer, pad_id: int, counts: dict[str, int], reply: str, end_token: str = "<|im_end|>"):
        self.tokenizer = tokenizer
        self.pad_id = pad_id
        self.counts = counts  # data URL -> token count
        self.reply = reply
        self.end_token = end_token  # the family's assistant stop token
        self.bodies: list[dict] = []

    async def do_proxy(self, request, path, *, body, headers):
        payload = json.loads(body)
        self.bodies.append(payload)
        input_ids = payload["input_ids"]
        urls = [
            part["image_url"]["url"]
            for message in payload["messages"]
            if isinstance(message.get("content"), list)
            for part in message["content"]
            if part.get("type") == "image_url"
        ]
        assert all(url.startswith("data:") for url in urls), "the engine must receive data URLs"
        prompt_tokens = len(input_ids) - len(urls) + sum(self.counts[url] for url in urls)
        output_ids = self.tokenizer.encode(self.reply, add_special_tokens=False) + [
            self.tokenizer.convert_tokens_to_ids(self.end_token)
        ]
        response = {
            "id": f"stub-{len(self.bodies)}",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": self.reply},
                    "finish_reason": "stop",
                    "meta_info": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": len(output_ids),
                        "output_token_logprobs": [[-0.1, tid] for tid in output_ids],
                    },
                }
            ],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(output_ids)},
        }
        return {
            "status_code": 200,
            "response_body": json.dumps(response),
            "headers": {"content-type": "application/json"},
        }


def _chat(core, session_id, messages):
    body = json.dumps({"model": "default", "messages": messages, "max_tokens": 8}).encode()
    return asyncio.run(core.chat_completions(session_id, method="POST", query="", headers={}, body=body))


def test_images_flow_through_the_v2_session_server(tokenizer):
    tito = get_tito_tokenizer(tokenizer, tokenizer_type="qwen38small")
    pad_id = tito.image_pad_id()
    assert pad_id == tokenizer.convert_tokens_to_ids("<|image_pad|>")
    first_url, second_url = _data_url(_png()), _data_url(_png(width=128, color=(5, 5, 5)))
    engine = ExactTokenEngine(tokenizer, pad_id, {first_url: 4, second_url: 8}, reply="ok")
    config = make_session_server_config(
        use_session_server="v2",
        hf_checkpoint=MODEL,
        tito_model="qwen38small",
        instance_id=uuid.uuid4().hex,
        session_sample_picker_path="miles.rollout.session.v2.picker_hub.drop_rolled_back_leaves",
        session_sample_postprocessor_path="miles.rollout.session.v2.postprocessor_hub.default_postprocess",
    )
    registry = SessionRegistryV2(tokenizer, tito_tokenizer=tito, processor=FakeProcessor())
    core = SessionCoreV2(engine, registry, config, config.instance_id)
    session_id = registry.create_session()

    turn1 = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": first_url}}],
        }
    ]
    response = _chat(core, session_id, turn1)
    assert response.status_code == 200, response.body
    assistant = json.loads(response.body)["choices"][0]["message"]
    # The Qwen3.8 template requires text in a user turn, so a screenshot rides with its note.
    turn2 = turn1 + [
        assistant,
        {
            "role": "user",
            "content": [{"type": "text", "text": "next"}, {"type": "image_url", "image_url": {"url": second_url}}],
        },
    ]
    assert _chat(core, session_id, turn2).status_code == 200

    # The engine got one placeholder per image and the client's bytes; the
    # session kept each image once as a reference.
    assert [body["input_ids"].count(pad_id) for body in engine.bodies] == [1, 2]
    session = registry.get_session(session_id)
    assert len(session.media.entries) == 2
    stored = session.latest().path_messages()
    refs = [
        p["image_url"]["url"]
        for m in stored
        if isinstance(m.get("content"), list)
        for p in m["content"]
        if p.get("type") == "image_url"
    ]
    assert len(refs) == 2 and all(ref.startswith(MEDIA_URL_PREFIX) for ref in refs)
    assert all(
        ref.startswith(MEDIA_URL_PREFIX)
        for node in session.tree.nodes
        for ref in [
            p["image_url"]["url"]
            for m in node.record.request["messages"]
            if isinstance(m.get("content"), list)
            for p in m["content"]
            if p.get("type") == "image_url"
        ]
    )

    # The samples carry the expanded prompt and the images.
    reply = asyncio.run(core.collect_samples(session_id, max_seq_len=None))
    assert reply.status_code == 200, reply.body
    decoded = decode_samples_and_merge_input_sample(reply.body, Sample(), fields=COMPUTED_FIELDS_V2)
    (sample,) = decoded.samples
    wire = engine.bodies[1]["input_ids"]
    reply_len = len(tokenizer.encode("ok", add_special_tokens=False)) + 1  # plus <|im_end|>
    # Turn 2's wire prompt holds 2 placeholders; expanded to 4 + 8 tokens, plus the second reply.
    assert sample.tokens.count(pad_id) == 12 and len(sample.tokens) == len(wire) - 2 + 12 + reply_len
    mm = sample.multimodal_train_inputs
    assert isinstance(mm["image_png"], torch.Tensor)
    assert mm["image_grid_thw"].tolist() == [[1, 4, 4], [1, 4, 8]]
    assert mm["image_png_lengths"].sum().item() == mm["image_png"].numel()
    assert sample.loss_mask.count(1) == 2 * reply_len, "only the two sampled replies are trained"


async def test_the_samples_worker_builds_from_the_images_of_its_snapshot(tokenizer, monkeypatch):
    """The samples op snapshots the session under its lock and builds in a
    worker thread. A chat that lands during the build registers a new image
    in the live store; the build must still see exactly the snapshot's
    images and return the same samples as a build with no chat in between."""
    tito = get_tito_tokenizer(tokenizer, tokenizer_type="qwen38small")
    pad_id = tito.image_pad_id()
    first_url, second_url = _data_url(_png()), _data_url(_png(width=128, color=(5, 5, 5)))
    engine = ExactTokenEngine(tokenizer, pad_id, {first_url: 4, second_url: 8}, reply="ok")
    config = make_session_server_config(
        use_session_server="v2",
        hf_checkpoint=MODEL,
        tito_model="qwen38small",
        instance_id=uuid.uuid4().hex,
        session_sample_picker_path="miles.rollout.session.v2.picker_hub.drop_rolled_back_leaves",
        session_sample_postprocessor_path="miles.rollout.session.v2.postprocessor_hub.default_postprocess",
    )
    registry = SessionRegistryV2(tokenizer, tito_tokenizer=tito, processor=FakeProcessor())
    core = SessionCoreV2(engine, registry, config, config.instance_id)
    session_id = registry.create_session()
    turn1 = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": first_url}}],
        }
    ]

    async def chat(messages):
        body = json.dumps({"model": "default", "messages": messages, "max_tokens": 8}).encode()
        return await core.chat_completions(session_id, method="POST", query="", headers={}, body=body)

    first = await chat(turn1)
    assert first.status_code == 200, first.body
    expected = await core.collect_samples(session_id, max_seq_len=None)
    assert expected.status_code == 200, expected.body

    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    collect = core._collect_samples
    seen = []

    def blocked(sid, snapshot, **kwargs):
        seen.append(snapshot)
        loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=5)
        return collect(sid, snapshot, **kwargs)

    monkeypatch.setattr(core, "_collect_samples", blocked)
    task = asyncio.create_task(core.collect_samples(session_id, max_seq_len=None))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        assistant = json.loads(first.body)["choices"][0]["message"]
        turn2 = turn1 + [
            assistant,
            {
                "role": "user",
                "content": [{"type": "text", "text": "next"}, {"type": "image_url", "image_url": {"url": second_url}}],
            },
        ]
        assert (await chat(turn2)).status_code == 200
    finally:
        release.set()
        response = await task
    await core.close()

    (snapshot,) = seen
    live = registry.get_session(session_id).media
    assert len(live.entries) == 2 and len(snapshot.media.entries) == 1
    assert snapshot.media is not live
    assert response.status_code == 200, response.body
    assert response.body == expected.body
    (sample,) = decode_samples_and_merge_input_sample(response.body, Sample(), fields=COMPUTED_FIELDS_V2).samples
    assert sample.multimodal_train_inputs["image_grid_thw"].tolist() == [[1, 4, 4]]


class RoutedExpertsEngine(ExactTokenEngine):
    """The exact-token engine with routed-experts capture: a request with
    ``return_routed_experts`` gets rows [routed_experts_start_len, seqlen - 1)
    in its meta_info, counted on the expanded ids it ran, each row holding its
    absolute position so a sample's rows can be checked against its tokens."""

    LAYERS, TOPK = 2, 2

    async def do_proxy(self, request, path, *, body, headers):
        result = await super().do_proxy(request, path, body=body, headers=headers)
        payload = self.bodies[-1]
        if not payload.get("return_routed_experts"):
            return result
        response = json.loads(result["response_body"])
        meta = response["choices"][0]["meta_info"]
        seqlen = meta["prompt_tokens"] + meta["completion_tokens"]
        start = payload.get("routed_experts_start_len", 0)
        rows = np.arange(start, seqlen - 1, dtype=np.int32)
        payload_rows = np.repeat(rows[:, None, None], self.LAYERS * self.TOPK, axis=1).reshape(
            -1, self.LAYERS, self.TOPK
        )
        meta["routed_experts"] = base64.b64encode(np.ascontiguousarray(payload_rows).tobytes()).decode("ascii")
        result["response_body"] = json.dumps(response)
        return result


def test_addition_r3_rows_follow_the_expanded_image_tokens_across_turns_and_branches(tokenizer):
    """Through the v2 server with a non-retract pause mode: every turn asks the
    engine for the rows after the checkpoint's expanded length (one placeholder
    per image on the wire, 4 or 8 tokens where the engine runs it), a branch
    off an earlier node starts at that node's expanded length, and each leaf's
    sample reassembles exactly len(tokens) - 1 rows whose positions match."""
    tito = get_tito_tokenizer(tokenizer, tokenizer_type="qwen38small")
    pad_id = tito.image_pad_id()
    first_url, second_url, third_url = (
        _data_url(_png()),
        _data_url(_png(width=128, color=(5, 5, 5))),
        _data_url(_png(width=128, color=(9, 9, 9))),
    )
    engine = RoutedExpertsEngine(tokenizer, pad_id, {first_url: 4, second_url: 8, third_url: 8}, reply="ok")
    config = make_session_server_config(
        use_session_server="v2",
        hf_checkpoint=MODEL,
        tito_model="qwen38small",
        instance_id=uuid.uuid4().hex,
        use_rollout_routing_replay=True,
        pause_generation_mode="in_place",
        num_layers=RoutedExpertsEngine.LAYERS,
        moe_router_topk=RoutedExpertsEngine.TOPK,
        # The fold picker: a sibling with a different prompt is a branch and
        # both leaves stay (drop_rolled_back_leaves would trim the pre-fold leaf).
        session_sample_picker_path="miles.rollout.session.v2.picker_hub.drop_same_prompt_retries",
        session_sample_postprocessor_path="miles.rollout.session.v2.postprocessor_hub.default_postprocess",
    )
    registry = SessionRegistryV2(tokenizer, tito_tokenizer=tito, processor=FakeProcessor())
    core = SessionCoreV2(engine, registry, config, config.instance_id, use_addition_r3=True)
    session_id = registry.create_session()

    def user(text, url):
        return {
            "role": "user",
            "content": [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": url}}],
        }

    turn1 = [user("look", first_url)]
    response = _chat(core, session_id, turn1)
    assert response.status_code == 200, response.body
    assistant = json.loads(response.body)["choices"][0]["message"]
    assert engine.bodies[0]["return_routed_experts"] is True and engine.bodies[0]["routed_experts_start_len"] == 0
    node1 = registry.get_session(session_id).latest()
    checkpoint1_expanded = len(node1.token_ids) - 1 + 4

    assert _chat(core, session_id, turn1 + [assistant, user("next", second_url)]).status_code == 200
    assert engine.bodies[1]["routed_experts_start_len"] == checkpoint1_expanded - 1
    # A branch off the turn-1 node (a fold rewrites the second user turn: a
    # different note and screenshot, so a different prompt, not a retry): the
    # offset returns to that node's expanded length.
    assert _chat(core, session_id, turn1 + [assistant, user("after the fold", third_url)]).status_code == 200
    assert engine.bodies[2]["routed_experts_start_len"] == checkpoint1_expanded - 1

    reply = asyncio.run(core.collect_samples(session_id, max_seq_len=None))
    assert reply.status_code == 200, reply.body
    decoded = decode_samples_and_merge_input_sample(reply.body, Sample(), fields=COMPUTED_FIELDS_V2)
    assert len(decoded.samples) == 2, "one sample per leaf"
    for sample in decoded.samples:
        rows = sample.rollout_routed_experts
        assert rows.shape == (len(sample.tokens) - 1, RoutedExpertsEngine.LAYERS, RoutedExpertsEngine.TOPK)
        # Rows carry their absolute position: a correct assembly is 0, 1, 2, ...
        assert rows[:, 0, 0].tolist() == list(range(len(sample.tokens) - 1))
        assert sample.tokens.count(pad_id) == 12
        sample.validate()


def test_an_engine_that_retokenizes_is_refused_at_sample_time(tokenizer):
    tito = get_tito_tokenizer(tokenizer, tokenizer_type="qwen38small")
    url = _data_url(_png())
    engine = ExactTokenEngine(tokenizer, tito.image_pad_id(), {url: 3}, reply="ok")  # processor says 4
    config = make_session_server_config(
        use_session_server="v2",
        hf_checkpoint=MODEL,
        tito_model="qwen38small",
        instance_id=uuid.uuid4().hex,
        session_sample_picker_path="miles.rollout.session.v2.picker_hub.drop_rolled_back_leaves",
        session_sample_postprocessor_path="miles.rollout.session.v2.postprocessor_hub.default_postprocess",
    )
    registry = SessionRegistryV2(tokenizer, tito_tokenizer=tito, processor=FakeProcessor())
    core = SessionCoreV2(engine, registry, config, config.instance_id)
    session_id = registry.create_session()
    message = {
        "role": "user",
        "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": url}}],
    }
    assert _chat(core, session_id, [message]).status_code == 200
    reply = asyncio.run(core.collect_samples(session_id, max_seq_len=None))
    assert reply.status_code == 422 and b"did not run the request ids as sent" in reply.body


def test_a_text_only_server_refuses_images(tokenizer):
    tito = get_tito_tokenizer(tokenizer, tokenizer_type="qwen38small")
    config = make_session_server_config(
        use_session_server="v2",
        hf_checkpoint=MODEL,
        tito_model="qwen38small",
        instance_id=uuid.uuid4().hex,
        session_sample_picker_path="miles.rollout.session.v2.picker_hub.drop_rolled_back_leaves",
        session_sample_postprocessor_path="miles.rollout.session.v2.postprocessor_hub.default_postprocess",
    )
    registry = SessionRegistryV2(tokenizer, tito_tokenizer=tito, processor=None)
    engine = ExactTokenEngine(tokenizer, tito.image_pad_id(), {}, reply="ok")
    core = SessionCoreV2(engine, registry, config, config.instance_id)
    session_id = registry.create_session()
    from miles.rollout.session.errors import MessageValidationError

    with pytest.raises(MessageValidationError, match="cannot carry images"):
        _chat(
            core,
            session_id,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {"type": "image_url", "image_url": {"url": _data_url(_png())}},
                    ],
                }
            ],
        )


def _image_urls_of(messages):
    return [
        p["image_url"]["url"]
        for m in messages
        if isinstance(m.get("content"), list)
        for p in m["content"]
        if p.get("type") == "image_url"
    ]


def test_images_flow_through_the_v2_session_server_for_glm53(glm_tokenizer):
    """The second family in the image-token table: GLM-5.3 writes
    <|begin_of_image|><|image|><|end_of_image|> per image; one <|image|> goes
    on the wire per image, the record expands it between the markers."""
    tito = get_tito_tokenizer(glm_tokenizer, tokenizer_type="glm53")
    ids = tito.image_token_ids()
    begin_id, pad_id, end_id = ids["begin"], ids["placeholder"], ids["end"]
    assert pad_id == glm_tokenizer.convert_tokens_to_ids("<|image|>")
    first_url, second_url = _data_url(_png()), _data_url(_png(width=128, color=(5, 5, 5)))
    engine = ExactTokenEngine(glm_tokenizer, pad_id, {first_url: 4, second_url: 8}, reply="ok", end_token="<|user|>")
    config = make_session_server_config(
        use_session_server="v2",
        hf_checkpoint=GLM_MODEL,
        tito_model="glm53",
        instance_id=uuid.uuid4().hex,
        session_sample_picker_path="miles.rollout.session.v2.picker_hub.drop_rolled_back_leaves",
        session_sample_postprocessor_path="miles.rollout.session.v2.postprocessor_hub.default_postprocess",
    )
    registry = SessionRegistryV2(glm_tokenizer, tito_tokenizer=tito, processor=FakeProcessor())
    core = SessionCoreV2(engine, registry, config, config.instance_id)
    session_id = registry.create_session()

    turn1 = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": first_url}}],
        }
    ]
    response = _chat(core, session_id, turn1)
    assert response.status_code == 200, response.body
    assistant = json.loads(response.body)["choices"][0]["message"]
    turn2 = turn1 + [
        assistant,
        {
            "role": "user",
            "content": [{"type": "text", "text": "next"}, {"type": "image_url", "image_url": {"url": second_url}}],
        },
    ]
    assert _chat(core, session_id, turn2).status_code == 200

    # one placeholder per image on the wire, each between its markers
    wires = [body["input_ids"] for body in engine.bodies]
    assert [wire.count(pad_id) for wire in wires] == [1, 2]
    for wire in wires:
        for position in [i for i, t in enumerate(wire) if t == pad_id]:
            assert wire[position - 1] == begin_id and wire[position + 1] == end_id
    assert all(url.startswith("data:") for body in engine.bodies for url in _image_urls_of(body["messages"]))
    session = registry.get_session(session_id)
    assert len(session.media.entries) == 2

    # the sample expands each placeholder to the processor's count, markers kept
    reply = asyncio.run(core.collect_samples(session_id, max_seq_len=None))
    assert reply.status_code == 200, reply.body
    decoded = decode_samples_and_merge_input_sample(reply.body, Sample(), fields=COMPUTED_FIELDS_V2)
    (sample,) = decoded.samples
    wire = wires[1]
    reply_len = len(glm_tokenizer.encode("ok", add_special_tokens=False)) + 1  # plus <|user|>
    assert sample.tokens.count(pad_id) == 12 and len(sample.tokens) == len(wire) - 2 + 12 + reply_len
    runs = [i for i, t in enumerate(sample.tokens) if t == pad_id]
    assert sample.tokens[runs[0] - 1] == begin_id and sample.tokens[runs[3] + 1] == end_id
    assert sample.tokens[runs[4] - 1] == begin_id and sample.tokens[runs[11] + 1] == end_id
    mm = sample.multimodal_train_inputs
    assert mm["image_grid_thw"].tolist() == [[1, 4, 4], [1, 4, 8]]
    assert mm["image_png_lengths"].sum().item() == mm["image_png"].numel()


def test_a_family_without_a_table_entry_refuses_images_and_names_the_table(tokenizer):
    """A vision checkpoint served with a TITO family that has no ImageTokens
    entry (here: qwen3, a text family, on the Qwen3.8 tokenizer) refuses the
    request instead of sending expanded ids the engine would re-tokenize."""
    tito = get_tito_tokenizer(tokenizer, tokenizer_type="qwen3")
    assert tito.image_tokens is None
    engine = ExactTokenEngine(tokenizer, 0, {}, reply="ok")
    config = make_session_server_config(
        use_session_server="v2",
        hf_checkpoint=MODEL,
        tito_model="qwen3",
        instance_id=uuid.uuid4().hex,
        session_sample_picker_path="miles.rollout.session.v2.picker_hub.drop_rolled_back_leaves",
        session_sample_postprocessor_path="miles.rollout.session.v2.postprocessor_hub.default_postprocess",
    )
    registry = SessionRegistryV2(tokenizer, tito_tokenizer=tito, processor=FakeProcessor())
    core = SessionCoreV2(engine, registry, config, config.instance_id)
    session_id = registry.create_session()
    from miles.rollout.session.errors import MessageValidationError

    url = _data_url(_png())
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": url}}],
        }
    ]
    with pytest.raises(MessageValidationError, match="image-token table") as info:
        _chat(core, session_id, messages)
    assert "Qwen3TITOTokenizer" in str(info.value)
    assert engine.bodies == [], "nothing reached the engine"
