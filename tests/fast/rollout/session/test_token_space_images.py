"""Images through the session server with --multimodal-token-space-processing.

The stub engine expands media with SGLang's own token-space processor (the Qwen VL strategy, on CPU),
as a server with --enable-token-space-processor does. The tests check what each turn sends, what the
session stores, and that the trainer rebuilds the engine's pixel values from the collected sample.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
import uuid

import pytest
import torch
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.session.core import SessionCore
from miles.rollout.session.errors import MessageValidationError, TokenizationError, UpstreamResponseError
from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.rollout.session.samples.codec import (
    COMPUTED_FIELDS,
    COMPUTED_FIELDS_V2,
    decode_samples_and_merge_input_sample,
)
from miles.rollout.session.token_space import engine_input_ids
from miles.rollout.session.v2.core import SessionCoreV2
from miles.rollout.session.v2.session_state import SessionRegistryV2
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.types import Sample

MODEL = "Qwen/Qwen3.8-27B"
IMAGE_PAD = "<|image_pad|>"


class TestEngineInputIds:
    # A checkpoint with prompt [1, 9, 2] (9 expands to three 9s in the engine's form) and completion [5, 6].
    CHECKPOINT = [1, 9, 2, 5, 6]
    EXPANDED = [1, 9, 9, 9, 2, 5, 6]

    def test_the_first_turn_is_sent_as_rendered(self):
        assert engine_input_ids([1, 2, 9, 3], [], [], 0) == ([1, 2, 9, 3], 0)

    def test_the_checkpoint_is_sent_expanded_and_the_new_part_as_rendered(self):
        # Qwen3 appends the newline its stop token lacks, then the new turn with one more image.
        prompt = self.CHECKPOINT + [7, 4, 9, 8]
        assert engine_input_ids(prompt, self.CHECKPOINT, self.EXPANDED, 2) == (self.EXPANDED + [7, 4, 9, 8], 7)

    def test_a_trimmed_trailing_token_is_sent_as_rendered(self):
        # GLM re-renders the ambiguous last token of the completion.
        prompt = [1, 9, 2, 5, 4, 9]
        assert engine_input_ids(prompt, self.CHECKPOINT, self.EXPANDED, 2) == ([1, 9, 9, 9, 2, 5, 4, 9], 6)

    def test_a_prompt_that_cuts_into_the_checkpoint_prompt_raises(self):
        # The engine tails coincide here ([9, 2] in both), so only the completion bound catches it.
        with pytest.raises(TokenizationError, match="cuts into its prompt"):
            engine_input_ids([1, 4], [1, 9, 2], [1, 9, 9, 2], 0)


@pytest.fixture(scope="module")
def model_path():
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(MODEL, local_files_only=True)
    except Exception as e:  # noqa: BLE001 - no cache
        pytest.skip(f"{MODEL} is not in the HF cache: {e}")


@pytest.fixture(scope="module")
def tokenizer(model_path):
    """The Qwen3.8 tokenizer with the qwen38small fixed template, as ``--chat-template-path`` installs it."""
    from transformers import AutoTokenizer

    from miles.utils.chat_template_utils.tito_tokenizer import TEMPLATE_DIR, Qwen38SmallTITOTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    tok.chat_template = (TEMPLATE_DIR / Qwen38SmallTITOTokenizer.FIXED_TEMPLATE.template).read_text(encoding="utf-8")
    return tok


@pytest.fixture(scope="module")
def engine_args(model_path):
    pytest.importorskip("sglang.srt.multimodal.token_space_mm_process.processor")
    from miles.backends.training_utils.data.token_space_media_process import RolloutEngineProcessorArgs

    return RolloutEngineProcessorArgs(
        model_path=model_path,
        tokenizer_path=model_path,
        tokenizer_mode="auto",
        tokenizer_backend="huggingface",
        revision=None,
        model_config_parser="auto",
        model_override_args_json="{}",
        image_processor_backend="pil",
        disable_fast_image_processor=False,
        mm_process_config_json="{}",
    )


@pytest.fixture(scope="module")
def processor_core(engine_args):
    """SGLang's token-space processor for the model, built the way the trainer builds it."""
    from miles.backends.training_utils.data.token_space_media_process import _trainer_processor

    return _trainer_processor(engine_args, torch.device("cpu")).core


def _data_url(width: int, height: int, color: tuple[int, int, int]) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _image_urls(messages: list[dict]) -> list[str]:
    return [
        part["image_url"]["url"]
        for message in messages
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") == "image_url"
    ]


def _user(text: str, url: str | None = None) -> dict:
    content = [{"type": "text", "text": text}]
    if url is not None:
        content.append({"type": "image_url", "image_url": {"url": url}})
    return {"role": "user", "content": content}


class TokenSpaceEngine:
    """A chat backend that expands media like SGLang with --enable-token-space-processor: the message
    media and the request's ``input_ids`` go through the token-space processor, from
    ``mm_token_expansion_start_len`` on, and the expanded prompt is returned when asked for."""

    def __init__(self, core, tokenizer, *, reply: str = "ok", return_prompt: bool = True):
        self.core = core
        self.reply = reply
        self.output_ids = tokenizer.encode(reply, add_special_tokens=False) + [
            tokenizer.convert_tokens_to_ids("<|im_end|>")
        ]
        self.return_prompt = return_prompt
        self.bodies: list[dict] = []
        self.pixel_values: list[torch.Tensor] = []
        self.prompts: list[list[int]] = []  # the prompt each request ran

    async def do_proxy(self, request, path, *, body, headers):
        payload = json.loads(body)
        self.bodies.append(payload)
        prompt_ids = payload["input_ids"]
        paths = _image_urls(payload["messages"])
        if paths:
            output = await self.core.process(
                image_data=paths,
                input_ids=prompt_ids,
                mm_token_expansion_start_len=payload.get("mm_token_expansion_start_len", 0),
            )
            prompt_ids = output["input_ids"][0].tolist()
            self.pixel_values.append(output["pixel_values"])
        self.prompts.append(prompt_ids)
        choice = {
            "index": 0,
            "message": {"role": "assistant", "content": self.reply},
            "finish_reason": "stop",
            "meta_info": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(self.output_ids),
                "output_token_logprobs": [[-0.1, token_id] for token_id in self.output_ids],
            },
        }
        if payload.get("return_prompt_token_ids") and self.return_prompt:
            choice["prompt_token_ids"] = prompt_ids
        response = {
            "id": f"stub-{len(self.bodies)}",
            "object": "chat.completion",
            "choices": [choice],
            "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(self.output_ids)},
        }
        return {
            "status_code": 200,
            "response_body": json.dumps(response),
            "headers": {"content-type": "application/json"},
        }


def _make_core(version: str, tokenizer, engine, media_dir, *, token_space: bool = True):
    tito = get_tito_tokenizer(tokenizer, tokenizer_type="qwen38small")
    config = make_session_server_config(
        use_session_server=version,
        hf_checkpoint=MODEL,
        tito_model="qwen38small",
        instance_id=uuid.uuid4().hex,
        session_sample_picker_path="miles.rollout.session.v2.picker_hub.drop_rolled_back_leaves",
        session_sample_postprocessor_path="miles.rollout.session.v2.postprocessor_hub.default_postprocess",
        multimodal_token_space_processing=token_space,
        shared_media_dir=str(media_dir),
    )
    if version == "v2":
        registry = SessionRegistryV2(tokenizer, tito_tokenizer=tito)
        return SessionCoreV2(engine, registry, config, config.instance_id), registry
    registry = SessionRegistry(tokenizer, tito_tokenizer=tito)
    return SessionCore(engine, registry, config, config.instance_id), registry


async def _chat(core, session_id: str, messages: list[dict]):
    body = json.dumps({"model": "default", "messages": messages, "max_tokens": 8}).encode()
    return await core.chat_completions(session_id, method="POST", query="", headers={}, body=body)


def _samples(core, session_id: str, version: str) -> list[Sample]:
    reply = asyncio.run(core.collect_samples(session_id, max_seq_len=None))
    assert reply.status_code == 200, reply.body
    fields = COMPUTED_FIELDS_V2 if version == "v2" else COMPUTED_FIELDS
    return decode_samples_and_merge_input_sample(reply.body, Sample(), fields=fields).samples


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_images_flow_through_the_session_server_in_token_space(
    version, tokenizer, processor_core, engine_args, tmp_path
):
    from miles.backends.training_utils.data.token_space_media_process import compute_multimodal_train_inputs

    pad_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD)
    engine = TokenSpaceEngine(processor_core, tokenizer)
    core, registry = _make_core(version, tokenizer, engine, tmp_path)
    session_id = registry.create_session()
    first, second = _data_url(64, 64, (200, 10, 10)), _data_url(96, 64, (5, 5, 5))

    turn1 = [_user("look", first)]
    response = asyncio.run(_chat(core, session_id, turn1))
    assert response.status_code == 200, response.body
    choice = json.loads(response.body)["choices"][0]
    assert "prompt_token_ids" not in choice, "the client did not ask for the prompt"
    turn2 = turn1 + [choice["message"], _user("next", second)]
    assert asyncio.run(_chat(core, session_id, turn2)).status_code == 200

    # Turn 1 is rendered with one placeholder and expanded from the start.
    body1, body2 = engine.bodies
    assert body1["mm_token_expansion_start_len"] == 0 and body1["return_prompt_token_ids"] is True
    assert body1["input_ids"].count(pad_id) == 1
    # Turn 2 sends turn 1's expanded prompt and completion, then the new turn with one placeholder.
    start = body2["mm_token_expansion_start_len"]
    assert body2["input_ids"][:start] == engine.prompts[0] + engine.output_ids
    assert body2["input_ids"][start:].count(pad_id) == 1
    # Both images reach the engine as shared paths, under the media dir, named by content.
    paths = _image_urls(body2["messages"])
    assert len(paths) == 2 and all(path.startswith(str(tmp_path.resolve())) for path in paths)
    assert _image_urls(body1["messages"]) == paths[:1]

    samples = _samples(core, session_id, version)
    (sample,) = samples
    assert sample.multimodal_inputs == {"images": paths}
    reply_len = len(engine.output_ids)
    assert sample.loss_mask.count(1) == 2 * reply_len, "only the two sampled replies are trained"
    # The sample is turn 2's expanded prompt plus its reply.
    assert sample.tokens == engine.prompts[1] + engine.output_ids

    # The trainer rebuilds from the references and the expanded tokens what the engine computed.
    (train_inputs,) = compute_multimodal_train_inputs(
        engine_args, [sample.multimodal_inputs], [torch.tensor(sample.tokens)], device=torch.device("cpu")
    )
    assert torch.equal(train_inputs["pixel_values"], engine.pixel_values[-1])
    assert train_inputs["image_grid_thw"].shape[0] == 2


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_a_retry_resends_the_checkpoint_it_continues(version, tokenizer, processor_core, tmp_path):
    """The agent retries turn 2 with another screenshot: v1 rolls back to turn 1, v2 branches under it.
    Both send turn 1's expanded checkpoint again and expand only the new image."""
    engine = TokenSpaceEngine(processor_core, tokenizer)
    core, registry = _make_core(version, tokenizer, engine, tmp_path)
    session_id = registry.create_session()
    first, second, retry = (_data_url(64, 64, c) for c in ((200, 10, 10), (5, 5, 5), (9, 90, 9)))

    turn1 = [_user("look", first)]
    assistant = json.loads(asyncio.run(_chat(core, session_id, turn1)).body)["choices"][0]["message"]
    for url in (second, retry):
        assert asyncio.run(_chat(core, session_id, turn1 + [assistant, _user("next", url)])).status_code == 200

    _, body2, body3 = engine.bodies
    start = body2["mm_token_expansion_start_len"]
    assert body3["mm_token_expansion_start_len"] == start
    assert body3["input_ids"][:start] == body2["input_ids"][:start] == engine.prompts[0] + engine.output_ids
    # Same text, other screenshot: the same placeholder form, but another image path.
    assert body3["input_ids"] == body2["input_ids"]
    assert _image_urls(body3["messages"])[1] != _image_urls(body2["messages"])[1]

    samples = _samples(core, session_id, version)
    retried = [s for s in samples if s.multimodal_inputs["images"][1] == _image_urls(body3["messages"])[1]]
    assert len(retried) == 1 and retried[0].tokens == engine.prompts[2] + engine.output_ids


def test_text_requests_are_unchanged_with_the_flag_on(tokenizer, processor_core, tmp_path):
    engine = TokenSpaceEngine(processor_core, tokenizer)
    core, registry = _make_core("v2", tokenizer, engine, tmp_path)
    session_id = registry.create_session()
    assert asyncio.run(_chat(core, session_id, [_user("hi")])).status_code == 200
    (body,) = engine.bodies
    assert "mm_token_expansion_start_len" not in body and "return_prompt_token_ids" not in body
    (sample,) = _samples(core, session_id, "v2")
    assert sample.multimodal_inputs is None and sample.tokens == body["input_ids"] + engine.output_ids


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_without_the_flag_media_are_refused(version, tokenizer, processor_core, tmp_path):
    engine = TokenSpaceEngine(processor_core, tokenizer)
    core, registry = _make_core(version, tokenizer, engine, tmp_path, token_space=False)
    session_id = registry.create_session()
    with pytest.raises(MessageValidationError, match="--multimodal-token-space-processing"):
        asyncio.run(_chat(core, session_id, [_user("look", _data_url(64, 64, (1, 2, 3)))]))
    assert engine.bodies == [], "nothing reached the engine"


def test_an_engine_without_token_space_processing_is_refused(tokenizer, processor_core, tmp_path):
    engine = TokenSpaceEngine(processor_core, tokenizer, return_prompt=False)
    core, registry = _make_core("v2", tokenizer, engine, tmp_path)
    session_id = registry.create_session()
    with pytest.raises(UpstreamResponseError, match="did not return the expanded prompt"):
        asyncio.run(_chat(core, session_id, [_user("look", _data_url(64, 64, (1, 2, 3)))]))
    assert registry.get_session(session_id).tree.nodes == [], "nothing was committed"


def test_a_client_cannot_set_the_expansion_offset(tokenizer, processor_core, tmp_path):
    engine = TokenSpaceEngine(processor_core, tokenizer)
    core, registry = _make_core("v2", tokenizer, engine, tmp_path)
    session_id = registry.create_session()
    body = json.dumps({"messages": [_user("hi")], "mm_token_expansion_start_len": 3}).encode()
    with pytest.raises(MessageValidationError, match="mm_token_expansion_start_len"):
        asyncio.run(core.chat_completions(session_id, method="POST", query="", headers={}, body=body))


def test_a_client_that_asks_for_the_prompt_gets_the_expanded_one(tokenizer, processor_core, tmp_path):
    engine = TokenSpaceEngine(processor_core, tokenizer)
    core, registry = _make_core("v2", tokenizer, engine, tmp_path)
    session_id = registry.create_session()
    messages = [_user("look", _data_url(64, 64, (1, 2, 3)))]
    body = json.dumps({"messages": messages, "return_prompt_token_ids": True}).encode()
    response = asyncio.run(core.chat_completions(session_id, method="POST", query="", headers={}, body=body))
    assert json.loads(response.body)["choices"][0]["prompt_token_ids"] == engine.prompts[0]


async def test_the_samples_worker_builds_from_its_snapshot(tokenizer, processor_core, tmp_path, monkeypatch):
    """The samples op snapshots the session under its lock and builds in a worker thread. A chat with a
    new image that lands during the build must not reach it: the build returns what a build with no
    chat in between returns. The media are immutable files named by content, so the snapshot's records
    are all the build reads."""
    engine = TokenSpaceEngine(processor_core, tokenizer)
    core, registry = _make_core("v2", tokenizer, engine, tmp_path)
    session_id = registry.create_session()
    turn1 = [_user("look", _data_url(64, 64, (200, 10, 10)))]
    first = await _chat(core, session_id, turn1)
    assert first.status_code == 200, first.body
    expected = await core.collect_samples(session_id, max_seq_len=None)
    assert expected.status_code == 200, expected.body

    started, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    collect = core._collect_samples

    def blocked(sid, snapshot, **kwargs):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=5)
        return collect(sid, snapshot, **kwargs)

    monkeypatch.setattr(core, "_collect_samples", blocked)
    task = asyncio.create_task(core.collect_samples(session_id, max_seq_len=None))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        assistant = json.loads(first.body)["choices"][0]["message"]
        turn2 = turn1 + [assistant, _user("next", _data_url(96, 64, (5, 5, 5)))]
        assert (await _chat(core, session_id, turn2)).status_code == 200
    finally:
        release.set()
        response = await task
    await core.close()

    assert response.status_code == 200, response.body
    assert response.body == expected.body
    (sample,) = decode_samples_and_merge_input_sample(response.body, Sample(), fields=COMPUTED_FIELDS_V2).samples
    assert len(sample.multimodal_inputs["images"]) == 1
