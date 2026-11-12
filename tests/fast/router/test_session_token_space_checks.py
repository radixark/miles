from __future__ import annotations

import base64
import io
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image
from tests.ci.ci_register import register_cpu_ci
from tests.fast.fixtures.args_fixtures import sglang_router_args
from tests.fast.fixtures.sglang_config_fixtures import make_sglang_config
from tests.fast.router.test_sessions import _create_session, _post_chat

from miles.backends.megatron_utils.megatron_config import MegatronConfig
from miles.rollout.session import core as session_core
from miles.rollout.session import image_expansion
from miles.rollout.session.config import compute_session_server_config
from miles.rollout.session.samples.codec import COMPUTED_FIELDS_V2, decode_samples_and_merge_input_sample
from miles.rollout.session.server import SessionServer
from miles.utils.chat_template_utils import resolve_fixed_chat_template
from miles.utils.http_utils import find_available_port
from miles.utils.media_refs import collect_media_refs, release_run_media
from miles.utils.processing_utils import load_processor, load_tokenizer
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer
from miles.utils.types import Sample

register_cpu_ci(est_time=50, suite="stage-b-cpu", labels=["rollout"])

CHECKPOINT = "Qwen/Qwen3-VL-2B-Instruct"
QWEN38 = {
    "hf_checkpoint": "Qwen/Qwen3.8-27B",
    "tito_model": "qwen38small",
    "chat_template_path": resolve_fixed_chat_template("qwen38small")[0],
}


def load_vlm(checkpoint: str, chat_template_path: str | None = None) -> SimpleNamespace:
    processor = load_processor(checkpoint, trust_remote_code=True)
    if processor is None:
        pytest.skip(f"{checkpoint} processor unavailable")
    tokenizer = load_tokenizer(checkpoint, chat_template_path=chat_template_path, trust_remote_code=True)
    return SimpleNamespace(
        processor=processor,
        tokenizer=tokenizer,
        pad_id=tokenizer.convert_tokens_to_ids(processor.image_token),
        video_pad_id=tokenizer.convert_tokens_to_ids(processor.video_token),
    )


@pytest.fixture(scope="module")
def vlm():
    return load_vlm(CHECKPOINT)


@pytest.fixture(scope="module")
def qwen38():
    return load_vlm(QWEN38["hf_checkpoint"], QWEN38["chat_template_path"])


def png(width: int, height: int, color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def user_turn(payload: bytes | None, text: str) -> dict:
    content: list[dict] = [{"type": "text", "text": text}]
    if payload is not None:
        url = f"data:image/png;base64,{base64.b64encode(payload).decode('ascii')}"
        content.insert(0, {"type": "image_url", "image_url": {"url": url}})
    return {"role": "user", "content": content}


class TokenSpaceChatBackend:
    def __init__(
        self,
        vlm,
        *,
        count_bias_on_call: dict[int, int] | None = None,
        return_prompt: bool = True,
        image_kwargs: dict | None = None,
    ):
        self.vlm = vlm
        self.image_kwargs = image_kwargs or {}
        self.count_bias_on_call = count_bias_on_call or {}
        self.return_prompt = return_prompt
        self.calls: list[dict[str, Any]] = []
        self.host = "127.0.0.1"
        self.port = find_available_port(32000)
        self.app = FastAPI()
        self._server = UvicornThreadServer(self.app, host=self.host, port=self.port)

        @self.app.get("/health")
        async def health():
            return JSONResponse(content={"status": "ok"})

        @self.app.post("/v1/chat/completions")
        async def chat(request: Request):
            payload = await request.json()
            self.calls.append(payload)
            return JSONResponse(content=self._reply(payload))

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self):
        self._server.start()

    def stop(self):
        self._server.stop()

    def _count(self, path: str, bias: int) -> int:
        image_processor = self.vlm.processor.image_processor
        with Image.open(path) as handle:
            grid = image_processor(images=[handle.convert("RGB")], return_tensors="pt", **self.image_kwargs)
        return int(grid["image_grid_thw"][0].prod()) // image_processor.merge_size**2 + bias

    def _reply(self, payload: dict) -> dict:
        index = len(self.calls) - 1
        prompt = list(payload["input_ids"])
        if "mm_token_expansion_start_len" in payload:
            start = payload["mm_token_expansion_start_len"]
            images = (collect_media_refs(payload["messages"], None) or {}).get("images", [])
            tail = prompt[start:]
            new_images = images[len(images) - tail.count(self.vlm.pad_id) :]
            bias = self.count_bias_on_call.get(index, 0)
            counts = iter([self._count(path, bias) for path in new_images])
            expanded = []
            for token_id in tail:
                if token_id == self.vlm.pad_id:
                    expanded.extend([token_id] * next(counts))
                elif token_id == self.vlm.video_pad_id:
                    expanded.extend([token_id] * VIDEO_TOKENS)
                else:
                    expanded.append(token_id)
            prompt = prompt[:start] + expanded
        output_ids = self.vlm.tokenizer.encode(f"reply {index}", add_special_tokens=False)
        choice = {
            "index": 0,
            "message": {"role": "assistant", "content": f"reply {index}"},
            "finish_reason": "stop",
            "meta_info": {
                "id": f"chat-{index}",
                "completion_tokens": len(output_ids),
                "weight_version": "w0",
                "output_token_logprobs": [[-0.1, token_id, None] for token_id in output_ids],
            },
        }
        if self.return_prompt and payload.get("return_prompt_token_ids"):
            choice["prompt_token_ids"] = prompt
        return {"id": f"chat-{index}", "object": "chat.completion", "created": 0, "model": "m", "choices": [choice]}


def server_args(shared_media_dir: Path, **extra) -> dict[str, Any]:
    return {
        "miles_router_timeout": 30,
        "hf_checkpoint": CHECKPOINT,
        "allow_stop_token_replace": False,
        "chat_template_path": None,
        "apply_chat_template_kwargs": None,
        "tito_model": "qwen3_vl",
        "use_rollout_routing_replay": False,
        "use_rollout_indexer_replay": False,
        "rollout_indexer_topk_num_streams": None,
        "use_sampling_support_replay": False,
        "use_output_store": False,
        "rollout_top_logprobs_num": 0,
        "rollout_sampling_logprobs_mode": "selected",
        "sglang": make_sglang_config(speculative_algorithm=None, mm_process_config=None),
        "num_layers": None,
        "raw_megatron": MegatronConfig(trainers=[], base_args={}),
        "save_debug_trajectory_data": None,
        "lora_rank": 0,
        "lora_adapter_path": None,
        "lora_train_only": False,
        "use_session_server": "v2",
        "multimodal_token_space_processing": True,
        "shared_media_dir": str(shared_media_dir),
        "run_uuid": "run0",
        "session_message_matcher": "strict",
        "session_server_instance_id": uuid.uuid4().hex,
        "pause_generation_mode": "retract",
        "session_sample_picker_path": "miles.rollout.session.v2.picker_hub.drop_same_prompt_retries",
        "session_sample_postprocessor_path": "miles.rollout.session.v2.postprocessor_hub.default_postprocess",
        **sglang_router_args("manual"),
        **extra,
    }


@contextmanager
def session_server(
    vlm,
    shared_media_dir: Path,
    *,
    count_bias_on_call: dict[int, int] | None = None,
    return_prompt: bool = True,
    engine_image_kwargs: dict | None = None,
    **extra,
):
    backend = TokenSpaceChatBackend(
        vlm, count_bias_on_call=count_bias_on_call, return_prompt=return_prompt, image_kwargs=engine_image_kwargs
    )
    backend.start()
    args = SimpleNamespace(**server_args(shared_media_dir, **extra))
    try:
        port = find_available_port(31000)
        config = compute_session_server_config(
            args, host="127.0.0.1", port=port, instance_id=args.session_server_instance_id, backend_url=backend.url
        )
        server = UvicornThreadServer(SessionServer(config).app, host="127.0.0.1", port=port)
        server.start()
        try:
            yield SimpleNamespace(url=f"http://127.0.0.1:{port}", backend=backend, args=args)
        finally:
            server.stop()
    finally:
        backend.stop()


def chat(env, session_id: str, messages: list[dict]) -> requests.Response:
    return _post_chat(env.url, session_id, {"messages": messages, "max_tokens": 16, "temperature": 1.0})


def collect(env, session_id: str) -> list[Sample]:
    response = requests.post(f"{env.url}/sessions/{session_id}/samples", json={}, timeout=20.0)
    assert response.status_code == 200, response.content
    return decode_samples_and_merge_input_sample(response.content, Sample(), fields=COMPUTED_FIELDS_V2).samples


VIDEO_TOKENS = 6
CLIP = b"\x00\x00\x00\x18ftypmp42clip"


def video_turn(text: str) -> dict:
    url = f"data:video/mp4;base64,{base64.b64encode(CLIP).decode('ascii')}"
    return {"role": "user", "content": [{"type": "video", "video": url}, {"type": "text", "text": text}]}


SHOT1 = png(120, 80, (10, 20, 30))
SHOT2 = png(160, 120, (40, 80, 120))
SHOT1_TWIN = png(120, 80, (200, 190, 180))


class TestEngineExpansionCheck:
    def test_matching_expansions_pass_across_turns(self, vlm, tmp_path):
        with session_server(vlm, tmp_path) as env:
            session_id = _create_session(env.url)
            first = [user_turn(SHOT1, "one")]
            response = chat(env, session_id, first)
            assert response.status_code == 200, response.content
            second = first + [response.json()["choices"][0]["message"], user_turn(SHOT2, "two")]
            assert chat(env, session_id, second).status_code == 200
            (sample,) = collect(env, session_id)
            assert len(sample.multimodal_inputs["images"]) == 2

    def test_a_first_turn_expanded_differently_fails(self, vlm, tmp_path):
        with session_server(vlm, tmp_path, count_bias_on_call={0: 1}) as env:
            session_id = _create_session(env.url)
            response = chat(env, session_id, [user_turn(SHOT1, "one")])
            assert response.status_code == 502
            assert b"expanded prompt differs" in response.content

    def test_a_new_image_expanded_differently_after_the_prefix_fails(self, vlm, tmp_path):
        with session_server(vlm, tmp_path, count_bias_on_call={1: -1}) as env:
            session_id = _create_session(env.url)
            first = [user_turn(SHOT1, "one")]
            response = chat(env, session_id, first)
            assert response.status_code == 200, response.content
            second = first + [response.json()["choices"][0]["message"], user_turn(SHOT2, "two")]
            response = chat(env, session_id, second)
            assert response.status_code == 502
            assert b"expanded prompt differs" in response.content

    def test_families_it_cannot_count_keep_the_prefix_check(self, vlm, tmp_path):
        with session_server(vlm, tmp_path, count_bias_on_call={0: 1}, tito_model="default") as env:
            session_id = _create_session(env.url)
            assert chat(env, session_id, [user_turn(SHOT1, "one")]).status_code == 200

    @pytest.mark.parametrize("use_session_server", ["v1", "v2"])
    @pytest.mark.parametrize(
        ("tito_model", "expected"),
        [("qwen3_vl", ["elementwise"]), ("default", ["prefix"])],
        ids=["counted", "uncounted"],
    )
    def test_a_turn_runs_one_engine_prompt_check(
        self, vlm, tmp_path, monkeypatch, use_session_server, tito_model, expected
    ):
        ran = []

        def spy(name, check):
            def wrapped(*args):
                ran.append(name)
                return check(*args)

            return wrapped

        monkeypatch.setattr(session_core, "check_engine_prompt", spy("prefix", session_core.check_engine_prompt))
        monkeypatch.setattr(
            session_core,
            "assert_engine_expansion_matches",
            spy("elementwise", image_expansion.assert_engine_expansion_matches),
        )
        with session_server(vlm, tmp_path, tito_model=tito_model, use_session_server=use_session_server) as env:
            session_id = _create_session(env.url)
            assert chat(env, session_id, [user_turn(SHOT1, "one")]).status_code == 200
        assert ran == expected

    @pytest.mark.parametrize("use_session_server", ["v1", "v2"])
    def test_a_video_turn_the_engine_expanded_keeps_the_prefix_check(self, vlm, tmp_path, use_session_server):
        with session_server(vlm, tmp_path, use_session_server=use_session_server) as env:
            session_id = _create_session(env.url)
            first = [user_turn(SHOT1, "one")]
            response = chat(env, session_id, first)
            assert response.status_code == 200, response.content
            second = first + [response.json()["choices"][0]["message"], video_turn("two")]
            response = chat(env, session_id, second)
            assert response.status_code == 200, response.content
            assert env.backend.calls[-1]["input_ids"].count(vlm.video_pad_id) == 1

    @pytest.mark.parametrize(
        "image",
        [{"min_pixels": 4096, "max_pixels": 16384}, {"size": {"shortest_edge": 4096, "longest_edge": 16384}}],
        ids=["pixels", "size"],
    )
    def test_images_are_counted_with_the_engines_image_settings(self, vlm, tmp_path, image):
        shot = tmp_path / "shot2.png"
        shot.write_bytes(SHOT2)
        engine = TokenSpaceChatBackend(vlm, image_kwargs=image)
        assert engine._count(str(shot), 0) != TokenSpaceChatBackend(vlm)._count(str(shot), 0)
        sglang = make_sglang_config(speculative_algorithm=None, mm_process_config={"image": image})
        with session_server(vlm, tmp_path, engine_image_kwargs=image, sglang=sglang) as env:
            session_id = _create_session(env.url)
            first = [user_turn(SHOT1, "one")]
            response = chat(env, session_id, first)
            assert response.status_code == 200, response.content
            second = first + [response.json()["choices"][0]["message"], user_turn(SHOT2, "two")]
            response = chat(env, session_id, second)
            assert response.status_code == 200, response.content

    def test_an_engine_that_returns_no_prompt_fails_the_counted_check(self, vlm, tmp_path):
        with session_server(vlm, tmp_path, return_prompt=False) as env:
            session_id = _create_session(env.url)
            response = chat(env, session_id, [user_turn(SHOT1, "one")])
            assert response.status_code == 502
            assert b"--enable-token-space-processor" in response.content


class TestQwen38ExpansionCheck:
    def test_matching_expansions_pass_across_turns(self, qwen38, tmp_path):
        with session_server(qwen38, tmp_path, **QWEN38) as env:
            session_id = _create_session(env.url)
            first = [user_turn(SHOT1, "one")]
            response = chat(env, session_id, first)
            assert response.status_code == 200, response.content
            second = first + [response.json()["choices"][0]["message"], user_turn(SHOT2, "two")]
            assert chat(env, session_id, second).status_code == 200

    def test_an_image_expanded_differently_fails(self, qwen38, tmp_path):
        with session_server(qwen38, tmp_path, count_bias_on_call={0: 1}, **QWEN38) as env:
            session_id = _create_session(env.url)
            response = chat(env, session_id, [user_turn(SHOT1, "one")])
            assert response.status_code == 502
            assert b"expanded prompt differs" in response.content


class TestScreenshotFolding:
    def test_a_history_with_an_old_screenshot_dropped_branches_at_a_new_root(self, vlm, tmp_path):
        with session_server(vlm, tmp_path) as env:
            session_id = _create_session(env.url)
            first = chat(env, session_id, [user_turn(SHOT1, "one")]).json()["choices"][0]["message"]
            history = [user_turn(SHOT1, "one"), first, user_turn(SHOT2, "two")]
            second = chat(env, session_id, history).json()["choices"][0]["message"]
            folded = [user_turn(None, "one"), first, user_turn(None, "two"), second, user_turn(SHOT1_TWIN, "three")]
            response = chat(env, session_id, folded)
            assert response.status_code == 200, response.content
            assert env.backend.calls[-1]["mm_token_expansion_start_len"] == 0
            assert len(collect(env, session_id)) == 2


class TestRetryPicker:
    def test_a_same_size_sibling_with_another_screenshot_is_not_a_retry(self, vlm, tmp_path):
        with session_server(vlm, tmp_path) as env:
            session_id = _create_session(env.url)
            assert chat(env, session_id, [user_turn(SHOT1, "look")]).status_code == 200
            assert chat(env, session_id, [user_turn(SHOT1_TWIN, "look")]).status_code == 200
            first, second = env.backend.calls
            assert first["input_ids"] == second["input_ids"]
            assert len(collect(env, session_id)) == 2

    def test_an_identical_resend_is_still_a_retry(self, vlm, tmp_path):
        with session_server(vlm, tmp_path) as env:
            session_id = _create_session(env.url)
            assert chat(env, session_id, [user_turn(SHOT1, "look")]).status_code == 200
            assert chat(env, session_id, [user_turn(SHOT1, "look")]).status_code == 200
            assert len(collect(env, session_id)) == 1


class TestRunMediaDir:
    def test_session_media_are_written_under_the_run_and_released_with_it(self, vlm, tmp_path):
        with session_server(vlm, tmp_path) as env:
            session_id = _create_session(env.url)
            assert chat(env, session_id, [user_turn(SHOT1, "look")]).status_code == 200
            (path,) = collect_media_refs(env.backend.calls[0]["messages"], None)["images"]
            assert Path(path).is_relative_to((tmp_path / "run0").resolve())
            release_run_media(env.args)
            assert not Path(path).exists()
