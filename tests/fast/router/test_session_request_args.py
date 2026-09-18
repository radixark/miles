"""HTTP-level tests for how the session server decides outbound chat-request arguments.

Body fields: ``request_args.resolve_request_args_by_config`` (what the server owns, what
it rejects, what it forwards).  Template args (``chat_template_kwargs`` and ``tools``):
``TITOTokenizer.resolve_request_args`` against the ``turn_args`` recorded by the
turn a request continues.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from unittest.mock import patch

import pytest
import requests
from fastapi.responses import JSONResponse
from tests.fast.router.test_sessions import _create_session, _post_chat, _serve_router
from tests.fast.router.test_sessions_v2 import _serve_router as _serve_router_v2

from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer
from miles.utils.lora import LORA_ADAPTER_NAME
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer

USER = {"role": "user", "content": "hi"}
LAUNCH_KWARGS = {"enable_thinking": False}  # both ``_serve_router`` helpers launch with this
THINKING_ON = {"enable_thinking": True}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}}}]
OTHER_TOOLS = [
    {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}}
]


def _serve(version: str, extra_args: dict | None = None):
    serve = _serve_router_v2 if version == "v2" else _serve_router
    return serve(extra_args)


def _records(url: str, session_id: str) -> list[dict]:
    return requests.get(f"{url}/sessions/{session_id}", timeout=5.0).json()["records"]


def _metadata(url: str, session_id: str) -> dict:
    return requests.get(f"{url}/sessions/{session_id}", timeout=5.0).json()["metadata"]


def _assert_exported_turn_args(exported: dict, request_args: dict):
    assert "input_ids" not in exported
    assert "messages" not in exported
    assert {**exported, "input_ids": request_args["input_ids"], "messages": request_args["messages"]} == request_args


class TestForbiddenClientFields:
    @pytest.mark.parametrize(
        ("field", "value"),
        [("input_ids", [1, 2, 3]), ("routed_experts_start_len", 0), ("logprob_start_len", 0), ("lora_path", "x")],
    )
    def test_tito_control_fields_return_400_and_record_nothing(self, field, value):
        with _serve_router() as env:
            session_id = _create_session(env.url)
            resp = _post_chat(env.url, session_id, {"messages": [USER], field: value})
            assert resp.status_code == 400
            assert f"{field}={value!r} is not accepted" in resp.json()["error"]
            assert _records(env.url, session_id) == []

    def test_model_adapter_suffix_rejected_only_when_lora_rollout_is_enabled(self):
        with _serve_router({"lora_rank": 8}) as env:
            session_id = _create_session(env.url)
            resp = _post_chat(env.url, session_id, {"messages": [USER], "model": "base:adapter"})
            assert resp.status_code == 400
            assert "LoRA adapter" in resp.json()["error"]
        with _serve_router() as env:
            session_id = _create_session(env.url)
            assert _post_chat(env.url, session_id, {"messages": [USER], "model": "base:adapter"}).status_code == 200
            assert env.backend.request_log[-1]["model"] == "base:adapter"


class TestServerOwnedFields:
    def test_client_values_are_replaced_and_replay_flags_are_always_present(self):
        with _serve_router() as env:
            session_id = _create_session(env.url)
            resp = _post_chat(
                env.url,
                session_id,
                {
                    "messages": [USER],
                    "temperature": 0.7,
                    "logprobs": False,
                    "return_meta_info": False,
                    "no_stop_trim": True,
                    "return_routed_experts": True,
                    "return_indexer_topk": True,
                },
            )
            assert resp.status_code == 200
            wire = env.backend.request_log[-1]
            assert wire["logprobs"] is True
            assert wire["return_meta_info"] is True
            assert wire["no_stop_trim"] is False
            assert wire["return_routed_experts"] is False
            assert wire["return_indexer_topk"] is False
            assert "lora_path" not in wire
            assert wire["temperature"] == 0.7  # not in the table: the client's

    def test_lora_path_follows_lora_rollout_enabled(self):
        with _serve_router({"lora_rank": 8}) as env:
            session_id = _create_session(env.url)
            assert _post_chat(env.url, session_id, {"messages": [USER]}).status_code == 200
            assert env.backend.request_log[-1]["lora_path"] == LORA_ADAPTER_NAME
        with _serve_router({"lora_rank": 8, "lora_train_only": True}) as env:
            session_id = _create_session(env.url)
            assert _post_chat(env.url, session_id, {"messages": [USER]}).status_code == 200
            assert "lora_path" not in env.backend.request_log[-1]


class TestChatTemplateKwargs:
    def test_request_kwargs_override_the_launch_for_renderer_and_wire(self):
        with _serve_router() as env:
            session_id = _create_session(env.url)
            assert _post_chat(env.url, session_id, {"messages": [USER]}).status_code == 200
            launch_wire = env.backend.request_log[-1]
            assert launch_wire["chat_template_kwargs"] == LAUNCH_KWARGS

            session_id = _create_session(env.url)
            resp = _post_chat(env.url, session_id, {"messages": [USER], "chat_template_kwargs": THINKING_ON})
            assert resp.status_code == 200
            wire = env.backend.request_log[-1]
            assert wire["chat_template_kwargs"] == THINKING_ON
            assert wire["input_ids"] != launch_wire["input_ids"]  # the local render followed the request

    def test_non_object_chat_template_kwargs_is_400(self):
        with _serve_router() as env:
            session_id = _create_session(env.url)
            resp = _post_chat(env.url, session_id, {"messages": [USER], "chat_template_kwargs": "oops"})
            assert resp.status_code == 400
            assert resp.json()["error"] == "chat_template_kwargs must be an object"

    def test_tools_inside_chat_template_kwargs_is_400(self):
        with _serve_router() as env:
            session_id = _create_session(env.url)
            resp = _post_chat(env.url, session_id, {"messages": [USER], "chat_template_kwargs": {"tools": TOOLS}})
            assert resp.status_code == 400
            assert "tools belongs at the top level" in resp.json()["error"]


@pytest.mark.parametrize("version", ["v1", "v2"])
class TestTurnArgs:
    """A committed turn records the full resolved request; continuation rules are per-field."""

    def _turn(self, env, session_id: str, messages: list, **extra) -> requests.Response:
        return _post_chat(env.url, session_id, {"messages": messages, **extra})

    def test_continuing_a_turn_inherits_omitted_kwargs_and_accepts_request_overrides(self, version):
        with _serve(version) as env:
            session_id = _create_session(env.url)
            assert _metadata(env.url, session_id)["turn_args"] == {}

            first = self._turn(env, session_id, [USER], chat_template_kwargs=THINKING_ON)
            assert first.status_code == 200
            _assert_exported_turn_args(_metadata(env.url, session_id)["turn_args"], env.backend.request_log[-1])
            assert _metadata(env.url, session_id)["turn_args"]["chat_template_kwargs"] == THINKING_ON
            assistant = first.json()["choices"][0]["message"]
            history = [USER, assistant, {"role": "user", "content": "more"}]

            second = self._turn(env, session_id, history)
            assert second.status_code == 200
            assert env.backend.request_log[-1]["chat_template_kwargs"] == THINKING_ON
            assert len(_records(env.url, session_id)) == 2

            third = self._turn(env, session_id, history, chat_template_kwargs=LAUNCH_KWARGS)
            assert third.status_code == 200
            assert env.backend.request_log[-1]["chat_template_kwargs"] == LAUNCH_KWARGS
            _assert_exported_turn_args(_metadata(env.url, session_id)["turn_args"], env.backend.request_log[-1])
            if version == "v1":
                assert len(_records(env.url, session_id)) == 2
            else:
                nodes = _metadata(env.url, session_id)["tree"]["nodes"]
                assert len(nodes) == 3
                assert nodes[1]["turn_args"]["chat_template_kwargs"] == THINKING_ON

    def test_continuing_a_turn_inherits_its_tools_and_rejects_a_change(self, version):
        with _serve(version) as env:
            session_id = _create_session(env.url)
            first = self._turn(env, session_id, [USER], tools=TOOLS)
            assert first.status_code == 200
            _assert_exported_turn_args(_metadata(env.url, session_id)["turn_args"], env.backend.request_log[-1])
            assert _metadata(env.url, session_id)["turn_args"]["tools"] == TOOLS
            assistant = first.json()["choices"][0]["message"]
            history = [USER, assistant, {"role": "user", "content": "more"}]

            second = self._turn(env, session_id, history)  # tools omitted: inherited, and back on the wire
            assert second.status_code == 200
            assert env.backend.request_log[-1]["tools"] == TOOLS
            before = requests.get(f"{env.url}/sessions/{session_id}", timeout=5.0).json()
            backend_requests = len(env.backend.request_log)

            third = self._turn(env, session_id, history, tools=OTHER_TOOLS)
            assert third.status_code == 400
            assert "tools changed on a continued turn" in third.json()["error"]
            assert requests.get(f"{env.url}/sessions/{session_id}", timeout=5.0).json() == before
            assert len(env.backend.request_log) == backend_requests

    def test_retry_inherits_target_checkpoint_kwargs_not_latest_turn(self, version):
        with _serve(version) as env:
            session_id = _create_session(env.url)
            first = self._turn(env, session_id, [USER], chat_template_kwargs=THINKING_ON)
            assert first.status_code == 200
            assistant = first.json()["choices"][0]["message"]
            history = [USER, assistant, {"role": "user", "content": "more"}]
            assert self._turn(env, session_id, history, chat_template_kwargs=LAUNCH_KWARGS).status_code == 200
            assert _metadata(env.url, session_id)["turn_args"]["chat_template_kwargs"] == LAUNCH_KWARGS

            assert self._turn(env, session_id, history).status_code == 200

            assert env.backend.request_log[-1]["chat_template_kwargs"] == THINKING_ON
            _assert_exported_turn_args(_metadata(env.url, session_id)["turn_args"], env.backend.request_log[-1])
            assert len(_records(env.url, session_id)) == 2

    def test_a_new_root_may_choose_again(self, version):
        """v1: retrying the first turn rolls back to the empty checkpoint; v2: a second root."""
        with _serve(version) as env:
            session_id = _create_session(env.url)
            assert self._turn(env, session_id, [USER], chat_template_kwargs=THINKING_ON).status_code == 200

            again = self._turn(env, session_id, [USER])
            assert again.status_code == 200
            assert env.backend.request_log[-1]["chat_template_kwargs"] == LAUNCH_KWARGS
            metadata = _metadata(env.url, session_id)
            _assert_exported_turn_args(metadata["turn_args"], env.backend.request_log[-1])
            if version == "v1":
                assert len(_records(env.url, session_id)) == 1
            else:
                nodes = metadata["tree"]["nodes"]
                assert [node["parent"] for node in nodes] == [None, None]
                for node, request_args in zip(nodes, env.backend.request_log, strict=True):
                    _assert_exported_turn_args(node["turn_args"], request_args)

    def test_failed_turn_records_nothing(self, version):
        original = MockSGLangServer._handle_generate_like_request
        calls = {"n": 0}

        async def fail_first(self, request, compute_fn):
            calls["n"] += 1
            if calls["n"] == 1:
                return JSONResponse(status_code=500, content={"error": "backend down"})
            return await original(self, request, compute_fn)

        with _serve(version) as env:
            session_id = _create_session(env.url)
            with patch.object(MockSGLangServer, "_handle_generate_like_request", new=fail_first):
                assert self._turn(env, session_id, [USER], chat_template_kwargs=THINKING_ON).status_code == 500
                assert _metadata(env.url, session_id)["turn_args"] == {}
                assert self._turn(env, session_id, [USER]).status_code == 200
            assert env.backend.request_log[-1]["chat_template_kwargs"] == LAUNCH_KWARGS
            _assert_exported_turn_args(_metadata(env.url, session_id)["turn_args"], env.backend.request_log[-1])

    def test_full_snapshot_records_sampling_without_inheriting_it(self, version):
        with _serve(version) as env:
            session_id = _create_session(env.url)
            first = self._turn(env, session_id, [USER], temperature=0.2, seed=42, tools=TOOLS)
            assert first.status_code == 200
            first_args = _metadata(env.url, session_id)["turn_args"]
            first_request = deepcopy(env.backend.request_log[-1])
            _assert_exported_turn_args(first_args, first_request)
            assert first_args["temperature"] == 0.2 and first_args["seed"] == 42
            assert first_request["input_ids"]
            assert first_request["messages"] == [USER]
            assert _records(env.url, session_id)[0]["request"] == first_request
            assistant = first.json()["choices"][0]["message"]
            history = [USER, assistant, {"role": "user", "content": "more"}]
            original_resolve = TITOTokenizer.resolve_request_args

            def resolve(tokenizer, request_args, *, turn_args):
                assert turn_args == first_request
                return original_resolve(tokenizer, request_args, turn_args=turn_args)

            with patch.object(TITOTokenizer, "resolve_request_args", new=resolve):
                assert self._turn(env, session_id, history, temperature=0.8).status_code == 200
            second_args = _metadata(env.url, session_id)["turn_args"]
            _assert_exported_turn_args(second_args, env.backend.request_log[-1])
            assert second_args["temperature"] == 0.8 and "seed" not in second_args
            assert second_args["tools"] == TOOLS
            assert second_args["chat_template_kwargs"] == LAUNCH_KWARGS
            if version == "v2":
                assert _metadata(env.url, session_id)["tree"]["nodes"][0]["turn_args"] == first_args

    def test_full_model_result_reaches_render_backend_and_snapshot(self, version):
        original = TITOTokenizer.request_arg_rules

        def resolve(request_args, *, request_source, turn_args):
            assert request_source["temperature"] == 0.8
            request_args["temperature"] = 0.25
            request_args["tools"] = deepcopy(TOOLS)
            request_args["chat_template_kwargs"]["enable_thinking"] = True

        def rules(self):
            result = original(self)
            result.append(resolve)
            return result

        with _serve(version) as env:
            session_id = _create_session(env.url)
            assert self._turn(env, session_id, [USER]).status_code == 200
            default_ids = env.backend.request_log[-1]["input_ids"]
            session_id = _create_session(env.url)
            with patch.object(TITOTokenizer, "request_arg_rules", new=rules):
                assert self._turn(env, session_id, [USER], temperature=0.8).status_code == 200
            wire = env.backend.request_log[-1]
            assert wire["temperature"] == 0.25
            assert wire["tools"] == TOOLS
            assert wire["chat_template_kwargs"] == THINKING_ON
            assert wire["input_ids"] != default_ids
            _assert_exported_turn_args(_metadata(env.url, session_id)["turn_args"], wire)


def test_v2_concurrent_first_turns_with_different_kwargs_both_commit_as_roots():
    """Neither first turn continues a recorded node, so both commit, each with its own kwargs."""
    with _serve("v2") as env:
        session_id = _create_session(env.url)
        arrivals = 0
        release = None

        async def wait_for_pair(self, request, compute_fn):
            nonlocal arrivals, release
            payload = await request.json()
            self.request_log.append(payload)
            if release is None:
                release = asyncio.Event()
            arrivals += 1
            if arrivals == 2:
                release.set()
            await asyncio.wait_for(release.wait(), timeout=5.0)
            return JSONResponse(content=compute_fn(payload))

        with patch.object(MockSGLangServer, "_handle_generate_like_request", new=wait_for_pair):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(_post_chat, env.url, session_id, {"messages": [USER], "chat_template_kwargs": kwargs})
                    for kwargs in (THINKING_ON, LAUNCH_KWARGS)
                ]
                responses = [future.result(timeout=10.0) for future in futures]

        assert all(response.status_code == 200 for response in responses)
        nodes = _metadata(env.url, session_id)["tree"]["nodes"]
        assert [node["parent"] for node in nodes] == [None, None]
        assert sorted(node["turn_args"]["chat_template_kwargs"]["enable_thinking"] for node in nodes) == [False, True]

        # Omitted kwargs inherit the selected root; explicit ordinary fields can override it.
        [record] = _records(env.url, session_id)  # the served chain: the root committed last
        recorded = record["request"]["chat_template_kwargs"]
        other = LAUNCH_KWARGS if recorded == THINKING_ON else THINKING_ON
        assistant = record["response"]["choices"][0]["message"]
        history = [USER, assistant, {"role": "user", "content": "more"}]
        assert _post_chat(env.url, session_id, {"messages": history, "chat_template_kwargs": other}).status_code == 200
        assert _metadata(env.url, session_id)["turn_args"]["chat_template_kwargs"] == other
        assert _post_chat(env.url, session_id, {"messages": history}).status_code == 200
        assert _metadata(env.url, session_id)["turn_args"]["chat_template_kwargs"] == recorded
