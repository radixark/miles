import contextlib
import uuid
from collections.abc import Callable
from copy import deepcopy
from unittest.mock import patch

import pytest
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from tests.session_parity_utils import (
    SESSION_PARITY_SEED,
    V1,
    V2,
    _serve_session,
    _training_metadata_projection,
    assert_agentic_retry_trajectory_parity,
    assert_sample_bitwise_equal,
    run_agentic_retry_trajectories,
)

from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.mock_sglang_server import (
    MockSGLangServer,
    ProcessResult,
    ProcessResultMetaInfo,
    with_mock_server,
)
from miles.utils.test_utils.session_verify_agent import (
    ASSISTANT_INPUT_FOLLOWUP_TEXT,
    FORCE_FINAL_TEXT,
    MOCK_TOOL_RESULTS,
    SYSTEM_REMINDER_TEXT,
    USER_FOLLOWUP_TEXT,
    build_initial_messages,
)
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer
from miles.utils.types import Sample

_MODEL = "Qwen/Qwen3-0.6B"
_BATCH_SIZE = 16
_REQUESTS_PER_TRAJECTORY = 9
_AGENT_RESPONSES = {
    "initial": ProcessResult(
        'I will check Beijing.\n<tool_call>\n{"name": "get_weather", '
        '"arguments": {"location": "Beijing"}}\n</tool_call>',
        meta_info=ProcessResultMetaInfo(weight_version="w0"),
    ),
    "beijing_result": ProcessResult(
        "Beijing is 22 degrees Celsius and sunny.",
        meta_info=ProcessResultMetaInfo(weight_version="w1"),
    ),
    "shanghai_call": ProcessResult(
        'I will check Shanghai.\n<tool_call>\n{"name": "get_weather", '
        '"arguments": {"location": "Shanghai"}}\n</tool_call>',
        meta_info=ProcessResultMetaInfo(weight_version="w2"),
    ),
    "shanghai_result": ProcessResult(
        "Shanghai is 15 degrees Celsius and cloudy.",
        meta_info=ProcessResultMetaInfo(weight_version="w3"),
    ),
    "rollback_retry": ProcessResult(
        "I will answer in one sentence.",
        meta_info=ProcessResultMetaInfo(weight_version="w4"),
    ),
    "force_final": ProcessResult(
        "<final_answer>Beijing is sunny and Shanghai is cloudy.</final_answer>",
        meta_info=ProcessResultMetaInfo(weight_version="w7"),
    ),
    "assistant_input": ProcessResult(
        "<final_answer>Beijing was sunny and Shanghai was rainy.</final_answer>",
        meta_info=ProcessResultMetaInfo(weight_version="w8"),
    ),
}
_SELECTED_WEIGHT_VERSIONS = ["w0", "w1", "w2", "w3", "w4", "w7", "w8"]
_SHARED_PREFIX_VERSION = "p0"


@pytest.mark.parametrize("version", [V1, V2])
def test_aborted_generation_is_not_committed(version):
    def abort_generation(_prompt: str) -> ProcessResult:
        return ProcessResult(text="partial response", finish_reason="abort")

    with with_mock_server(model_name=_MODEL, process_fn=abort_generation) as backend:
        with _serve_session(backend_url=backend.url, hf_checkpoint=_MODEL, version=version) as args:
            session_url = args.session_server_instances[0].url
            session_id = requests.post(f"{session_url}/sessions", timeout=5.0).json()["session_id"]
            response = requests.post(
                f"{session_url}/sessions/{session_id}/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}]},
                timeout=10.0,
            )

            assert response.status_code == 503
            assert response.json() == {"error": "upstream generation aborted before completion"}
            session = requests.get(f"{session_url}/sessions/{session_id}", timeout=5.0).json()
            assert session["records"] == []
            assert session["metadata"]["last_model_call_failed"] is True


@pytest.mark.parametrize("version", [V1, V2])
def test_session_flags_whether_the_last_model_call_failed(version):
    outcomes = iter(["ok", "fail", "reject", "fail", "ok"])

    def engine(_prompt: str) -> ProcessResult:
        outcome = next(outcomes)
        if outcome == "fail":
            raise HTTPException(status_code=500, detail="engine failed")
        if outcome == "reject":
            raise HTTPException(status_code=400, detail="context too long")
        return ProcessResult(text="hi")

    with with_mock_server(model_name=_MODEL, process_fn=engine) as backend:
        with _serve_session(backend_url=backend.url, hf_checkpoint=_MODEL, version=version) as args:
            session_url, session_id = _new_session(args)
            seen = [_chat_and_read_failures(session_url, session_id) for _ in range(5)]

    assert seen == [(200, False), (500, True), (400, False), (500, True), (200, False)]


@pytest.mark.parametrize("version", [V1, V2])
def test_unreachable_backend_counts_as_a_backend_failure(version):
    backend_url = f"http://127.0.0.1:{find_available_port(32000)}"
    with _serve_session(backend_url=backend_url, hf_checkpoint=_MODEL, version=version) as args:
        assert _chat_and_read_failures(*_new_session(args)) == (502, True)


@pytest.mark.parametrize("version", [V1, V2])
@pytest.mark.parametrize("route", ["v1/chat/completions", "v1/messages"])
def test_a_rejected_request_after_an_engine_failure_is_the_last_call(version, route):
    def engine(_prompt: str) -> ProcessResult:
        raise HTTPException(status_code=500, detail="engine failed")

    with with_mock_server(model_name=_MODEL, process_fn=engine) as backend:
        with _serve_session(backend_url=backend.url, hf_checkpoint=_MODEL, version=version) as args:
            session_url, session_id = _new_session(args)
            assert _chat_and_read_failures(session_url, session_id) == (500, True)
            rejected = requests.post(f"{session_url}/sessions/{session_id}/{route}", data=b"{", timeout=10.0)
            metadata = requests.get(f"{session_url}/sessions/{session_id}", timeout=5.0).json()["metadata"]

    assert (rejected.status_code, metadata["last_model_call_failed"]) == (400, False)


@pytest.mark.parametrize("version", [V1, V2])
@pytest.mark.parametrize(
    "reply",
    [
        b"not json",
        b'{"choices": []}',
        b'{"choices": [{"message": {"content": ""}, "meta_info": {"output_token_logprobs": []}}]}',
    ],
)
def test_a_malformed_engine_reply_counts_as_a_backend_failure(version, reply):
    app = FastAPI()
    app.post("/v1/chat/completions")(lambda: Response(content=reply, media_type="application/json"))
    port = find_available_port(33000)
    backend = UvicornThreadServer(app, host="127.0.0.1", port=port)
    backend.start()
    try:
        with _serve_session(backend_url=f"http://127.0.0.1:{port}", hf_checkpoint=_MODEL, version=version) as args:
            session_url, session_id = _new_session(args)
            with contextlib.suppress(requests.ConnectionError):
                requests.post(
                    f"{session_url}/sessions/{session_id}/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hello"}]},
                    timeout=10.0,
                )
            metadata = requests.get(f"{session_url}/sessions/{session_id}", timeout=5.0).json()["metadata"]
    finally:
        backend.stop()

    assert metadata["last_model_call_failed"] is True


def _new_session(args) -> tuple[str, str]:
    session_url = args.session_server_instances[0].url
    return session_url, requests.post(f"{session_url}/sessions", timeout=5.0).json()["session_id"]


def _chat_and_read_failures(session_url: str, session_id: str) -> tuple[int, bool]:
    response = requests.post(
        f"{session_url}/sessions/{session_id}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
        timeout=10.0,
    )
    metadata = requests.get(f"{session_url}/sessions/{session_id}", timeout=5.0).json()["metadata"]
    return response.status_code, metadata["last_model_call_failed"]


@pytest.mark.parametrize("random_tool_ids", [False, True])
def test_agentic_v2_same_prompt_retries_match_v1_training_payload_bitwise(
    random_tool_ids: bool, monkeypatch: pytest.MonkeyPatch, synchronize_mock_requests: Callable[[int], None]
) -> None:
    """Concurrent v1 and v2 trajectories preserve identical training payloads and retry history."""
    synchronize_mock_requests(_BATCH_SIZE)
    if random_tool_ids:
        original = MockSGLangServer._compute_chat_completions_response

        def response_with_random_tool_ids(self, payload):
            response = original(self, payload)
            for choice in response["choices"]:
                for call in choice["message"].get("tool_calls") or []:
                    call["id"] = f"call_{uuid.uuid4().hex}"
            return response

        monkeypatch.setattr(f"{__name__}._original_chat_completions_response", response_with_random_tool_ids)

    v1_runs = _run_scripted_agents(V1)
    v2_runs = _run_scripted_agents(V2)

    assert len(v1_runs) == len(v2_runs) == _BATCH_SIZE
    for index, (v1, v2) in enumerate(zip(v1_runs, v2_runs, strict=True)):
        assert v1.samples[0].index == v2.samples[0].index == index
        assert v1.samples[0].rollout_id == v2.samples[0].rollout_id == index
        expected_weight_versions = [[version] for version in _SELECTED_WEIGHT_VERSIONS]
        assert [[span.version for span in call.spans] for call in v1.samples[0].weight_versions] == (
            expected_weight_versions
        )
        assert [[span.version for span in call.spans] for call in v2.samples[0].weight_versions] == (
            expected_weight_versions
        )
        for run in (v1, v2):
            _assert_prefill_spans_follow_the_kept_turns(run.samples[0].weight_versions)
        assert_agentic_retry_trajectory_parity(v1, v2)


def _assert_prefill_spans_follow_the_kept_turns(calls):
    """Every kept turn carries its own prompt versions in absolute coordinates, tiled up to its output."""
    assert [call.output_start for call in calls] == sorted({call.output_start for call in calls})
    assert [call.spans[0].abs_start for call in calls] == [call.output_start for call in calls]
    assert [[(span.version, span.abs_start, span.abs_end) for span in call.prefill_spans] for call in calls] == [
        [(_SHARED_PREFIX_VERSION, 0, 1), (version, 1, call.output_start)]
        for call, version in zip(calls, _SELECTED_WEIGHT_VERSIONS, strict=True)
    ]


def test_sample_bitwise_comparator_distinguishes_signed_zero():
    with pytest.raises(AssertionError, match="sample.reward is not bitwise equal"):
        assert_sample_bitwise_equal(Sample(reward=0.0), Sample(reward=-0.0))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("tools", 0, "function", "name"), "get_time"),
        (("chat_template_kwargs", "enable_thinking"), True),
        (("temperature",), 1),
    ],
)
def test_turn_args_parity_checks_exported_request_values(path, value):
    metadata = {
        "turn_args": {
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0,
        }
    }
    changed = deepcopy(metadata)
    target = changed["turn_args"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(AssertionError):
        assert_sample_bitwise_equal(
            Sample(metadata=metadata), Sample(metadata=changed), metadata_projection=_training_metadata_projection
        )


def _run_scripted_agents(version: str):
    input_samples = [
        Sample(
            index=index,
            rollout_id=index,
            prompt=build_initial_messages(),
            reward=0.25,
            metadata={"source": "parity"},
        )
        for index in range(_BATCH_SIZE)
    ]
    with (
        patch.object(MockSGLangServer, "_compute_chat_completions_response", new=_stamp_prompt_versions),
        with_mock_server(model_name=_MODEL, process_fn=_process_agent_prompt) as backend,
    ):
        results = run_agentic_retry_trajectories(
            backend_url=backend.url,
            hf_checkpoint=_MODEL,
            version=version,
            input_samples=input_samples,
        )
        assert len(backend.request_log) == _BATCH_SIZE * _REQUESTS_PER_TRAJECTORY
        assert {request["seed"] for request in backend.request_log} == {SESSION_PARITY_SEED}
        assert backend.max_concurrent == _BATCH_SIZE
    return results


_original_chat_completions_response = MockSGLangServer._compute_chat_completions_response


def _stamp_prompt_versions(mock_self: MockSGLangServer, payload: dict) -> dict:
    response = _original_chat_completions_response(mock_self, payload)
    meta_info = response["choices"][0]["meta_info"]
    num_prompt_tokens = len(payload["input_ids"])
    meta_info["prefill_weight_versions"] = [
        {"version": _SHARED_PREFIX_VERSION, "start": 0, "end": 1},
        {"version": meta_info["weight_version"], "start": 1, "end": num_prompt_tokens},
    ]
    return response


def _process_agent_prompt(prompt: str) -> ProcessResult:
    if ASSISTANT_INPUT_FOLLOWUP_TEXT in prompt:
        return _AGENT_RESPONSES["assistant_input"]
    if FORCE_FINAL_TEXT in prompt:
        return _AGENT_RESPONSES["force_final"]
    if SYSTEM_REMINDER_TEXT in prompt:
        return _AGENT_RESPONSES["rollback_retry"]
    if MOCK_TOOL_RESULTS[1] in prompt:
        return _AGENT_RESPONSES["shanghai_result"]
    if USER_FOLLOWUP_TEXT in prompt:
        return _AGENT_RESPONSES["shanghai_call"]
    if MOCK_TOOL_RESULTS[0] in prompt:
        return _AGENT_RESPONSES["beijing_result"]
    return _AGENT_RESPONSES["initial"]
