import uuid
from copy import deepcopy

import pytest
from tests.session_parity_utils import (
    SESSION_PARITY_SEED,
    V1,
    V2,
    _training_metadata_projection,
    assert_agentic_retry_trajectory_parity,
    assert_sample_bitwise_equal,
    run_agentic_retry_trajectories,
)

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


@pytest.mark.parametrize("random_tool_ids", [False, True])
def test_agentic_v2_drop_retries_matches_v1_training_payload_bitwise(random_tool_ids, monkeypatch):
    if random_tool_ids:
        original = MockSGLangServer._compute_chat_completions_response

        def response_with_random_tool_ids(self, payload):
            response = original(self, payload)
            for choice in response["choices"]:
                for call in choice["message"].get("tool_calls") or []:
                    call["id"] = f"call_{uuid.uuid4().hex}"
            return response

        monkeypatch.setattr(MockSGLangServer, "_compute_chat_completions_response", response_with_random_tool_ids)

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
        assert_agentic_retry_trajectory_parity(v1, v2)


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
    with with_mock_server(model_name=_MODEL, process_fn=_process_agent_prompt, latency=0.05) as backend:
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
