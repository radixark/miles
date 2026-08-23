"""Six-turn Anthropic Messages agent for per-model session verification."""

from __future__ import annotations

import json
import logging

import httpx

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.agentic_tool_call import generate as _base_generate
from miles.utils.chat_template_utils.message_matcher_hub import loose_tool_call_message_matches
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizerType
from miles.utils.test_utils.session_verify_agent import (
    INITIAL_SYSTEM_PROMPT,
    INITIAL_USER_PROMPT,
    MOCK_TOOL_RESULTS,
    SYSTEM_REMINDER_TEXT,
    TOOLS,
    USER_FOLLOWUP_TEXT,
    _MAX_INCOMPLETE_TURN_RETRIES,
    _journal_verifier_assertions,
    _verify_tito_samples,
    fixed_template_append_roles,
)
from miles.utils.test_utils.session_verify_agent import generate as _session_verify_generate

logger = logging.getLogger(__name__)

_ANTHROPIC_TOOLS = [
    {
        "name": tool["function"]["name"],
        "description": tool["function"]["description"],
        "input_schema": tool["function"]["parameters"],
    }
    for tool in TOOLS
]
_TOOL_PROMPTS = (
    INITIAL_USER_PROMPT,
    USER_FOLLOWUP_TEXT,
    "Finally, check the weather in London.",
)
_REQUEST_COUNT = len(_TOOL_PROMPTS) * 2
_MAX_TOKENS_PER_TURN = 1024
_MINIMAX_TITO_MODELS = frozenset(
    {
        TITOTokenizerType.MINIMAX_M25.value,
        TITOTokenizerType.MINIMAX_M27.value,
    }
)
_TOOL_RECOVERY_TEXT = "The weather service is temporarily unavailable."
_TOOL_RESULT_EVENTS = (
    "anthropic_tool_result_string",
    "anthropic_tool_result_list",
    "anthropic_tool_result_string",
)
_INTERMEDIATE_SYSTEM_EXPECTATIONS = frozenset({"required", "forbidden"})


def _assert_supported_tito_model(tito_model: str) -> None:
    assert (
        tito_model not in _MINIMAX_TITO_MODELS
    ), f"Anthropic session verification does not support tito_model={tito_model!r}"


def _verify_intermediate_system(tito_model: str, *, route_supports: bool) -> bool:
    """Require both the live route decision and the TITO append capability."""
    return route_supports and "system" in fixed_template_append_roles(tito_model)


def _assert_intermediate_system_expectation(expectation: str, *, actual: bool) -> None:
    assert (
        expectation in _INTERMEDIATE_SYSTEM_EXPECTATIONS
    ), f"invalid Anthropic intermediate-system expectation: {expectation!r}"
    expected = expectation == "required"
    assert actual is expected, (
        "Anthropic intermediate-system capability did not match the per-model E2E contract: "
        f"expected={expectation}, actual={actual}"
    )


def _build_payload(request_kwargs: dict, metadata: dict, messages: list[dict], *, tool_choice: dict) -> dict:
    payload = {
        "model": metadata["anthropic_model"],
        "max_tokens": min(request_kwargs["max_tokens"], _MAX_TOKENS_PER_TURN),
        "system": INITIAL_SYSTEM_PROMPT,
        "messages": list(messages),
        "tools": _ANTHROPIC_TOOLS,
        "tool_choice": tool_choice,
        "stream": False,
    }
    for key in ("temperature", "top_p", "top_k"):
        if request_kwargs.get(key) is not None:
            payload[key] = request_kwargs[key]
    if request_kwargs.get("stop") is not None:
        stop = request_kwargs["stop"]
        payload["stop_sequences"] = [stop] if isinstance(stop, str) else stop
    return payload


def _build_tool_result(tool_use: dict, turn_index: int) -> dict:
    result_text = MOCK_TOOL_RESULTS[turn_index % len(MOCK_TOOL_RESULTS)]
    content = [{"type": "text", "text": result_text}] if turn_index == 1 else result_text
    if turn_index == 2:
        content = _TOOL_RECOVERY_TEXT
    return {
        "type": "tool_result",
        "tool_use_id": tool_use["id"],
        "content": content,
    }


async def _post_complete(client, url: str, payload: dict, *, label: str) -> dict:
    for _ in range(_MAX_INCOMPLETE_TURN_RETRIES + 1):
        response = await client.post(url, json=payload)
        assert response.status_code == 200, f"{label} failed ({response.status_code}): {response.text}"
        body = response.json()
        if body.get("stop_reason") != "max_tokens":
            return body
    raise AssertionError(
        f"{label} exhausted {_MAX_INCOMPLETE_TURN_RETRIES} retries after stop_reason='max_tokens'"
    )


def _expected_driver_events(*, include_system: bool) -> list[str]:
    events = []
    for turn_index, result_event in enumerate(_TOOL_RESULT_EVENTS):
        if turn_index == 2 and include_system:
            events.append("anthropic_system")
        events.extend(("anthropic_tool_use", result_event, "anthropic_text"))
    return events


@_journal_verifier_assertions("anthropic_session_verify_agent.run_agent")
async def run_agent(base_url, prompt, request_kwargs, metadata, **kwargs):
    """Run three tool-use/result/text cycles and verify their canonical records."""
    tito_model = metadata["tito_model"]
    _assert_supported_tito_model(tito_model)
    intermediate_system_expectation = metadata["anthropic_intermediate_system_expectation"]
    messages = []
    events = []
    tool_uses_per_turn = []
    tool_use_count = 0

    async with httpx.AsyncClient(timeout=180) as client:
        server_url = base_url.rsplit("/sessions/", 1)[0]
        health_response = await client.get(f"{server_url}/health")
        assert health_response.status_code == 200, health_response.text
        route_supports = health_response.json().get("anthropic_intermediate_system_supported")
        assert type(route_supports) is bool, "session health did not report Anthropic intermediate-system capability"
        include_system = _verify_intermediate_system(tito_model, route_supports=route_supports)
        _assert_intermediate_system_expectation(intermediate_system_expectation, actual=include_system)

        for turn_index, user_prompt in enumerate(_TOOL_PROMPTS):
            if turn_index == 2 and include_system:
                messages.append({"role": "system", "content": SYSTEM_REMINDER_TEXT})
                events.append("anthropic_system")
            user_content = [{"type": "text", "text": user_prompt}] if turn_index == 2 else user_prompt
            messages.append({"role": "user", "content": user_content})

            payload = _build_payload(
                request_kwargs,
                metadata,
                messages,
                tool_choice={"type": "tool", "name": "get_weather"},
            )
            tool_body = await _post_complete(
                client,
                f"{base_url}/v1/messages",
                payload,
                label=f"Anthropic tool turn {turn_index + 1}",
            )
            tool_uses = _assert_anthropic_tool_response(tool_body)
            tool_uses_per_turn.append(tool_uses)
            [tool_use] = tool_uses
            tool_use_count += 1
            events.append("anthropic_tool_use")

            messages.append({"role": "assistant", "content": tool_body["content"]})
            tool_result = _build_tool_result(tool_use, turn_index)
            messages.append({"role": "user", "content": [tool_result]})
            events.append(_TOOL_RESULT_EVENTS[turn_index])

            payload = _build_payload(request_kwargs, metadata, messages, tool_choice={"type": "none"})
            text_body = await _post_complete(
                client,
                f"{base_url}/v1/messages",
                payload,
                label=f"Anthropic text turn {turn_index + 1}",
            )
            _assert_anthropic_text_response(text_body)
            messages.append({"role": "assistant", "content": text_body["content"]})
            events.append("anthropic_text")

        session_response = await client.get(base_url)
        assert session_response.status_code == 200, session_response.text
        _assert_canonical_records(
            session_response.json(),
            tool_uses_per_turn,
            include_system=include_system,
        )

    return {
        "endpoint": "anthropic",
        "driver_events": events,
        "request_count": _REQUEST_COUNT,
        "tool_use_count": tool_use_count,
        "tool_result_count": tool_use_count,
        "text_turn_count": len(_TOOL_PROMPTS),
        "tool_result_string_count": 2,
        "tool_result_list_count": 1,
        "intermediate_system_used": include_system,
    }


def _assert_anthropic_tool_response(body: dict) -> list[dict]:
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["stop_reason"] == "tool_use"
    tool_uses = [block for block in body["content"] if block["type"] == "tool_use"]
    assert len(tool_uses) == 1, f"Anthropic response must contain exactly one tool_use block: {body!r}"
    assert tool_uses[0]["name"] == "get_weather"
    assert isinstance(tool_uses[0]["input"], dict)
    return tool_uses


def _assert_anthropic_text_response(body: dict) -> None:
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["stop_reason"] == "end_turn"
    assert not any(block["type"] == "tool_use" for block in body["content"])
    assert any(block["type"] == "text" and block["text"] for block in body["content"])


def _assert_canonical_tool_calls(record: dict, tool_uses: list[dict]) -> None:
    tool_calls = record["response"]["choices"][0]["message"]["tool_calls"]
    calls_by_id = {tool_call["id"]: tool_call for tool_call in tool_calls}
    assert set(calls_by_id) == {tool_use["id"] for tool_use in tool_uses}
    for tool_use in tool_uses:
        tool_call = calls_by_id[tool_use["id"]]
        assert tool_call["function"]["name"] == tool_use["name"]
        assert json.loads(tool_call["function"]["arguments"]) == tool_use["input"]


def _assert_canonical_records(snapshot: dict, tool_uses_per_turn: list[list[dict]], *, include_system: bool) -> None:
    records = snapshot["records"]
    assert len(records) == _REQUEST_COUNT
    for record in records:
        assert record["path"] == "/v1/chat/completions"
        assert record["request"]["input_ids"]
    max_trim_tokens = snapshot["metadata"]["max_trim_tokens"]
    for previous, current in zip(records, records[1:], strict=False):
        previous_choice = previous["response"]["choices"][0]
        completion_ids = [item[1] for item in previous_choice["meta_info"]["output_token_logprobs"]]
        previous_ids = previous["request"]["input_ids"] + completion_ids
        current_ids = current["request"]["input_ids"]
        check_len = max(0, len(previous_ids) - max_trim_tokens)
        assert current_ids[:check_len] == previous_ids[:check_len]
    for record_index, current in enumerate(records[1:], start=1):
        replayed_assistants = [message for message in current["request"]["messages"] if message["role"] == "assistant"]
        stored_assistants = [record["response"]["choices"][0]["message"] for record in records[:record_index]]
        assert len(replayed_assistants) == len(stored_assistants)
        assert all(
            loose_tool_call_message_matches(stored, replayed)
            for stored, replayed in zip(stored_assistants, replayed_assistants, strict=True)
        )

    history_roles = []
    for turn_index, tool_uses in enumerate(tool_uses_per_turn):
        if turn_index == 2 and include_system:
            history_roles.append("system")
        history_roles.append("user")
        tool_record = records[turn_index * 2]
        assert [message["role"] for message in tool_record["request"]["messages"]] == [
            "system",
            *history_roles,
        ]
        _assert_canonical_tool_calls(tool_record, tool_uses)

        history_roles.extend(["assistant", *(["tool"] * len(tool_uses))])
        text_record = records[turn_index * 2 + 1]
        assert [message["role"] for message in text_record["request"]["messages"]] == [
            "system",
            *history_roles,
        ]
        text_message = text_record["response"]["choices"][0]["message"]
        assert text_message["content"]
        assert not text_message.get("tool_calls")
        history_roles.append("assistant")

    tree = snapshot["metadata"]["tree"]
    assert [node["parent"] for node in tree["nodes"]] == [None, 0, 1, 2, 3, 4]
    assert tree["leaves"] == [{"node_id": 5, "path_node_ids": [0, 1, 2, 3, 4, 5]}]
    last_choice = records[-1]["response"]["choices"][0]
    last_completion_ids = [item[1] for item in last_choice["meta_info"]["output_token_logprobs"]]
    assert snapshot["metadata"]["accumulated_token_ids"] == records[-1]["request"]["input_ids"] + last_completion_ids


@_journal_verifier_assertions("anthropic_session_verify_agent.generate")
async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    """Run the Anthropic agent, check hard TITO mismatches, and write metrics."""
    tito_model = input.args.tito_model
    _assert_supported_tito_model(tito_model)
    intermediate_system_expectation = input.args.anthropic_intermediate_system_expectation
    input.sample.metadata["anthropic_model"] = input.args.hf_checkpoint
    input.sample.metadata["tito_model"] = tito_model
    input.sample.metadata["anthropic_intermediate_system_expectation"] = intermediate_system_expectation
    output = await _base_generate(input)

    samples = output.samples if isinstance(output.samples, list) else [output.samples]
    events_per_sample = [sample.metadata.get("driver_events", []) for sample in samples]
    allowed_roles = list(fixed_template_append_roles(tito_model))
    _verify_tito_samples(samples, events_per_sample, allowed_roles=allowed_roles)
    if len(samples) != 1:
        raise AssertionError(f"Anthropic per-model e2e: expected one linear sample, got {len(samples)}")
    include_system = samples[0].metadata.get("intermediate_system_used")
    if type(include_system) is not bool:
        raise AssertionError("Anthropic per-model e2e: missing intermediate-system capability result")
    _assert_intermediate_system_expectation(intermediate_system_expectation, actual=include_system)
    if include_system and "system" not in fixed_template_append_roles(tito_model):
        raise AssertionError(f"Anthropic per-model e2e: {tito_model!r} used an unsupported intermediate system")
    expected_events = _expected_driver_events(include_system=include_system)
    expected_counters = {
        "request_count": _REQUEST_COUNT,
        "tool_use_count": len(_TOOL_PROMPTS),
        "tool_result_count": len(_TOOL_PROMPTS),
        "text_turn_count": len(_TOOL_PROMPTS),
        "tool_result_string_count": 2,
        "tool_result_list_count": 1,
    }
    for i, sample in enumerate(samples):
        if sample.metadata.get("endpoint") != "anthropic":
            raise AssertionError(f"Anthropic per-model e2e: sample {i} did not retain agent metadata")
        if events_per_sample[i] != expected_events:
            raise AssertionError(
                f"Anthropic per-model e2e: sample {i} events={events_per_sample[i]!r}, "
                f"expected {expected_events!r}"
            )
        for key, expected in expected_counters.items():
            if sample.metadata.get(key) != expected:
                raise AssertionError(
                    f"Anthropic per-model e2e: sample {i} {key}={sample.metadata.get(key)!r}, expected {expected}"
                )
        if sample.metadata.get("leaf", {}).get("path_node_ids") != list(range(_REQUEST_COUNT)):
            raise AssertionError(f"Anthropic per-model e2e: sample {i} did not retain the linear six-turn leaf")

    logger.info("Anthropic endpoint verified: samples=%d, requests_per_sample=%d", len(samples), _REQUEST_COUNT)
    return output


def _add_arguments(parser):
    _session_verify_generate.add_arguments(parser)
    parser.add_argument(
        "--anthropic-intermediate-system-expectation",
        required=True,
        choices=sorted(_INTERMEDIATE_SYSTEM_EXPECTATIONS),
        help=("Require or forbid the intermediate system turn in the Anthropic " "session-verification trajectory."),
    )


generate.add_arguments = _add_arguments
