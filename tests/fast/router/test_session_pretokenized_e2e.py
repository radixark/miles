"""Representative session-layer smoke tests for miles-maintained fixed templates.

These tests intentionally stay narrow:

- only bundled fixed templates maintained by miles
- only tool-only multi-turn session flow
- only session/TITO plumbing + mismatch taxonomy checks

Detailed template correctness remains covered by the lower-level chat-template
tests in ``tests/fast/utils/chat_template_utils/``.
"""

from __future__ import annotations

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-fast")

from dataclasses import dataclass

import pytest
import requests
from tests.fast.router.session_pretokenized_test_utils import (
    ScriptedBackendTurn,
    ScriptedChatBackend,
    compute_local_session_mismatch,
    fetch_session_payload,
    forbidden_mismatches,
    load_test_tokenizer,
    make_router_env,
    teardown_router_env,
)

from miles.utils.chat_template_utils import TITOTokenizerType, resolve_fixed_chat_template
from miles.utils.test_utils.mock_trajectories import LongChainTrajectory, build_trajectory


@dataclass(frozen=True)
class FixedTemplateSmokeConfig:
    name: str
    hf_checkpoint: str
    chat_template_path: str
    tito_model: str


FIXED_TEMPLATE_SMOKE_CONFIGS: tuple[FixedTemplateSmokeConfig, ...] = (
    FixedTemplateSmokeConfig(
        name="qwen3-fixed",
        hf_checkpoint="Qwen/Qwen3-0.6B",
        chat_template_path=resolve_fixed_chat_template(TITOTokenizerType.QWEN3, ["tool"])[0],
        tito_model=TITOTokenizerType.QWEN3.value,
    ),
    FixedTemplateSmokeConfig(
        name="qwen3.5-fixed",
        hf_checkpoint="Qwen/Qwen3.5-0.8B",
        chat_template_path=resolve_fixed_chat_template(TITOTokenizerType.QWEN35, ["tool"])[0],
        tito_model=TITOTokenizerType.QWEN35.value,
    ),
    FixedTemplateSmokeConfig(
        name="qwen3-thinking2507-fixed",
        hf_checkpoint="Qwen/Qwen3-4B-Thinking-2507",
        chat_template_path=resolve_fixed_chat_template(TITOTokenizerType.QWENNEXT, ["tool"])[0],
        tito_model=TITOTokenizerType.QWENNEXT.value,
    ),
    FixedTemplateSmokeConfig(
        name="qwen3-next-thinking-fixed",
        hf_checkpoint="Qwen/Qwen3-Next-80B-A3B-Thinking",
        chat_template_path=resolve_fixed_chat_template(TITOTokenizerType.QWENNEXT, ["tool"])[0],
        tito_model=TITOTokenizerType.QWENNEXT.value,
    ),
)


def _get_followup_messages_after_assistant(full_messages: list[dict], assistant_idx: int) -> list[dict]:
    followup = []
    i = assistant_idx + 1
    while i < len(full_messages) and full_messages[i]["role"] != "assistant":
        followup.append(full_messages[i])
        i += 1
    return followup


def _remap_followup_messages(followup_msgs: list[dict], response_tool_calls: list[dict]) -> list[dict]:
    remapped = []
    tool_idx = 0
    for msg in followup_msgs:
        new_msg = dict(msg)
        if msg["role"] == "tool":
            if tool_idx < len(response_tool_calls):
                new_msg["tool_call_id"] = response_tool_calls[tool_idx]["id"]
            tool_idx += 1
        remapped.append(new_msg)
    return remapped


@pytest.mark.parametrize("config", FIXED_TEMPLATE_SMOKE_CONFIGS, ids=[c.name for c in FIXED_TEMPLATE_SMOKE_CONFIGS])
def test_bundled_fixed_template_session_smoke(config: FixedTemplateSmokeConfig):
    assert config.chat_template_path is not None, f"{config.name} should resolve to a bundled fixed template"

    try:
        tokenizer = load_test_tokenizer(config.hf_checkpoint, config.chat_template_path)
    except (ValueError, OSError) as exc:
        pytest.skip(f"Cannot load tokenizer for {config.hf_checkpoint}: {exc}")

    trajectory = build_trajectory(tokenizer, LongChainTrajectory)
    scripted_turns = [
        ScriptedBackendTurn(
            response_message={**turn.assistant_message, "content": turn.assistant_message.get("content") or ""},
            render_message=turn.assistant_message,
        )
        for turn in trajectory.turns
    ]
    backend = ScriptedChatBackend(tokenizer, scripted_turns)
    backend.start()
    env = make_router_env(
        backend,
        hf_checkpoint=config.hf_checkpoint,
        chat_template_path=config.chat_template_path,
        tito_model=config.tito_model,
        allowed_append_roles=["tool"],
    )

    try:
        backend.reset_stats()
        session_id = requests.post(f"{env.url}/sessions", timeout=5.0).json()["session_id"]
        assistant_indices = [i for i, m in enumerate(trajectory.full_messages) if m["role"] == "assistant"]

        accumulated_messages: list[dict] = []
        for turn_idx, turn in enumerate(trajectory.turns):
            messages = list(accumulated_messages) if turn_idx > 0 else list(turn.request_messages)
            if turn_idx == 0:
                accumulated_messages = list(messages)

            payload = {"messages": messages, "tools": trajectory.tools}
            response = requests.post(
                f"{env.url}/sessions/{session_id}/v1/chat/completions",
                json=payload,
                timeout=10.0,
            )
            assert response.status_code == 200, f"{config.name} turn {turn_idx} failed: {response.text}"

            body = response.json()
            assert len(body["choices"]) == 1
            assert "prompt_token_ids" in body["choices"][0]
            if turn_idx > 0:
                assert "input_ids" in backend.request_log[turn_idx], f"{config.name} turn {turn_idx} missing input_ids"

            assistant_msg = body["choices"][0]["message"]
            session_messages = list(messages) + [assistant_msg]

            session_payload = fetch_session_payload(env.url, session_id)
            metadata = session_payload["metadata"]
            remote_mismatch = metadata.get("tito_session_mismatch", [])
            local_mismatch = compute_local_session_mismatch(
                tokenizer,
                tito_model=config.tito_model,
                allowed_append_roles=["tool"],
                messages=session_messages,
                accumulated_token_ids=metadata["accumulated_token_ids"],
                tools=trajectory.tools,
            )
            assert remote_mismatch == local_mismatch
            assert (
                forbidden_mismatches(remote_mismatch) == []
            ), f"{config.name} turn {turn_idx} has forbidden mismatch types: {remote_mismatch}"

            accumulated_messages.append(assistant_msg)
            ass_idx = assistant_indices[turn_idx]
            followup_msgs = _get_followup_messages_after_assistant(trajectory.full_messages, ass_idx)
            response_tool_calls = assistant_msg.get("tool_calls") or []
            accumulated_messages.extend(_remap_followup_messages(followup_msgs, response_tool_calls))

        final_session_payload = fetch_session_payload(env.url, session_id)
        records = final_session_payload["records"]
        assert len(records) == len(trajectory.turns)
        assert all(r["status_code"] == 200 for r in records)
        assert all(r["path"] == "/v1/chat/completions" for r in records)
    finally:
        teardown_router_env(env)
