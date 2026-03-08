"""Custom agent function for the session-server tool-call e2e test.

Performs a multi-turn tool-calling conversation through the session proxy and
verifies the pretokenized TITO prefix invariant by comparing prompt_token_ids
obtained via the session route (with pretokenized injection) against the
non-session route (full tokenization, no pretokenized prefix).

If both routes produce the same prompt_token_ids on turn 2+, the fixed chat
template is append-only and pretokenized TITO is correct.

The agent is loaded at runtime by ``agentic_tool_call.generate`` via
``--custom-agent-function-path tests.e2e.sglang.session_tool_agent.run_agent``.
"""

import json
import logging
import os

import httpx

logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("SGLANG_E2E_MODEL_PATH", "Qwen/Qwen3-0.6B")

MAX_TOOL_TURNS = 4

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g. Beijing",
                    },
                },
                "required": ["location"],
            },
        },
    },
]

MOCK_TOOL_RESULTS = [
    '{"temperature_celsius": 22, "condition": "sunny", "humidity": 45}',
    '{"temperature_celsius": 15, "condition": "cloudy", "humidity": 70}',
    '{"temperature_celsius": 30, "condition": "rainy", "humidity": 90}',
    '{"temperature_celsius": 8, "condition": "snowy", "humidity": 85}',
]


def _router_url_from_base(base_url: str) -> str:
    """Extract the router root URL from the session base_url.

    base_url looks like ``http://host:port/sessions/{session_id}``.
    """
    parts = base_url.split("/sessions/")
    return parts[0]


def _extract_tool_calls(assistant_msg: dict) -> list[dict] | None:
    """Return structured tool_calls from the assistant message, or None."""
    if assistant_msg.get("tool_calls"):
        return assistant_msg["tool_calls"]

    # Fallback: small models may output tool calls as raw JSON text
    # while sglang sets finish_reason=tool_calls without parsing them.
    content = (assistant_msg.get("content") or "").strip()
    if content.endswith("<|im_end|>"):
        content = content[: -len("<|im_end|>")].strip()
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list) and parsed and "name" in parsed[0]:
            return [
                {
                    "id": f"call_{i:05d}",
                    "type": "function",
                    "function": {
                        "name": item["name"],
                        "arguments": json.dumps(item.get("arguments") or item.get("parameters", {})),
                    },
                }
                for i, item in enumerate(parsed)
            ]
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return None


async def _chat(
    client: httpx.AsyncClient,
    url: str,
    messages,
    rk,
    label="",
    tool_choice=None,
):
    """Send a chat completions request and return (response_json, prompt_token_ids)."""
    payload = {
        "messages": messages,
        "tools": TOOLS,
        "chat_template_kwargs": {"enable_thinking": False},
        **rk,
    }
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    resp = await client.post(f"{url}/v1/chat/completions", json=payload)
    assert resp.status_code == 200, f"{label} failed ({resp.status_code}): {resp.text}"
    data = resp.json()
    prompt_ids = data["choices"][0].get("prompt_token_ids")
    assert prompt_ids is not None, f"{label}: prompt_token_ids missing"
    return data, prompt_ids


def _verify_turn(
    turn: int,
    session_prompt_ids: list[int],
    direct_prompt_ids: list[int],
    prev_prompt_ids: list[int] | None,
):
    """Assert strict prefix invariant for one turn."""
    assert session_prompt_ids == direct_prompt_ids, (
        f"TITO PREFIX INVARIANT VIOLATED on turn {turn}!\n"
        f"Session route (pretokenized): {len(session_prompt_ids)} tokens\n"
        f"Direct route (full tokenize): {len(direct_prompt_ids)} tokens\n"
        f"First diff at index {_first_diff(session_prompt_ids, direct_prompt_ids)}"
    )

    if prev_prompt_ids is not None:
        assert len(session_prompt_ids) > len(prev_prompt_ids), (
            f"Turn {turn} prompt ({len(session_prompt_ids)} tokens) should be "
            f"strictly longer than previous turn ({len(prev_prompt_ids)} tokens)"
        )
        assert session_prompt_ids[: len(prev_prompt_ids)] == prev_prompt_ids, (
            f"Prefix mismatch on turn {turn}: previous turn's prompt_token_ids "
            f"({len(prev_prompt_ids)} tokens) is not a prefix of this turn's "
            f"({len(session_prompt_ids)} tokens). "
            f"First diff at index "
            f"{_first_diff(prev_prompt_ids, session_prompt_ids[:len(prev_prompt_ids)])}"
        )


async def run_agent(base_url, prompt, request_kwargs, metadata, **kwargs):
    """Multi-turn tool-call agent with pretokenized prefix invariant checks.

    Verification strategy:
    After multi-turn conversation through the session proxy (which injects
    pretokenized_token_ids on turn 2+), send the same accumulated messages
    via the non-session route (direct to sglang, no pretokenized injection).
    If the prompt_token_ids match exactly, the template is append-only and
    TITO is correct.
    """
    router_url = _router_url_from_base(base_url)
    messages = list(prompt)

    rk = {k: v for k, v in request_kwargs.items() if k not in ("tools",)}
    rk.setdefault("return_prompt_token_ids", True)
    rk.setdefault("logprobs", True)

    turns_completed = 0
    prev_prompt_ids = None

    async with httpx.AsyncClient(timeout=180) as client:
        for turn in range(1, MAX_TOOL_TURNS + 1):
            label = f"Session Turn {turn}"
            resp_data, session_ids = await _chat(
                client,
                base_url,
                messages,
                rk,
                label=label,
                tool_choice="required",
            )
            logger.info("%s: %d prompt tokens", label, len(session_ids))

            if turn >= 2:
                _, direct_ids = await _chat(
                    client,
                    router_url,
                    messages,
                    rk,
                    label=f"Direct Turn {turn}",
                    tool_choice="required",
                )
                logger.info("Direct Turn %d: %d prompt tokens", turn, len(direct_ids))

                _verify_turn(turn, session_ids, direct_ids, prev_prompt_ids)
                logger.info(
                    "Turn %d VERIFIED: session (%d) == direct (%d), " "prefix from turn %d (%d tokens) preserved",
                    turn,
                    len(session_ids),
                    len(direct_ids),
                    turn - 1,
                    len(prev_prompt_ids) if prev_prompt_ids else 0,
                )

            prev_prompt_ids = session_ids
            turns_completed = turn

            assistant_msg = resp_data["choices"][0]["message"]
            messages.append(assistant_msg)

            tool_calls = _extract_tool_calls(assistant_msg)
            if not tool_calls:
                logger.warning(
                    "Turn %d: no tool calls despite tool_choice=required – " "ending conversation early",
                    turn,
                )
                break

            mock_result = MOCK_TOOL_RESULTS[(turn - 1) % len(MOCK_TOOL_RESULTS)]
            for tc in tool_calls:
                messages.append(
                    {
                        "role": "tool",
                        "content": mock_result,
                        "tool_call_id": tc["id"],
                    }
                )
            logger.info(
                "Turn %d: appended %d tool result(s) with mock data [%d]",
                turn,
                len(tool_calls),
                (turn - 1) % len(MOCK_TOOL_RESULTS),
            )

    verified = turns_completed >= 3
    if not verified:
        raise AssertionError(
            f"Only completed {turns_completed} tool-call turn(s), need >= 3 "
            f"for meaningful multi-turn prefix invariant verification"
        )
    return {
        "turns_completed": turns_completed,
        "prefix_invariant_verified": verified,
    }


def _first_diff(a: list, b: list) -> int | str:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return "none"
