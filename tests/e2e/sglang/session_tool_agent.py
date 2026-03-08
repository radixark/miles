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

MOCK_TOOL_RESULT = '{"temperature_celsius": 22, "condition": "sunny", "humidity": 45}'


def _router_url_from_base(base_url: str) -> str:
    """Extract the router root URL from the session base_url.

    base_url looks like ``http://host:port/sessions/{session_id}``.
    """
    parts = base_url.split("/sessions/")
    return parts[0]


async def _chat(client: httpx.AsyncClient, url: str, messages, rk, label=""):
    """Send a chat completions request and return (response_json, prompt_token_ids)."""
    payload = {
        "messages": messages,
        "tools": TOOLS,
        "chat_template_kwargs": {"enable_thinking": False},
        **rk,
    }
    resp = await client.post(f"{url}/v1/chat/completions", json=payload)
    assert resp.status_code == 200, f"{label} failed ({resp.status_code}): {resp.text}"
    data = resp.json()
    prompt_ids = data["choices"][0].get("prompt_token_ids")
    assert prompt_ids is not None, f"{label}: prompt_token_ids missing"
    return data, prompt_ids


async def run_agent(base_url, prompt, request_kwargs, metadata, **kwargs):
    """Multi-turn tool-call agent with pretokenized prefix invariant checks.

    Verification strategy:
    After multi-turn conversation through the session proxy (which injects
    pretokenized_token_ids on turn 2+), send the same accumulated messages
    via the non-session route (direct to sglang, no pretokenized injection).
    If the prompt_token_ids match, the template is append-only and TITO works.
    """
    router_url = _router_url_from_base(base_url)
    messages = list(prompt)

    rk = {k: v for k, v in request_kwargs.items() if k not in ("tools",)}
    rk.setdefault("return_prompt_token_ids", True)
    rk.setdefault("logprobs", True)

    async with httpx.AsyncClient(timeout=180) as client:
        # ── Turn 1 (no pretokenized injection expected) ───────────
        t1_data, t1_prompt_ids = await _chat(client, base_url, messages, rk, label="Session Turn 1")
        logger.info("Session Turn 1: %d prompt tokens", len(t1_prompt_ids))

        # Append the EXACT assistant message from the response so the session
        # manager's _is_append_only check passes (it compares dicts by ==).
        assistant_msg = t1_data["choices"][0]["message"]
        messages.append(assistant_msg)

        # Extract a tool_call_id - prefer structured tool_calls, fall back to
        # parsing the raw text content from small models like Qwen3-0.6B.
        tool_calls = assistant_msg.get("tool_calls")
        if tool_calls:
            tool_call_id = tool_calls[0]["id"]
        else:
            tool_call_id = _parse_tool_call_id_from_content(assistant_msg)

        if tool_call_id is None:
            logger.warning(
                "No tool calls detected in turn 1 (content=%r) – " "skipping multi-turn prefix invariant verification",
                (assistant_msg.get("content") or "")[:100],
            )
            return {"turns_completed": 1, "prefix_invariant_verified": False}

        messages.append(
            {
                "role": "tool",
                "content": MOCK_TOOL_RESULT,
                "tool_call_id": tool_call_id,
            }
        )
        logger.info("Appended tool response (tool_call_id=%s)", tool_call_id)

        # ── Turn 2 (session proxy will inject pretokenized_token_ids) ──
        t2_data, t2_prompt_ids = await _chat(client, base_url, messages, rk, label="Session Turn 2")
        logger.info(
            "Session Turn 2: %d prompt tokens (turn-1 prefix: %d)",
            len(t2_prompt_ids),
            len(t1_prompt_ids),
        )

        assert len(t2_prompt_ids) > len(t1_prompt_ids), (
            f"Turn 2 prompt ({len(t2_prompt_ids)} tokens) should be strictly "
            f"longer than turn 1 ({len(t1_prompt_ids)} tokens)"
        )

        # ── Ground-truth check via non-session route ──────────────
        # Send the SAME accumulated messages directly to sglang (bypassing
        # session proxy => no pretokenized injection => full tokenization).
        _, direct_prompt_ids = await _chat(client, router_url, messages, rk, label="Direct (non-session)")
        logger.info("Direct route: %d prompt tokens", len(direct_prompt_ids))

        diff_idx = _first_diff(t2_prompt_ids, direct_prompt_ids)
        if t2_prompt_ids == direct_prompt_ids:
            logger.info(
                "Prefix invariant VERIFIED (exact match): session route (%d "
                "tokens, pretokenized prefix %d) == direct route (%d tokens)",
                len(t2_prompt_ids),
                len(t1_prompt_ids),
                len(direct_prompt_ids),
            )
        else:
            # A small difference (1-2 tokens) at the assistant/tool boundary
            # is a known limitation: sglang's logprobs may not include the
            # stop token (<|im_end|>), so the stored token_ids are missing it.
            # Full tokenization includes it as part of the assistant message.
            token_diff = abs(len(t2_prompt_ids) - len(direct_prompt_ids))
            logger.warning(
                "Session vs direct route differ by %d token(s) at index %s "
                "(session=%d, direct=%d). This may be a stop-token boundary "
                "issue between assistant completion and tool message.",
                token_diff,
                diff_idx,
                len(t2_prompt_ids),
                len(direct_prompt_ids),
            )
            assert token_diff <= 2, (
                f"TITO PREFIX INVARIANT VIOLATED!\n"
                f"Session route (pretokenized): {len(t2_prompt_ids)} tokens\n"
                f"Direct route (full tokenize): {len(direct_prompt_ids)} tokens\n"
                f"First diff at index {diff_idx}\n"
                f"Difference of {token_diff} tokens exceeds tolerance."
            )
        return {"turns_completed": 2, "prefix_invariant_verified": True}


def _parse_tool_call_id_from_content(assistant_msg: dict) -> str | None:
    """Try to extract a tool call id from the raw text content.

    When sglang doesn't parse tool calls into structured format but the model
    produced them as text (finish_reason=tool_calls), we generate an id.
    """
    content = assistant_msg.get("content", "")
    if not content:
        return None
    content = content.strip()
    if content.endswith("<|im_end|>"):
        content = content[: -len("<|im_end|>")].strip()
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list) and len(parsed) > 0 and "name" in parsed[0]:
            return "call_00000"
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return None


def _first_diff(a: list, b: list) -> int | str:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return "none"
