"""Custom agent function for the session-server tool-call e2e test.

Performs a multi-turn tool-calling conversation through the session proxy and
verifies that the session's prompt_token_ids, when decoded, exactly match
the text produced by locally applying the chat template to the same messages.

The agent is loaded at runtime by ``agentic_tool_call.generate`` via
``--custom-agent-function-path tests.e2e.sglang.session_tool_agent.run_agent``.
"""

import logging
import os

import httpx

from miles.utils.chat_template_utils import apply_chat_template, try_get_fixed_chat_template
from miles.utils.processing_utils import load_tokenizer

logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("SGLANG_E2E_MODEL_PATH", "Qwen/Qwen3-4B")

MAX_TOOL_TURNS = 8
MAX_RETRIES = 2

RETRY_SYSTEM_MESSAGE = (
    "Your previous response did not include a valid tool call or a final answer. "
    "Please either call a tool or provide your answer in "
    "<final_answer>...</final_answer> tags."
)

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


def _load_tokenizer(model_path: str):
    """Load tokenizer with the fixed chat template applied (if one exists)."""
    template_path = try_get_fixed_chat_template(model_path)
    return load_tokenizer(model_path, chat_template_path=template_path, trust_remote_code=True)


def _is_task_complete(assistant_msg: dict) -> bool:
    """Check if the assistant has produced a final answer."""
    content = assistant_msg.get("content") or ""
    return "<final_answer>" in content and "</final_answer>" in content


def _extract_tool_calls(assistant_msg: dict) -> list[dict] | None:
    """Return structured tool_calls from the assistant message, or None.

    Only trusts the structured ``tool_calls`` field populated by sglang's
    tool-call parser.  No fallback parsing — if the parser didn't produce
    structured output, the caller should treat it as a failed tool call
    and retry via a system message.
    """
    if assistant_msg.get("tool_calls"):
        return assistant_msg["tool_calls"]
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
    messages: list[dict],
    prev_prompt_ids: list[int] | None,
    tokenizer,
) -> bool:
    """Check TITO text match and prefix monotonicity. Returns True if text matched."""
    # --- DEBUG: dump message trajectory ---
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        rc = msg.get("reasoning_content")
        content = msg.get("content")
        tc = "yes" if msg.get("tool_calls") else "no"
        logger.info(
            "Turn %d MSG[%d] role=%s reasoning_content=%r content=%r tool_calls=%s",
            turn,
            i,
            role,
            rc[:80] if isinstance(rc, str) and len(rc) > 80 else rc,
            content[:80] if isinstance(content, str) and len(content) > 80 else content,
            tc,
        )

    session_text = tokenizer.decode(session_prompt_ids)

    expected_text = apply_chat_template(
        messages,
        tokenizer=tokenizer,
        tools=TOOLS,
        add_generation_prompt=True,
        tokenize=False,
    )

    text_matched = session_text == expected_text

    # --- DEBUG: always log the tail (generation prompt area) ---
    logger.info(
        "Turn %d TAIL COMPARE:\n" "  session_text[-80:] = %r\n" "  expected_text[-80:] = %r",
        turn,
        session_text[-80:],
        expected_text[-80:],
    )

    if not text_matched:
        first_char_diff = next(
            (i for i, (a, b) in enumerate(zip(session_text, expected_text, strict=False)) if a != b),
            min(len(session_text), len(expected_text)),
        )
        ctx_lo = max(0, first_char_diff - 80)
        ctx_hi = min(max(len(session_text), len(expected_text)), first_char_diff + 80)
        logger.warning(
            "TITO TEXT MISMATCH on turn %d (non-fatal):\n"
            "Session decoded length: %d chars (%d tokens)\n"
            "Template rendered length: %d chars\n"
            "First char diff at index %d\n"
            "\n--- Session decoded text around diff ---\n%r\n"
            "\n--- Template rendered text around diff ---\n%r\n"
            "\n--- Session FULL text (last 500 chars) ---\n%r\n"
            "\n--- Template FULL text (last 500 chars) ---\n%r",
            turn,
            len(session_text),
            len(session_prompt_ids),
            len(expected_text),
            first_char_diff,
            session_text[ctx_lo:ctx_hi],
            expected_text[ctx_lo:ctx_hi],
            session_text,
            expected_text,
        )

    if prev_prompt_ids is not None:
        n = len(prev_prompt_ids)
        assert session_prompt_ids[:n] == prev_prompt_ids, (
            f"Prefix token mismatch on turn {turn}: previous turn's prompt_ids "
            f"({n} tokens) is not a prefix of this turn's "
            f"({len(session_prompt_ids)} tokens)."
        )

    return text_matched


async def run_agent(base_url, prompt, request_kwargs, metadata, **kwargs):
    """Multi-turn tool-call agent that verifies session prompt correctness.

    On every turn, decodes the session's prompt_token_ids and compares with
    locally rendered chat template text — they must be identical.
    """
    messages = list(prompt)

    rk = {k: v for k, v in request_kwargs.items() if k not in ("tools",)}

    model_path = metadata.get("model_path", MODEL_NAME) if metadata else MODEL_NAME
    tokenizer = _load_tokenizer(model_path)

    # --- DEBUG: confirm which chat template is loaded ---
    _tmpl = getattr(tokenizer, "chat_template", None) or ""
    _gen_prompt_sample = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True
    )
    logger.info(
        "TEMPLATE CHECK: model_path=%s, template_len=%d, gen_prompt_ending=%r",
        model_path,
        len(_tmpl),
        _gen_prompt_sample[-60:],
    )

    turns_completed = 0
    total_tool_calls = 0
    prev_prompt_ids: list[int] | None = None
    consecutive_retries = 0
    local_total = 0
    local_mismatch = 0

    async with httpx.AsyncClient(
        timeout=180, limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100)
    ) as client:
        for turn in range(1, MAX_TOOL_TURNS + 1):
            label = f"Session Turn {turn}"
            resp_data, session_ids = await _chat(
                client,
                base_url,
                messages,
                rk,
                label=label,
                tool_choice="auto",
            )
            logger.info("%s: %d prompt tokens", label, len(session_ids))

            text_matched = _verify_turn(
                turn,
                session_ids,
                messages,
                prev_prompt_ids,
                tokenizer,
            )
            prev_prompt_ids = list(session_ids)
            local_total += 1
            if not text_matched:
                local_mismatch += 1
            logger.info(
                "Turn %d verified: text_match=%s, prefix_ok=True (%d tokens)",
                turn,
                text_matched,
                len(session_ids),
            )

            turns_completed = turn

            assistant_msg = resp_data["choices"][0]["message"]

            # --- DEBUG: dump assistant message fields ---
            logger.info(
                "Turn %d ASSISTANT MSG: reasoning_content=%r, content=%r, " "tool_calls=%s, all_keys=%s",
                turn,
                assistant_msg.get("reasoning_content"),
                assistant_msg.get("content"),
                "present" if assistant_msg.get("tool_calls") else "absent",
                list(assistant_msg.keys()),
            )

            messages.append(assistant_msg)

            if _is_task_complete(assistant_msg):
                logger.info("Turn %d: task complete, ending loop", turn)
                break

            tool_calls = _extract_tool_calls(assistant_msg)
            if tool_calls:
                for i, tc in enumerate(tool_calls):
                    mock_idx = (total_tool_calls + i) % len(MOCK_TOOL_RESULTS)
                    messages.append(
                        {
                            "role": "tool",
                            "content": MOCK_TOOL_RESULTS[mock_idx],
                            "tool_call_id": tc["id"],
                        }
                    )
                total_tool_calls += len(tool_calls)
                logger.info(
                    "Turn %d: appended %d tool result(s), total tool calls so far: %d",
                    turn,
                    len(tool_calls),
                    total_tool_calls,
                )
                consecutive_retries = 0
            else:
                consecutive_retries += 1
                if consecutive_retries > MAX_RETRIES:
                    logger.warning("Turn %d: exceeded %d retries, ending loop", turn, MAX_RETRIES)
                    break
                messages.append({"role": "system", "content": RETRY_SYSTEM_MESSAGE})
                logger.info(
                    "Turn %d: no tool calls and not complete, appended retry system message (%d/%d)",
                    turn,
                    consecutive_retries,
                    MAX_RETRIES,
                )

    MIN_TOOL_CALLS = 3
    if total_tool_calls < MIN_TOOL_CALLS:
        logger.warning(
            "Only made %d successful tool call(s) in %d turn(s), "
            "need >= %d for meaningful multi-turn prefix verification",
            total_tool_calls,
            turns_completed,
            MIN_TOOL_CALLS,
        )

    logger.info(
        "Agent done: %d turns, %d tool_calls, tito_mismatch=%d/%d",
        turns_completed,
        total_tool_calls,
        local_mismatch,
        local_total,
    )

    return {
        "turns_completed": turns_completed,
        "total_tool_calls": total_tool_calls,
        "tito_session_mismatch": 1 if local_mismatch > 0 else 0,
        "tito_local_total": local_total,
        "tito_local_mismatch": local_mismatch,
    }
