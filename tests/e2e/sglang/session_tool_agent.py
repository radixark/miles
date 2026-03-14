"""Custom agent function for the session-server tool-call e2e test.

Performs a multi-turn tool-calling conversation through the session proxy and
verifies token-level correctness after all turns complete.  The full message
history is re-tokenized locally and compared against the actual inference
token IDs using ``TokenSeqComparator``.

The agent is loaded at runtime by ``agentic_tool_call.generate`` via
``--custom-agent-function-path tests.e2e.sglang.session_tool_agent.run_agent``.
"""

import logging
import os

import httpx

from miles.utils.chat_template_utils import (
    TokenSeqComparator,
    apply_chat_template,
    get_tito_tokenizer,
    try_get_fixed_chat_template,
)
from miles.utils.processing_utils import load_tokenizer

TITO_MODEL_TYPE = os.environ.get("MILES_TITO_MODEL", "default")

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


async def run_agent(base_url, prompt, request_kwargs, metadata, **kwargs):
    """Multi-turn tool-call agent that verifies token-level correctness.

    Runs the full agentic loop first, then after all turns complete,
    re-tokenizes the entire message history and compares against the
    actual inference token IDs using ``TokenSeqComparator``.
    Asserts no ``MismatchType.SPECIAL_TOKEN`` mismatches.
    """
    messages = list(prompt)

    rk = {k: v for k, v in request_kwargs.items() if k not in ("tools",)}

    model_path = metadata.get("model_path", MODEL_NAME) if metadata else MODEL_NAME
    tokenizer = _load_tokenizer(model_path)

    turns_completed = 0
    total_tool_calls = 0
    consecutive_retries = 0

    # Accumulated across turns for final comparison.
    last_prompt_ids: list[int] | None = None
    last_completion_ids: list[int] | None = None

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

            # Extract completion token IDs from output_token_logprobs.
            meta_info = resp_data["choices"][0].get("meta_info", {})
            output_logprobs = meta_info.get("output_token_logprobs", [])
            completion_ids = [t[1] for t in output_logprobs]

            last_prompt_ids = list(session_ids)
            last_completion_ids = completion_ids

            turns_completed = turn

            assistant_msg = resp_data["choices"][0]["message"]
            logger.info(
                "Turn %d: content=%r, tool_calls=%s",
                turn,
                (assistant_msg.get("content") or "")[:80],
                "present" if assistant_msg.get("tool_calls") else "absent",
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

    # ------------------------------------------------------------------
    # Post-loop verification: compare full token sequences
    # ------------------------------------------------------------------

    MIN_TOOL_CALLS = 3
    if total_tool_calls < MIN_TOOL_CALLS:
        logger.warning(
            "Only made %d successful tool call(s) in %d turn(s), " "need >= %d for meaningful multi-turn verification",
            total_tool_calls,
            turns_completed,
            MIN_TOOL_CALLS,
        )

    assert last_prompt_ids is not None, "No turns completed"
    assert last_completion_ids is not None, "No completion tokens collected"

    # Actual token IDs used in inference: prompt + completion.
    actual_ids = last_prompt_ids + last_completion_ids

    # Expected: re-tokenize the full message history (including last
    # assistant response) without generation prompt.
    expected_ids = apply_chat_template(
        messages,
        tokenizer=tokenizer,
        tools=TOOLS,
        add_generation_prompt=False,
        tokenize=True,
    )

    tito_tokenizer = get_tito_tokenizer(tokenizer, tokenizer_type=TITO_MODEL_TYPE)
    trim_trailing_ids = tito_tokenizer.get_trim_trailing_ids()

    comparator = TokenSeqComparator(tokenizer)
    mismatches = comparator.compare_sequences(
        expected_ids,
        actual_ids,
        trim_trailing_ids=trim_trailing_ids or None,
    )

    for m in mismatches:
        logger.error(
            "Mismatch [%s] segment=%d: expected=%r actual=%r detail=%s",
            m.type.value,
            m.segment_index,
            m.expected_text[:120],
            m.actual_text[:120],
            m.detail,
        )
        e, a = m.expected_text, m.actual_text
        for i, (ec, ac) in enumerate(zip(e, a, strict=False)):
            if ec != ac:
                ctx = 40
                logger.error(
                    "  first diff at char %d: expected=...%r... actual=...%r...",
                    i,
                    e[max(0, i - ctx) : i + ctx],
                    a[max(0, i - ctx) : i + ctx],
                )
                break
        else:
            if len(e) != len(a):
                shorter = min(len(e), len(a))
                logger.error(
                    "  length diff: expected=%d actual=%d, tail expected=%r actual=%r",
                    len(e),
                    len(a),
                    e[shorter : shorter + 80],
                    a[shorter : shorter + 80],
                )

    logger.info(
        "Agent done: %d turns, %d tool_calls, %d mismatches, " "%d expected tokens, %d actual tokens",
        turns_completed,
        total_tool_calls,
        len(mismatches),
        len(expected_ids),
        len(actual_ids),
    )

    assert not mismatches, f"Found {len(mismatches)} mismatch(es) after {turns_completed} turns: " + "; ".join(
        f"[{m.type.value}] seg[{m.segment_index}] expected={m.expected_text!r} actual={m.actual_text!r} ({m.detail})"
        for m in mismatches
    )

    return {
        "turns_completed": turns_completed,
        "total_tool_calls": total_tool_calls,
        "total_mismatches": len(mismatches),
    }
