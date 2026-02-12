import json
from copy import deepcopy
from typing import Any

from transformers import AutoTokenizer

from miles.utils.test_utils.mock_sglang_server import ProcessResult

AGENTIC_MAX_TURNS: int | None = None
from miles.utils.http_utils import post

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_year",
            "description": "Get current year",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_temperature",
            "description": "Get temperature for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    },
]


def _get_year(params: dict) -> str:
    assert len(params) == 0
    return json.dumps({"year": 2026})


def _get_temperature(params: dict) -> str:
    temps = {"Mars": -60, "Earth": 15}
    location = params.get("location")
    assert location in temps, f"Unknown location: {location}"
    return json.dumps({"temperature": temps[location]})


TOOL_EXECUTORS = {
    "get_year": _get_year,
    "get_temperature": _get_temperature,
}


async def execute_tool_call(name: str, params: dict) -> str:
    return TOOL_EXECUTORS[name](params)


async def run_agentic_tool_call(
    base_url: str,
    prompt: list[dict[str, Any]] | str,
    request_kwargs: dict[str, Any] | None = None,
    max_turns: int = 8,
) -> None:
    if AGENTIC_MAX_TURNS is not None:
        max_turns = AGENTIC_MAX_TURNS
    messages = deepcopy(prompt) if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    request_kwargs = request_kwargs or {}
    model = request_kwargs.get("model", "default")
    tools = request_kwargs.get("tools", SAMPLE_TOOLS)

    for _ in range(max_turns):
        payload = {"model": model, "messages": messages, "tools": tools}
        response = await post(base_url + "/v1/chat/completions", payload)
        choice = response["choices"][0]["message"]
        tool_calls = choice.get("tool_calls") or []
        if not tool_calls:
            break

        assistant_msg = {
            "content": choice.get("content"),
            "refusal": choice.get("refusal"),
            "role": choice.get("role", "assistant"),
            "annotations": choice.get("annotations"),
            "audio": choice.get("audio"),
            "function_call": choice.get("function_call"),
            "tool_calls": tool_calls,
        }
        messages.append(assistant_msg)

        for tool_call in tool_calls:
            name = tool_call["function"]["name"]
            raw_args = tool_call["function"].get("arguments") or "{}"
            try:
                params = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                params = {}
            result = await execute_tool_call(name, params)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "content": result,
                    "name": name,
                }
            )


_SYSTEM_PROMPT = (
    "<|im_start|>system\n"
    "# Tools\n"
    "\n"
    "You may call one or more functions to assist with the user query.\n"
    "\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    '{"type": "function", "function": {"name": "get_year", "description": "Get current year", "parameters": {"type": "object", "properties": {}, "required": []}}}\n'
    '{"type": "function", "function": {"name": "get_temperature", "description": "Get temperature for a location", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}\n'
    "</tools>\n"
    "\n"
    "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call><|im_end|>\n"
)


_TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)


class TwoTurnStub:
    """Stub for 2-turn: get_year + get_temperature(Mars) -> final answer"""

    USER_QUESTION = "What is 42 + year + temperature?"

    FIRST_RESPONSE = (
        "Let me get the year and temperature first.\n"
        "<tool_call>\n"
        '{"name": "get_year", "arguments": {}}\n'
        "</tool_call>\n"
        "<tool_call>\n"
        '{"name": "get_temperature", "arguments": {"location": "Mars"}}\n'
        "</tool_call><|im_end|>\n"
    )

    FIRST_TOOL_RESPONSE = (
        "<|im_start|>user\n"
        "<tool_response>\n"
        '{"year": 2026}\n'
        "</tool_response>\n"
        "<tool_response>\n"
        '{"temperature": -60}\n'
        "</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    SECOND_RESPONSE = "The answer is: 42 + 2026 + -60 = 2008."

    FIRST_PROMPT = _SYSTEM_PROMPT + "<|im_start|>user\n" + USER_QUESTION + "<|im_end|>\n" + "<|im_start|>assistant\n"
    SECOND_PROMPT = FIRST_PROMPT + FIRST_RESPONSE + FIRST_TOOL_RESPONSE

    PROMPT = [{"role": "user", "content": USER_QUESTION}]

    FIRST_PROMPT_TOKEN_IDS = _TOKENIZER(FIRST_PROMPT, add_special_tokens=False)["input_ids"]
    SECOND_PROMPT_TOKEN_IDS = _TOKENIZER(SECOND_PROMPT, add_special_tokens=False)["input_ids"]

    FIRST_RESPONSE_CONTENT = "Let me get the year and temperature first."
    FIRST_TOOL_CALLS_OPENAI_FORMAT = [
        {"id": "call00000", "function": {"arguments": "{}", "name": "get_year"}, "type": "function"},
        {
            "id": "call00001",
            "function": {"arguments": '{"location": "Mars"}', "name": "get_temperature"},
            "type": "function",
        },
    ]

    OPENAI_MESSAGES_FIRST_TURN = [{"role": "user", "content": USER_QUESTION}]

    OPENAI_MESSAGES_SECOND_TURN_FROM_CLIENT = OPENAI_MESSAGES_FIRST_TURN + [
        {
            "content": FIRST_RESPONSE_CONTENT,
            "refusal": None,
            "role": "assistant",
            "annotations": None,
            "audio": None,
            "function_call": None,
            "tool_calls": FIRST_TOOL_CALLS_OPENAI_FORMAT,
        },
        {"role": "tool", "tool_call_id": "call00000", "content": '{"year": 2026}', "name": "get_year"},
        {"role": "tool", "tool_call_id": "call00001", "content": '{"temperature": -60}', "name": "get_temperature"},
    ]

    @staticmethod
    def process_fn(prompt: str) -> ProcessResult:
        prompt_response_pairs = {
            TwoTurnStub.FIRST_PROMPT: TwoTurnStub.FIRST_RESPONSE,
            TwoTurnStub.SECOND_PROMPT: TwoTurnStub.SECOND_RESPONSE,
        }

        for expect_prompt, response in prompt_response_pairs.items():
            if prompt == expect_prompt:
                return ProcessResult(text=response, finish_reason="stop")

        raise ValueError(f"Unexpected {prompt=}")


class ThreeTurnStub:
    """Stub for 3-turn: get_year + get_temperature(Mars) -> get_temperature(Earth) -> final answer"""

    USER_QUESTION = "What is 42 + year + Mars temperature + Earth temperature?"

    FIRST_RESPONSE = (
        "Let me get the year and Mars temperature first.\n"
        "<tool_call>\n"
        '{"name": "get_year", "arguments": {}}\n'
        "</tool_call>\n"
        "<tool_call>\n"
        '{"name": "get_temperature", "arguments": {"location": "Mars"}}\n'
        "</tool_call><|im_end|>\n"
    )

    SECOND_RESPONSE = (
        "Now let me get Earth temperature.\n"
        "<tool_call>\n"
        '{"name": "get_temperature", "arguments": {"location": "Earth"}}\n'
        "</tool_call><|im_end|>\n"
    )

    FIRST_TOOL_RESPONSE = (
        "<|im_start|>user\n"
        "<tool_response>\n"
        '{"year": 2026}\n'
        "</tool_response>\n"
        "<tool_response>\n"
        '{"temperature": -60}\n'
        "</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    SECOND_TOOL_RESPONSE = (
        "<|im_start|>user\n"
        "<tool_response>\n"
        '{"temperature": 15}\n'
        "</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    THIRD_RESPONSE = "The answer is: 42 + 2026 + -60 + 15 = 2023."

    FIRST_PROMPT = _SYSTEM_PROMPT + "<|im_start|>user\n" + USER_QUESTION + "<|im_end|>\n" + "<|im_start|>assistant\n"
    SECOND_PROMPT = FIRST_PROMPT + FIRST_RESPONSE + FIRST_TOOL_RESPONSE
    THIRD_PROMPT = SECOND_PROMPT + SECOND_RESPONSE + SECOND_TOOL_RESPONSE

    PROMPT = [{"role": "user", "content": USER_QUESTION}]

    FIRST_PROMPT_TOKEN_IDS = _TOKENIZER(FIRST_PROMPT, add_special_tokens=False)["input_ids"]
    SECOND_PROMPT_TOKEN_IDS = _TOKENIZER(SECOND_PROMPT, add_special_tokens=False)["input_ids"]
    THIRD_PROMPT_TOKEN_IDS = _TOKENIZER(THIRD_PROMPT, add_special_tokens=False)["input_ids"]

    FIRST_RESPONSE_CONTENT = "Let me get the year and Mars temperature first."
    FIRST_TOOL_CALLS_OPENAI_FORMAT = [
        {"id": "call00000", "function": {"arguments": "{}", "name": "get_year"}, "type": "function"},
        {
            "id": "call00001",
            "function": {"arguments": '{"location": "Mars"}', "name": "get_temperature"},
            "type": "function",
        },
    ]

    SECOND_RESPONSE_CONTENT = "Now let me get Earth temperature."
    SECOND_TOOL_CALLS_OPENAI_FORMAT = [
        {
            "id": "call00000",
            "function": {"arguments": '{"location": "Earth"}', "name": "get_temperature"},
            "type": "function",
        },
    ]

    OPENAI_MESSAGES_FIRST_TURN = [{"role": "user", "content": USER_QUESTION}]

    OPENAI_MESSAGES_SECOND_TURN_FROM_CLIENT = OPENAI_MESSAGES_FIRST_TURN + [
        {
            "content": FIRST_RESPONSE_CONTENT,
            "refusal": None,
            "role": "assistant",
            "annotations": None,
            "audio": None,
            "function_call": None,
            "tool_calls": FIRST_TOOL_CALLS_OPENAI_FORMAT,
        },
        {"role": "tool", "tool_call_id": "call00000", "content": '{"year": 2026}', "name": "get_year"},
        {"role": "tool", "tool_call_id": "call00001", "content": '{"temperature": -60}', "name": "get_temperature"},
    ]

    OPENAI_MESSAGES_THIRD_TURN_FROM_CLIENT = OPENAI_MESSAGES_SECOND_TURN_FROM_CLIENT + [
        {
            "content": SECOND_RESPONSE_CONTENT,
            "refusal": None,
            "role": "assistant",
            "annotations": None,
            "audio": None,
            "function_call": None,
            "tool_calls": SECOND_TOOL_CALLS_OPENAI_FORMAT,
        },
        {"role": "tool", "tool_call_id": "call00000", "content": '{"temperature": 15}', "name": "get_temperature"},
    ]

    @staticmethod
    def process_fn(prompt: str) -> ProcessResult:
        prompt_response_pairs = {
            ThreeTurnStub.FIRST_PROMPT: ThreeTurnStub.FIRST_RESPONSE,
            ThreeTurnStub.SECOND_PROMPT: ThreeTurnStub.SECOND_RESPONSE,
            ThreeTurnStub.THIRD_PROMPT: ThreeTurnStub.THIRD_RESPONSE,
        }

        for expect_prompt, response in prompt_response_pairs.items():
            if prompt == expect_prompt:
                return ProcessResult(text=response, finish_reason="stop")

        raise ValueError(f"Unexpected {prompt=}")
