import json

from miles.utils.test_utils.mock_sglang_server import ProcessResult

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
    assert params.get("location") == "Mars"
    return json.dumps({"temperature": -60})


TOOL_EXECUTORS = {
    "get_year": _get_year,
    "get_temperature": _get_temperature,
}


async def execute_tool_call(name: str, params: dict) -> str:
    return TOOL_EXECUTORS[name](params)


MULTI_TURN_FIRST_PROMPT = (
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
    "<|im_start|>user\n"
    "What is 42 + year + temperature?<|im_end|>\n"
    "<|im_start|>assistant\n"
)
MULTI_TURN_FIRST_RESPONSE = (
    "Let me get the year and temperature first.\n"
    "<tool_call>\n"
    '{"name": "get_year", "arguments": {}}\n'
    "</tool_call>\n"
    "<tool_call>\n"
    '{"name": "get_temperature", "arguments": {"location": "Mars"}}\n'
    "</tool_call><|im_end|>\n"
)

MULTI_TURN_SECOND_PROMPT = (
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
    "<|im_start|>user\n"
    "What is 42 + year + temperature?<|im_end|>\n"
    "<|im_start|>assistant\n"
    "Let me get the year and temperature first.\n"
    "<tool_call>\n"
    '{"name": "get_year", "arguments": {}}\n'
    "</tool_call>\n"
    "<tool_call>\n"
    '{"name": "get_temperature", "arguments": {"location": "Mars"}}\n'
    "</tool_call><|im_end|>\n"
    "<|im_start|>user\n"
    "<tool_response>\n"
    '{"year": 2026}\n'
    "</tool_response>\n"
    "<tool_response>\n"
    '{"temperature": -60}\n'
    "</tool_response><|im_end|>\n"
    "<|im_start|>assistant\n"
)
MULTI_TURN_SECOND_RESPONSE = "The answer is: 42 + 2026 + -60 = 2008."

MULTI_TURN_USER_QUESTION = "What is 42 + year + temperature?"
MULTI_TURN_FIRST_RESPONSE_CONTENT = "Let me get the year and temperature first."
MULTI_TURN_FIRST_TOOL_CALLS = [
    {"id": "call00000", "type": "function", "function": {"name": "get_year", "arguments": "{}"}},
    {
        "id": "call00001",
        "type": "function",
        "function": {"name": "get_temperature", "arguments": '{"location": "Mars"}'},
    },
]
MULTI_TURN_FIRST_TOOL_CALLS_OPENAI_FORMAT = [
    {"id": "call00000", "function": {"arguments": "{}", "name": "get_year"}, "type": "function"},
    {
        "id": "call00001",
        "function": {"arguments": '{"location": "Mars"}', "name": "get_temperature"},
        "type": "function",
    },
]

MULTI_TURN_OPENAI_MESSAGES_FIRST_TURN = [
    {"role": "user", "content": MULTI_TURN_USER_QUESTION},
]

MULTI_TURN_OPENAI_MESSAGES_SECOND_TURN = [
    {"role": "user", "content": MULTI_TURN_USER_QUESTION},
    {
        "role": "assistant",
        "content": MULTI_TURN_FIRST_RESPONSE_CONTENT,
        "tool_calls": MULTI_TURN_FIRST_TOOL_CALLS,
    },
    {"role": "tool", "tool_call_id": "call00000", "content": '{"year": 2026}'},
    {"role": "tool", "tool_call_id": "call00001", "content": '{"temperature": -60}'},
]

MULTI_TURN_OPENAI_MESSAGES_SECOND_TURN_FROM_CLIENT = [
    {"role": "user", "content": MULTI_TURN_USER_QUESTION},
    {
        "content": MULTI_TURN_FIRST_RESPONSE_CONTENT,
        "refusal": None,
        "role": "assistant",
        "annotations": None,
        "audio": None,
        "function_call": None,
        "tool_calls": MULTI_TURN_FIRST_TOOL_CALLS_OPENAI_FORMAT,
    },
    {"role": "tool", "tool_call_id": "call00000", "content": '{"year": 2026}', "name": "get_year"},
    {"role": "tool", "tool_call_id": "call00001", "content": '{"temperature": -60}', "name": "get_temperature"},
]


def multi_turn_tool_call_process_fn(prompt: str) -> ProcessResult:
    prompt_response_pairs = {
        MULTI_TURN_FIRST_PROMPT: MULTI_TURN_FIRST_RESPONSE,
        MULTI_TURN_SECOND_PROMPT: MULTI_TURN_SECOND_RESPONSE,
    }

    for expect_prompt, response in prompt_response_pairs.items():
        if prompt == expect_prompt:
            return ProcessResult(text=response, finish_reason="stop")

    raise ValueError(f"Unexpected {prompt=}")
