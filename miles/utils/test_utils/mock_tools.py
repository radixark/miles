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


def _get_year(params: dict) -> dict:
    assert len(params) == 0
    return {"year": 2026}


def _get_temperature(params: dict) -> dict:
    assert params.get("location") == "Mars"
    return {"temperature": -60}


TOOL_EXECUTORS = {
    "get_year": _get_year,
    "get_temperature": _get_temperature,
}


def execute_tool_call(name: str, params: dict) -> dict:
    return TOOL_EXECUTORS[name](params)


MULTI_TURN_FIRST_PROMPT = "What is 42 + year + temperature?"
MULTI_TURN_FIRST_RESPONSE = (
    "Let me get the year and temperature first.\n"
    "<tool_call>\n"
    '{"name": "get_year", "arguments": {}}\n'
    "</tool_call>\n"
    "<tool_call>\n"
    '{"name": "get_temperature", "arguments": {"location": "Mars"}}\n'
    "</tool_call>"
)
MULTI_TURN_SECOND_RESPONSE = "The answer is: 42 + 2026 + -60 = 2008."

MULTI_TURN_REPLIES = {
    MULTI_TURN_FIRST_PROMPT: MULTI_TURN_FIRST_RESPONSE,
    '{"year": 2026}': MULTI_TURN_SECOND_RESPONSE,
}


def multi_turn_tool_call_process_fn(prompt: str) -> ProcessResult:
    for key, response in MULTI_TURN_REPLIES.items():
        if key in prompt:
            return ProcessResult(text=response, finish_reason="stop")
    raise ValueError(f"Unexpected prompt, no matching key found. Prompt: {prompt[:500]}")
