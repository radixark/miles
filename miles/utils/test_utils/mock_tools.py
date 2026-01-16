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


def multi_turn_tool_call_process_fn(prompt: str) -> ProcessResult:
    first_prompt = "What is 42 + year + temperature?"
    first_response = (
        "Let me get the year and temperature first.\n"
        "<tool_call>\n"
        '{"name": "get_year", "arguments": {}}\n'
        "</tool_call>\n"
        "<tool_call>\n"
        '{"name": "get_temperature", "arguments": {"location": "Mars"}}\n'
        "</tool_call>"
    )

    second_prompt = '{"year": 2026}'
    second_response = "The answer is: 42 + 2026 + -60 = 2008."

    prompt_response_pairs = {
        first_prompt: first_response,
        second_prompt: second_response,
    }

    for key, response in prompt_response_pairs.items():
        if key in prompt:
            return ProcessResult(text=response, finish_reason="stop")

    raise ValueError(f"Unexpected prompt, no matching key found. {prompt=}")
