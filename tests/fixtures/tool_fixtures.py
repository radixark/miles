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
    return {"year": 2025}


def _get_temperature(params: dict) -> dict:
    assert params.get("location") == "Mars"
    return {"temperature": 25}


TOOL_EXECUTORS = {
    "get_year": _get_year,
    "get_temperature": _get_temperature,
}


def execute_tool_call(name: str, params: dict) -> dict:
    return TOOL_EXECUTORS[name](params)
