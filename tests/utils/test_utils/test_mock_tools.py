from miles.utils.test_utils.mock_tools import execute_tool_call

YEAR = 2026
TEMPERATURE = -60


class TestExecuteToolCall:
    def test_execute_get_year(self):
        result = execute_tool_call("get_year", {})
        assert result == {"year": YEAR}

    def test_execute_get_temperature(self):
        result = execute_tool_call("get_temperature", {"location": "Mars"})
        assert result == {"temperature": TEMPERATURE}
