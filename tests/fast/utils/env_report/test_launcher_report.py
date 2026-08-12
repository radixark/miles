import json

from miles.utils.env_report.launcher_report import decode_env_report


class TestDecodeEnvReport:
    def test_decodes_base64_json(self) -> None:
        import base64

        data = {"flavor": "test"}
        encoded = base64.b64encode(json.dumps(data).encode()).decode()
        assert decode_env_report(encoded) == data

    def test_decodes_raw_json(self) -> None:
        assert decode_env_report('{"x": 1}') == {"x": 1}

    def test_returns_none_for_empty(self) -> None:
        assert decode_env_report("") is None

    def test_returns_none_for_invalid(self) -> None:
        assert decode_env_report("not json at all!!!") is None
