import pytest
from tests.fast.utils.test_arguments import _parse_megatron_args

from miles.utils.env_report.launcher_report import LAUNCHER_REPORT_ENV_VAR


class TestCategoryEnvironmentDefaults:
    @pytest.mark.parametrize(
        "field,variable,flag,first,second,explicit",
        [
            (
                "env_report",
                LAUNCHER_REPORT_ENV_VAR,
                "--env-report",
                "/first-report.json",
                "/second-report.json",
                "/explicit-report.json",
            ),
            ("prometheus_port", "PROMETHEUS_PORT", "--prometheus-port", 19101, 19102, 19103),
        ],
    )
    def test_new_parsers_read_current_environment_and_preserve_explicit_cli_values(
        self,
        monkeypatch: pytest.MonkeyPatch,
        field: str,
        variable: str,
        flag: str,
        first: str | int,
        second: str | int,
        explicit: str | int,
    ) -> None:
        """Environment defaults are resolved for each parser without overriding explicit CLI values."""
        monkeypatch.setenv(variable, str(first))
        initial = _parse_megatron_args([])
        monkeypatch.setenv(variable, str(second))

        refreshed = _parse_megatron_args([])
        overridden = _parse_megatron_args([flag, str(explicit)])

        assert vars(initial)[field] == first
        assert vars(refreshed)[field] == second
        assert vars(overridden)[field] == explicit
        assert type(vars(refreshed)[field]) is type(second)
        assert type(vars(overridden)[field]) is type(explicit)
