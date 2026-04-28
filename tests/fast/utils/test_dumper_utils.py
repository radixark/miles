from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-fast")

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from miles.utils import dumper_utils


class TestWrapForwardStepWithStepping:

    @pytest.fixture()
    def setup(self):
        inner = MagicMock(return_value=("output", "loss_fn"))
        wrapped = dumper_utils._wrap_forward_step_with_stepping(inner)
        mock_dumper = MagicMock()
        return inner, wrapped, mock_dumper

    @pytest.mark.parametrize(("n_calls", "expected_steps"), [(1, 0), (2, 1), (5, 4)])
    def test_step_called_n_minus_1_times(self, setup, n_calls: int, expected_steps: int) -> None:
        _inner, wrapped, mock_dumper = setup
        with patch("miles.utils.dumper_utils.dumper", mock_dumper):
            for _ in range(n_calls):
                wrapped("iter", "model")
        assert mock_dumper.step.call_count == expected_steps

    def test_passes_args_and_returns_result(self, setup) -> None:
        inner, wrapped, mock_dumper = setup
        with patch("miles.utils.dumper_utils.dumper", mock_dumper):
            result = wrapped("my_iter", "my_model", extra=True)
        inner.assert_called_once_with("my_iter", "my_model", extra=True)
        assert result == ("output", "loss_fn")


def test_sglang_env_includes_startup_dumper_settings() -> None:
    args = SimpleNamespace(
        dumper_enable=False,
        dumper_inference=["enable=true", "non_intrusive_mode=all"],
        dumper_source_patcher_config_inference="/tmp/patcher.yaml",
    )

    env = dumper_utils.get_sglang_env(args)

    assert env == {
        "DUMPER_SERVER_PORT": "reuse",
        "DUMPER_NON_INTRUSIVE_MODE": "all",
        "DUMPER_SOURCE_PATCHER_CONFIG": "/tmp/patcher.yaml",
    }


def test_sglang_env_disabled_when_inference_phase_disabled() -> None:
    args = SimpleNamespace(
        dumper_enable=False,
        dumper_inference=["non_intrusive_mode=all"],
        dumper_source_patcher_config_inference=None,
    )

    assert dumper_utils.get_sglang_env(args) == {}
