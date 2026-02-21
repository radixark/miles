from unittest.mock import MagicMock, patch

from miles.utils import dumper_utils


class TestWrapForwardStepWithStepping:
    """Test that the wrapper calls dumper.step() between microbatches but not before the first one."""

    def test_first_call_does_not_step(self) -> None:
        inner = MagicMock(return_value=("output", "loss_fn"))
        wrapped = dumper_utils._wrap_forward_step_with_stepping(inner)

        mock_dumper = MagicMock()
        with patch("miles.utils.dumper_utils.dumper", mock_dumper):
            wrapped("iter", "model")

        mock_dumper.step.assert_not_called()

    def test_second_call_steps_once(self) -> None:
        inner = MagicMock(return_value=("output", "loss_fn"))
        wrapped = dumper_utils._wrap_forward_step_with_stepping(inner)

        mock_dumper = MagicMock()
        with patch("miles.utils.dumper_utils.dumper", mock_dumper):
            wrapped("iter", "model")
            wrapped("iter", "model")

        mock_dumper.step.assert_called_once()

    def test_n_calls_step_n_minus_1_times(self) -> None:
        inner = MagicMock(return_value=("output", "loss_fn"))
        wrapped = dumper_utils._wrap_forward_step_with_stepping(inner)

        mock_dumper = MagicMock()
        with patch("miles.utils.dumper_utils.dumper", mock_dumper):
            for _ in range(5):
                wrapped("iter", "model")

        assert mock_dumper.step.call_count == 4

    def test_passes_args_and_kwargs_through(self) -> None:
        inner = MagicMock(return_value=("output", "loss_fn"))
        wrapped = dumper_utils._wrap_forward_step_with_stepping(inner)

        mock_dumper = MagicMock()
        with patch("miles.utils.dumper_utils.dumper", mock_dumper):
            result = wrapped("my_iter", "my_model", return_schedule_plan=True)

        inner.assert_called_once_with("my_iter", "my_model", return_schedule_plan=True)
        assert result == ("output", "loss_fn")

    def test_returns_inner_result(self) -> None:
        sentinel = object()
        inner = MagicMock(return_value=sentinel)
        wrapped = dumper_utils._wrap_forward_step_with_stepping(inner)

        mock_dumper = MagicMock()
        with patch("miles.utils.dumper_utils.dumper", mock_dumper):
            result = wrapped("iter", "model")

        assert result is sentinel
