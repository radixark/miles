import shlex

import pytest
from tests.fast.utils.test_utils.session_threshold_fixtures import SessionThresholdInputs, session_threshold_inputs

from miles.utils.test_utils.session_verify_runner import run_session_verify

__all__ = ["session_threshold_inputs"]


class TestSessionThresholdTransport:
    @pytest.mark.parametrize("threshold,passes", [(0.25, False), (0.5, True)])
    def test_the_runner_uses_the_parsed_nonzero_threshold_for_both_launch_and_actual_metrics_gate(
        self, session_threshold_inputs: SessionThresholdInputs, threshold: float, passes: bool
    ) -> None:
        """The nonzero launch threshold also controls whether a half-mismatching metrics stream passes."""
        inputs = session_threshold_inputs

        if passes:
            run_session_verify(inputs.args)
        else:
            with pytest.raises(AssertionError, match="hard TITO mismatches"):
                run_session_verify(inputs.args)

        [call] = inputs.backend.calls
        argv = shlex.split(call["train_args"])
        assert argv.count("--ci-tito-special-token-count-threshold") == 1
        assert float(argv[argv.index("--ci-tito-special-token-count-threshold") + 1]) == threshold
        assert inputs.backend.metrics_path is not None
        assert not inputs.backend.metrics_path.exists()
        assert inputs.backend.metrics_path.with_suffix(".jsonl.failed").exists() is not passes
