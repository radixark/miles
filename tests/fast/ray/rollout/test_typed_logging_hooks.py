from typing import Any

import pytest
from tests.fast.fixtures.hook_consumer_fixtures import _hook

from miles.ray.rollout.metrics import log_eval_rollout_data, log_rollout_data
from miles.utils.args.custom_view import ImmutableNamespace
from miles.utils.function_registry import function_registry


class TestTypedLoggingHooks:
    @pytest.mark.parametrize("evaluation", [False, True])
    def test_logging_override_receives_custom_fields_and_can_short_circuit_default_metrics(
        self, evaluation: bool
    ) -> None:
        """Both logging hooks receive flat plugin values before default metric computation starts."""
        hook = _hook()
        args = ImmutableNamespace(custom_eval_rollout_log_function_path=hook, custom_rollout_log_function_path=hook)
        data = object()
        metrics = {"custom": 7}
        seen = []

        def callback(rollout_id: int, view: ImmutableNamespace, values: Any, extras: Any, *rest: Any) -> bool:
            assert rollout_id == 3
            assert view.hook_marker == 41
            assert values is data
            assert extras is metrics
            seen.append(rest)
            return True

        with function_registry.temporary(hook.path, callback):
            if evaluation:
                log_eval_rollout_data(3, args, data, extra_metrics=metrics)
            else:
                log_rollout_data(3, args, data, rollout_extra_metrics=metrics, rollout_time=2.5)
        assert seen == [() if evaluation else (2.5,)]
        assert "hook_marker" not in dict(args)
