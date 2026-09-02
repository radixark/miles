from miles.backends.training_utils.operation_execution import (
    ADAM_PARAM_DEFAULTS,
    StepRequest,
    resolve_adam_params,
    run_optim_controls,
)
from miles.utils.operation_contract import BatchExecutionLease


class FakeExecutor:
    def __init__(self, step_outcomes=None, discard_outcomes=None):
        self.discarded: list[str] = []
        self.stepped: list[StepRequest] = []
        self._step_outcomes = step_outcomes or {}
        self._discard_outcomes = discard_outcomes

    def discard_many(self, lease, operation_ids):
        self.discarded.extend(operation_ids)
        if self._discard_outcomes is not None:
            return self._discard_outcomes
        return {op_id: dict(ok=True) for op_id in operation_ids}

    def step_many(self, lease, requests):
        self.stepped.extend(requests)
        return {
            request.operation_id: self._step_outcomes.get(request.operation_id, dict(ok=True, result={}))
            for request in requests
        }


LEASE = BatchExecutionLease(dispatch_id="d", bindings_by_operation=(("opt1", "opaque-1"), ("opt2", "opaque-2")))


def optim(op_id, adam=None, poison=None):
    op = dict(operation_id=op_id, kind="optim_step", payload={"adam_params": adam} if adam else {})
    if poison:
        op["poison"] = poison
    return op


class TestResolveAdamParams:
    def test_defaults_fill_and_none_is_absent(self):
        resolved = resolve_adam_params({"learning_rate": 3e-4, "grad_clip_norm": None})
        assert resolved["learning_rate"] == 3e-4
        assert resolved["grad_clip_norm"] == ADAM_PARAM_DEFAULTS["grad_clip_norm"]
        assert resolve_adam_params(None) == ADAM_PARAM_DEFAULTS


class TestRunOptimControls:
    def test_poisoned_steps_discard_and_fail_as_user_errors(self):
        executor = FakeExecutor()
        results = run_optim_controls(
            [optim("opt1", poison="window poisoned"), optim("opt2", adam={"learning_rate": 2e-4})],
            LEASE,
            executor,
        )
        assert executor.discarded == ["opt1"]
        assert results["opt1"] == dict(
            ok=False, error="window poisoned", category="user", gradient_window_consumed=True
        )
        [request] = executor.stepped
        assert request.operation_id == "opt2" and request.adam_params["learning_rate"] == 2e-4
        assert results["opt2"]["ok"] is True

    def test_executor_refusal_wins_over_the_poison_policy(self):
        executor = FakeExecutor(discard_outcomes={"opt1": dict(ok=False, error="stale binding", category="server")})
        results = run_optim_controls([optim("opt1", poison="poisoned")], LEASE, executor)
        assert results["opt1"] == dict(ok=False, error="stale binding", category="server")
        assert not results["opt1"].get("gradient_window_consumed")

    def test_missing_discard_outcome_fails_closed_as_a_server_error(self):
        executor = FakeExecutor(discard_outcomes={})
        results = run_optim_controls([optim("opt1", poison="poisoned")], LEASE, executor)
        outcome = results["opt1"]
        assert outcome["ok"] is False and outcome["category"] == "server"
        assert "discard" in outcome["error"]
        assert not outcome.get("gradient_window_consumed")

    def test_missing_step_outcome_fails_closed_as_a_server_error(self):
        class SilentExecutor(FakeExecutor):
            def step_many(self, lease, requests):
                return {}

        results = run_optim_controls([optim("opt1")], LEASE, SilentExecutor())
        outcome = results["opt1"]
        assert outcome["ok"] is False and outcome["category"] == "server"
        assert not outcome.get("gradient_window_consumed")

    def test_clean_step_needs_no_prior_fb(self):
        executor = FakeExecutor()
        results = run_optim_controls([optim("opt1")], LEASE, executor)
        assert results["opt1"]["ok"] is True

    def test_non_optim_operations_are_not_the_coordinators_business(self):
        executor = FakeExecutor()
        results = run_optim_controls([dict(operation_id="save1", kind="save_state")], LEASE, executor)
        assert results == {} and executor.stepped == [] and executor.discarded == []
