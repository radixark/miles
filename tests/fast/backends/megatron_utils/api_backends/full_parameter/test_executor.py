from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch
from miles.backends.megatron_utils.api_backends.full_parameter.executor import (
    FullParameterBinding,
    FullParameterExecutor,
)
from miles.backends.training_utils.operation_execution import StepRequest, run_optim_controls
from miles.utils.operation_contract import BatchExecutionLease


class FakeModelChunk:
    def __init__(self, *, zero_error: Exception | None = None, gradient: torch.Tensor | None = None):
        self.zero_calls = 0
        self.zero_error = zero_error
        self.parameter = SimpleNamespace(main_grad=gradient, grad=None, decoupled_grad=None)

    def zero_grad_buffer(self):
        self.zero_calls += 1
        if self.zero_error is not None:
            raise self.zero_error

    def parameters(self):
        return [self.parameter]


class FakeOptimizer:
    def __init__(
        self,
        *,
        step_result=(True, 3.5, 0),
        step_error: Exception | None = None,
        optimizer_name: str = "adam",
    ):
        self.param_groups = [dict(lr=9.0, params=[]), dict(lr=8.0, params=[])]
        self.config = SimpleNamespace(clip_grad=17.0, optimizer=optimizer_name)
        self.step_result = step_result
        self.step_error = step_error
        self.step_calls = 0
        self.zero_calls = 0
        self.seen_groups = None
        self.seen_clip = None

    def step(self):
        self.step_calls += 1
        self.seen_groups = [dict(group) for group in self.param_groups]
        self.seen_clip = self.config.clip_grad
        if self.step_error is not None:
            raise self.step_error
        return self.step_result(self) if callable(self.step_result) else self.step_result

    def zero_grad(self):
        self.zero_calls += 1


TARGET = FullParameterBinding(target_id="actor")


def make_lease(operation_id="op", binding=TARGET, *, extras=()):
    return BatchExecutionLease(
        dispatch_id="dispatch",
        bindings_by_operation=((operation_id, binding), *extras),
    )


def make_request(operation_id="op", **overrides):
    adam = dict(
        learning_rate=0.25,
        beta1=0.7,
        beta2=0.8,
        eps=1e-7,
        weight_decay=0.03,
        grad_clip_norm=2.5,
    )
    adam.update(overrides)
    return StepRequest(operation_id=operation_id, adam_params=adam)


def make_executor(*, gradient: torch.Tensor | None = None, **optimizer_kwargs):
    model = [FakeModelChunk(gradient=gradient), FakeModelChunk()]
    optimizer = FakeOptimizer(**optimizer_kwargs)
    return FullParameterExecutor(model_chunks=model, optimizer=optimizer, binding=TARGET), model, optimizer


def test_binding_and_executor_configuration_are_immutable():
    binding = FullParameterBinding(target_id="actor")
    assert binding == TARGET
    with pytest.raises(FrozenInstanceError):
        binding.target_id = "slot-0"


def test_discard_clears_model_buffers_and_optimizer_gradients():
    executor, model, optimizer = make_executor()

    assert executor.discard_many(make_lease(), ["op"]) == {"op": {"ok": True, "gradient_window_consumed": True}}
    assert [chunk.zero_calls for chunk in model] == [1, 1]
    assert optimizer.zero_calls == 1
    assert optimizer.step_calls == 0


@pytest.mark.parametrize(
    ("lease", "operation_ids"),
    [
        (make_lease("leased"), ["requested"]),
        (make_lease(binding=FullParameterBinding(target_id="other")), ["op"]),
        (make_lease(extras=(("other", TARGET),)), ["op"]),
        (make_lease(), ["op", "other"]),
        (make_lease(), ["op", "op"]),
    ],
)
def test_invalid_or_non_singleton_discard_is_refused_before_mutation(lease, operation_ids):
    executor, model, optimizer = make_executor()

    outcomes = executor.discard_many(lease, operation_ids)

    assert set(outcomes) == set(operation_ids)
    assert all(outcome["ok"] is False for outcome in outcomes.values())
    assert all("gradient_window_consumed" not in outcome for outcome in outcomes.values())
    assert [chunk.zero_calls for chunk in model] == [0, 0]
    assert optimizer.zero_calls == 0


def test_step_applies_per_call_adam_uses_temporary_clip_and_clears_window():
    executor, model, optimizer = make_executor()

    outcome = executor.step_many(make_lease(), [make_request()])["op"]

    assert outcome == {
        "ok": True,
        "gradient_window_consumed": True,
        "result": {"grad_norm": 3.5, "learning_rate": 0.25},
    }
    assert optimizer.step_calls == 1
    assert optimizer.seen_clip == 2.5
    assert optimizer.config.clip_grad == 17.0
    for group in optimizer.seen_groups:
        assert group["lr"] == 0.25
        assert group["betas"] == (0.7, 0.8)
        assert group["eps"] == 1e-7
        assert group["weight_decay"] == 0.03
    assert [chunk.zero_calls for chunk in model] == [1, 1]
    assert optimizer.zero_calls == 1


def test_zero_clip_uses_infinite_stock_clip_to_measure_norm_without_scaling():
    def direct_optimizer_result(optimizer):
        return (True, 4.25, 0) if optimizer.config.clip_grad == float("inf") else (True, None, 0)

    executor, _, optimizer = make_executor(step_result=direct_optimizer_result)

    outcome = executor.step_many(make_lease(), [make_request(grad_clip_norm=0.0)])["op"]

    assert outcome["ok"] is True
    assert outcome["result"]["grad_norm"] == 4.25
    assert optimizer.seen_clip == float("inf")
    assert optimizer.config.clip_grad == 17.0


def test_success_without_stock_grad_norm_is_fail_stop():
    executor, model, optimizer = make_executor(step_result=(True, None, 0))

    with pytest.raises(RuntimeError, match="did not report a gradient norm"):
        executor.step_many(make_lease(), [make_request()])

    assert optimizer.config.clip_grad == 17.0
    assert optimizer.zero_calls == 1
    assert [chunk.zero_calls for chunk in model] == [1, 1]


def test_generic_coordinator_refuses_poisoned_and_clean_shared_whole_lease_without_mutation():
    executor, model, optimizer = make_executor()
    operations = [
        dict(kind="optim_step", operation_id="poisoned", poison="bad gradient window"),
        dict(kind="optim_step", operation_id="clean", payload=dict(adam_params=dict(learning_rate=0.2))),
    ]
    lease = BatchExecutionLease(
        dispatch_id="mixed",
        bindings_by_operation=(("poisoned", TARGET), ("clean", TARGET)),
    )

    outcomes = run_optim_controls(operations, lease, executor)

    assert set(outcomes) == {"poisoned", "clean"}
    assert all(outcome["ok"] is False for outcome in outcomes.values())
    assert all(outcome["category"] == "server" for outcome in outcomes.values())
    assert all("singleton whole-model lease" in outcome["error"] for outcome in outcomes.values())
    assert all("gradient_window_consumed" not in outcome for outcome in outcomes.values())
    assert optimizer.step_calls == 0
    assert optimizer.zero_calls == 0
    assert [chunk.zero_calls for chunk in model] == [0, 0]


def test_optimizer_veto_fails_closed_and_consumes_the_window():
    executor, model, optimizer = make_executor(step_result=(False, None, 0))

    outcome = executor.step_many(make_lease(), [make_request()])["op"]

    assert outcome["ok"] is False
    assert outcome["category"] == "server"
    assert outcome["gradient_window_consumed"] is True
    assert optimizer.config.clip_grad == 17.0
    assert [chunk.zero_calls for chunk in model] == [1, 1]
    assert optimizer.zero_calls == 1


@pytest.mark.parametrize("step_kwargs", [dict(step_error=RuntimeError("boom")), dict(step_result=(True, 1.0))])
def test_step_fault_is_fail_stop_after_restoring_clip_and_clearing_window(step_kwargs):
    executor, model, optimizer = make_executor(**step_kwargs)

    with pytest.raises(RuntimeError):
        executor.step_many(make_lease(), [make_request()])

    assert optimizer.config.clip_grad == 17.0
    assert [chunk.zero_calls for chunk in model] == [1, 1]
    assert optimizer.zero_calls == 1


def test_cleanup_failure_is_fail_stop():
    model = [FakeModelChunk(zero_error=RuntimeError("cannot clear")), FakeModelChunk()]
    optimizer = FakeOptimizer()
    executor = FullParameterExecutor(model_chunks=model, optimizer=optimizer, binding=TARGET)

    with pytest.raises(RuntimeError, match="cannot clear"):
        executor.step_many(make_lease(), [make_request()])
    assert [chunk.zero_calls for chunk in model] == [1, 1]
    assert optimizer.zero_calls == 1


def test_discard_cleanup_failure_is_fail_stop():
    model = [FakeModelChunk(zero_error=RuntimeError("cannot discard"))]
    optimizer = FakeOptimizer()
    executor = FullParameterExecutor(model_chunks=model, optimizer=optimizer, binding=TARGET)

    with pytest.raises(RuntimeError, match="cannot discard"):
        executor.discard_many(make_lease(), ["op"])
    assert optimizer.zero_calls == 1


def test_nonfinite_gradient_vetoes_before_physical_step_and_clears_window():
    executor, model, optimizer = make_executor(gradient=torch.tensor([float("nan")]))

    outcome = executor.step_many(make_lease(), [make_request()])["op"]

    assert outcome == {
        "ok": False,
        "error": "non-finite gradient norm; step vetoed and gradients cleared",
        "category": "server",
        "gradient_window_consumed": True,
    }
    assert optimizer.step_calls == 0
    assert optimizer.config.clip_grad == 17.0
    assert optimizer.zero_calls == 1
    assert [chunk.zero_calls for chunk in model] == [1, 1]


def test_non_adam_optimizer_is_refused_before_mutation():
    executor, model, optimizer = make_executor(optimizer_name="sgd")

    outcome = executor.step_many(make_lease(), [make_request()])["op"]

    assert outcome["ok"] is False
    assert "require an Adam optimizer" in outcome["error"]
    assert optimizer.step_calls == 0
    assert optimizer.zero_calls == 0
    assert [chunk.zero_calls for chunk in model] == [0, 0]


def test_empty_model_is_refused_before_mutation():
    optimizer = FakeOptimizer()
    executor = FullParameterExecutor(model_chunks=[], optimizer=optimizer, binding=TARGET)

    outcome = executor.step_many(make_lease(), [make_request()])["op"]

    assert outcome["ok"] is False
    assert "at least one model chunk" in outcome["error"]
    assert optimizer.step_calls == 0
    assert optimizer.zero_calls == 0


def test_malformed_adam_is_refused_before_mutation():
    executor, model, optimizer = make_executor()
    request = StepRequest(operation_id="op", adam_params=[("learning_rate", 0.1)])

    outcome = executor.step_many(make_lease(), [request])["op"]

    assert outcome["ok"] is False
    assert "invalid Adam parameters" in outcome["error"]
    assert optimizer.step_calls == 0
    assert optimizer.zero_calls == 0
    assert [chunk.zero_calls for chunk in model] == [0, 0]


def test_non_singleton_step_refuses_every_operation_without_mutation():
    executor, model, optimizer = make_executor()
    lease = make_lease(extras=(("other", TARGET),))

    outcomes = executor.step_many(lease, [make_request(), make_request("other")])

    assert set(outcomes) == {"op", "other"}
    assert all(outcome["ok"] is False for outcome in outcomes.values())
    assert optimizer.step_calls == 0
    assert optimizer.zero_calls == 0
    assert [chunk.zero_calls for chunk in model] == [0, 0]


def test_empty_control_batch_is_a_noop():
    executor, model, optimizer = make_executor()

    assert executor.discard_many(make_lease(), []) == {}
    assert executor.step_many(make_lease(), []) == {}
    assert optimizer.step_calls == 0
    assert optimizer.zero_calls == 0
    assert [chunk.zero_calls for chunk in model] == [0, 0]
