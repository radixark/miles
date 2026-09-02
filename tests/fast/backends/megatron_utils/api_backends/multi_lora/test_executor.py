from types import SimpleNamespace

import miles.backends.megatron_utils.api_backends.multi_lora.executor as executor_module
from miles.backends.megatron_utils.api_backends.multi_lora.executor import MultiLoraParameterExecutor
from miles.backends.training_utils.operation_execution import StepRequest
from miles.ray.multi_lora.residency import ResidentBinding
from miles.utils.operation_contract import BatchExecutionLease


def loaded(name="A", registration_id="r-A", slot=0):
    return {name: SimpleNamespace(registration_id=registration_id, slot=slot)}


def make_executor(loaded_adapters=None):
    return MultiLoraParameterExecutor(model=object(), optimizer=object(), loaded_adapters=loaded_adapters or loaded())


def lease_of(*bindings):
    return BatchExecutionLease(dispatch_id="d", bindings_by_operation=tuple(bindings))


def binding(name="A", registration_id="r-A", slot=0):
    return ResidentBinding(registration_key=(name, registration_id), training_slot=slot)


def step(op_id, lr=1e-4):
    return StepRequest(operation_id=op_id, adam_params={"learning_rate": lr})


class TestStepMany:
    def test_step_and_veto_both_report_the_window_consumed(self, monkeypatch):
        monkeypatch.setattr(
            executor_module, "step_adapter_slots", lambda optimizer, model, adam: ({0: 1.5}, {1}, set())
        )
        executor = make_executor({**loaded("A", "r-A", 0), **loaded("B", "r-B", 1)})
        lease = lease_of(("op-A", binding("A", "r-A", 0)), ("op-B", binding("B", "r-B", 1)))
        outcomes = executor.step_many(lease, [step("op-A"), step("op-B")])

        assert outcomes["op-A"]["ok"] is True
        assert outcomes["op-A"]["gradient_window_consumed"] is True
        assert outcomes["op-A"]["result"]["grad_norm"] == 1.5
        assert outcomes["op-B"]["ok"] is False
        assert outcomes["op-B"]["gradient_window_consumed"] is True

    def test_stale_binding_refusal_does_not_claim_consumption(self):
        executor = make_executor()
        lease = lease_of(("op-A", binding("A", "stale-registration", 0)))
        outcomes = executor.step_many(lease, [step("op-A")])
        assert outcomes["op-A"]["ok"] is False and outcomes["op-A"]["category"] == "server"
        assert not outcomes["op-A"].get("gradient_window_consumed")

    def test_duplicate_physical_step_targets_never_silently_drop_an_operation(self, monkeypatch):
        stepped = []
        monkeypatch.setattr(
            executor_module,
            "step_adapter_slots",
            lambda optimizer, model, adam: (stepped.append(dict(adam)) or ({s: 1.0 for s in adam}, set(), set())),
        )
        executor = make_executor()
        lease = lease_of(("op-1", binding("A", "r-A", 0)), ("op-2", binding("A", "r-A", 0)))
        outcomes = executor.step_many(lease, [step("op-1", 1e-4), step("op-2", 2e-4)])

        assert set(outcomes) == {"op-1", "op-2"}
        for op_id in ("op-1", "op-2"):
            assert outcomes[op_id]["ok"] is False and outcomes[op_id]["category"] == "server"
            assert not outcomes[op_id].get("gradient_window_consumed")
        assert stepped == []


class TestDiscardMany:
    def test_successful_discard_reports_the_window_consumed(self, monkeypatch):
        cleared = []
        monkeypatch.setattr(executor_module, "zero_adapter_slot_grads", lambda model, slot: cleared.append(slot))
        executor = make_executor()
        outcomes = executor.discard_many(lease_of(("op-A", binding())), ["op-A"])
        assert outcomes["op-A"] == dict(ok=True, gradient_window_consumed=True)
        assert cleared == [0]

    def test_refused_discard_does_not_claim_consumption(self):
        executor = make_executor()
        outcomes = executor.discard_many(lease_of(("op-A", binding(slot=5))), ["op-A"])
        assert outcomes["op-A"]["ok"] is False
        assert not outcomes["op-A"].get("gradient_window_consumed")
