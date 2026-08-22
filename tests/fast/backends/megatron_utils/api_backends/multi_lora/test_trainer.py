from pathlib import Path
from types import SimpleNamespace

import pytest

import miles.backends.megatron_utils.api_backends.multi_lora.executor as executor_module
import miles.backends.megatron_utils.api_backends.multi_lora.trainer as trainer
from miles.ray.multi_lora.config import AdapterRun, AdapterRunConfig


def make_run(name="X", slot=0, step=3, save="/tmp/tinker-trainer-test"):
    config = AdapterRunConfig(rank=8, alpha=16, save=Path(save) / name if save else None)
    return AdapterRun(name=name, config=config, slot=slot, step=step, registration_id="reg1")


def control_op(kind, name="X", slot=0, op_id="op1", payload=None, step=3, serving_version=1):
    return dict(
        operation_id=op_id,
        name=name,
        kind=kind,
        payload=payload,
        step=step,
        serving_version=serving_version,
        _lease_slot=slot,  # Harness-only; the lease remains the binding source.
    )


@pytest.fixture()
def harness(monkeypatch):
    calls = SimpleNamespace(step_args=None, saved=[], loaded=[], backups=0)

    def fake_step(optimizer, model, adam_params_by_slot):
        calls.step_args = adam_params_by_slot
        vetoed = {slot for slot, adam in adam_params_by_slot.items() if (adam or {}).get("veto")}
        return {slot: 1.25 for slot in adam_params_by_slot if slot not in vetoed}, vetoed, set()

    monkeypatch.setattr(executor_module, "step_adapter_slots", fake_step)
    monkeypatch.setattr(trainer, "save_slot_state", lambda *a, **k: calls.saved.append(k) or Path("/saved"))
    monkeypatch.setattr(trainer, "load_slot_state", lambda *a, base=None, **k: 42 if "good" in str(base) else None)

    loaded = {"X": make_run(), "Y": make_run("Y", slot=1)}
    pending: set = set()
    backuper = SimpleNamespace(backup=lambda tag: setattr(calls, "backups", calls.backups + 1))

    def run(operations):
        lease = {
            "dispatch_id": "lease-t",
            "bindings_by_operation": [
                [op["operation_id"], [op["name"], "reg1", op.pop("_lease_slot", 0)]] for op in operations
            ],
        }
        return trainer.execute_controls(SimpleNamespace(), None, None, loaded, pending, backuper, operations, lease)

    return SimpleNamespace(run=run, calls=calls, loaded=loaded, pending=pending)


class TestExecuteControls:
    def test_optim_steps_apply_per_call_adam_and_report_norms(self, harness):
        results = harness.run([control_op("optim_step", payload={"adam_params": {"learning_rate": 3e-4}})])
        # The coordinator resolves the SDK defaults into the request.
        assert harness.calls.step_args[0]["learning_rate"] == 3e-4
        assert harness.calls.step_args[0]["beta1"] == 0.9
        assert results["op1"] == dict(
            ok=True, gradient_window_consumed=True, result=dict(grad_norm=1.25, learning_rate=3e-4)
        )

    def test_poisoned_optim_discards_the_window_and_never_steps(self, harness, monkeypatch):
        zeroed = []
        monkeypatch.setattr(executor_module, "zero_adapter_slot_grads", lambda model, slot: zeroed.append(slot))
        poison = "a forward_backward in this gradient window failed; the window's gradients were discarded"
        results = harness.run(
            [
                {**control_op("optim_step", op_id="bad", payload={"adam_params": {}}), "poison": poison},
                control_op(
                    "optim_step", name="Y", op_id="good", slot=1, payload={"adam_params": {"learning_rate": 2e-4}}
                ),
            ]
        )
        assert zeroed == [0]
        assert set(harness.calls.step_args) == {1}
        assert harness.calls.step_args[1]["learning_rate"] == 2e-4
        assert results["bad"] == dict(ok=False, error=poison, category="user", gradient_window_consumed=True)
        assert results["good"]["ok"] is True

    def test_vetoed_slot_fails_as_server_error(self, harness):
        results = harness.run([control_op("optim_step", payload={"adam_params": {"veto": True}})])
        assert results["op1"]["ok"] is False and results["op1"]["category"] == "server"
        assert "vetoed" in results["op1"]["error"]

    def test_publish_stages_the_push_and_defers(self, harness):
        results = harness.run([control_op("save_weights_for_sampler")])
        assert results["op1"] == dict(ok=True, deferred="publish")
        assert harness.pending == {"X"}

    def test_non_resident_adapter_is_a_server_error(self, harness):
        results = harness.run([control_op("save_state", name="ghost", slot=2)])
        assert results["op1"]["ok"] is False and "not resident" in results["op1"]["error"]

    def test_lease_binding_must_match_the_loaded_registration_and_slot(self, harness):
        wrong_slot = harness.run([control_op("optim_step", slot=1)])
        assert wrong_slot["op1"]["ok"] is False and "not resident" in wrong_slot["op1"]["error"]
        assert harness.calls.step_args is None

    def test_state_operation_validates_the_binding_name_before_mutation(self):
        """The binding name is part of tenant identity and is checked before mutation."""
        from miles.ray.multi_lora.residency import ResidentBinding
        from miles.utils.operation_contract import BatchExecutionLease

        lease = BatchExecutionLease(
            dispatch_id="lease-t",
            bindings_by_operation=(("op1", ResidentBinding(("B", "reg1"), 0)),),
        )
        pending: set = set()
        result = trainer._execute_state_op(
            dict(operation_id="op1", name="A", kind="save_weights_for_sampler"),
            lease,
            None,
            None,
            None,
            {"A": make_run("A")},
            pending,
        )
        assert result["ok"] is False and result["category"] == "server"
        assert pending == set()

    def test_operation_missing_from_the_lease_is_refused(self, harness):
        op = control_op("optim_step")
        op.pop("_lease_slot")
        lease = {"dispatch_id": "lease-t", "bindings_by_operation": []}
        results = trainer.execute_controls(
            SimpleNamespace(),
            None,
            None,
            harness.loaded,
            harness.pending,
            SimpleNamespace(backup=lambda t: None),
            [op],
            lease,
        )
        assert results["op1"]["ok"] is False and "no binding in the batch lease" in results["op1"]["error"]

    def test_save_state_validates_tag_and_immutability(self, harness, tmp_path, monkeypatch):
        results = harness.run([control_op("save_state", payload={"tag": "../evil"})])
        assert "invalid state tag" in results["op1"]["error"] and results["op1"]["category"] == "user"

        harness.loaded["X"] = make_run(save=None)
        results = harness.run([control_op("save_state", payload={"tag": "t0"})])
        assert "no save dir" in results["op1"]["error"]

        harness.loaded["X"] = make_run(save=tmp_path)
        existing = tmp_path / "X" / "states" / "t0"
        existing.mkdir(parents=True)
        (existing / "manifest.pt").touch()
        results = harness.run([control_op("save_state", payload={"tag": "t0"})])
        assert "immutable" in results["op1"]["error"]

        results = harness.run([control_op("save_state", payload={"tag": "t1"})])
        # The registry clock rides the op, not the stale loaded view.
        assert results["op1"] == dict(ok=True, result=dict(path=str(tmp_path / "X" / "states" / "t1"), step=3))
        assert harness.calls.saved[0]["reason"] == "state:t1"

    def test_load_state_restores_step_and_stages_republish(self, harness):
        results = harness.run([control_op("load_state", payload={"path": "/good/state"})])
        # Deferred: the operation completes only after the re-publish lands, so
        # a client that saw SUCCEEDED can never sample pre-restore weights.
        assert results["op1"] == dict(ok=True, deferred="publish", result=dict(step=42, path="/good/state"))
        assert harness.pending == {"X"}
        assert harness.calls.backups == 1

        results = harness.run([control_op("load_state", op_id="op2", payload={"path": "/missing"})])
        assert results["op2"]["ok"] is False and results["op2"]["category"] == "user"

    def test_unknown_kind_fails_every_leftover(self, harness):
        results = harness.run([control_op("compile_model")])
        assert results["op1"]["ok"] is False and "no executor" in results["op1"]["error"]


class TestLoadAdapters:
    def test_master_reload_skips_restored_slots(self, monkeypatch):
        import sys
        from types import ModuleType

        restored = {"fresh": None, "resumed": 9, "resumed-at-zero": 0}
        inits: list = []
        reloaded: list = []
        bridge = ModuleType("megatron.bridge.peft.multi_lora_layers")
        bridge.init_adapter_slot = lambda model, slot, rank, alpha: inits.append(slot)
        monkeypatch.setitem(sys.modules, "megatron.bridge.peft.multi_lora_layers", bridge)
        monkeypatch.setattr(trainer, "load_slot_state", lambda args, model, optimizer, adapter: restored[adapter.name])
        monkeypatch.setattr(trainer, "reload_adapter_slot_model_params", lambda optimizer, slot: reloaded.append(slot))
        # Patch the CANONICAL module instance (fresh import -> sys.modules),
        # not the string path: pytest's string resolution walks package
        # ATTRIBUTES from the top, and a sys.modules-restoring fixture
        # elsewhere (test_model_initialize) leaves a stale submodule attribute
        # on the parent package — the string form then patches the evicted
        # instance while load_adapters' function-level import gets the fresh
        # one (real function -> "ParallelState not initialized").
        import miles.backends.megatron_utils.initialize as megatron_initialize

        monkeypatch.setattr(megatron_initialize, "is_first_replica_megatron_main_rank", lambda: False)

        adapters = [make_run("fresh", slot=0), make_run("resumed", slot=1), make_run("resumed-at-zero", slot=2)]
        assert trainer.load_adapters(SimpleNamespace(), None, None, adapters) == 3
        assert inits == [0]
        # A restored slot's fp32 masters came from the checkpoint; rebuilding
        # them from the bf16 model weights would drop the saved precision.
        assert reloaded == [0]


class TestGatherAndCommit:
    def test_gather_groups_rows_per_operation_in_order(self):
        rollout_data = {
            # (0, -1) is a zero-weight DP pad: filtered from the result plane.
            "tinker_logprob_collector": {(0, 1): [-2.0], (0, 0): [-1.0], (1, 0): [-9.0], (0, -1): [-7.0]},
            "operation_by_lane": {0: "fb1", 1: "fb2", 2: None},
        }
        assert trainer._gather_logprobs(rollout_data) == {"fb1": [[-1.0], [-2.0]], "fb2": [[-9.0]]}

    def test_commit_pins_accumulators_and_completes_ops(self, monkeypatch):
        committed = {}

        class FakeController:
            class commit_tinker_batch:  # noqa: N801 - mimics the .remote handle
                @staticmethod
                def remote(accumulated, operation_ids, logprobs_by_op):
                    committed.update(
                        accumulated=accumulated, operation_ids=operation_ids, logprobs_by_op=logprobs_by_op
                    )

        monkeypatch.setattr(trainer, "get_multi_lora_controller", lambda: FakeController)
        monkeypatch.setattr(trainer.ray, "get", lambda ref: ref)
        # Canonical-instance patch; see test_master_reload_skips_restored_slots.
        import miles.backends.megatron_utils.initialize as megatron_initialize

        monkeypatch.setattr(megatron_initialize, "is_first_replica_megatron_main_rank", lambda: True)

        rollout_data = {
            "registration_by_lane": {0: ("A", "r-A"), 1: ("B", "r-B")},
            "operation_by_lane": {0: "fb1", 1: None},
            "tinker_logprob_collector": {(0, 0): [-1.0]},
        }
        trainer.commit_batch(rollout_data, pending_push=set())
        # Exact registration keys, never a bare name list.
        assert committed["accumulated"] == [("A", "r-A"), ("B", "r-B")]
        assert committed["operation_ids"] == ["fb1"]
        assert committed["logprobs_by_op"] == {"fb1": [[-1.0]]}

        committed.clear()
        trainer.commit_batch({**rollout_data, "tinker_forward_only": True}, pending_push=set())
        assert committed["accumulated"] == []  # forward batches pin nothing


class TestPushPlumbing:
    def test_select_pushes_only_staged_unless_new_engines(self):
        loaded = {"A": make_run("A"), "B": make_run("B", slot=1)}
        pushes, bumps = trainer.select_adapters_to_push(loaded, {"B", "gone"}, has_new_engines=False)
        assert list(pushes) == ["B"] and bumps == ["B"]

        pushes, bumps = trainer.select_adapters_to_push(loaded, {"B"}, has_new_engines=True)
        assert list(pushes) == ["A", "B"]
        assert bumps == ["B"]  # re-pushes to fresh engines bump nothing

    def test_commit_weight_push_only_on_main_rank(self, monkeypatch):
        recorded = []

        class FakeController:
            class record_weight_update:  # noqa: N801
                @staticmethod
                def remote(names):
                    recorded.append(names)

        monkeypatch.setattr(trainer, "get_multi_lora_controller", lambda: FakeController)
        monkeypatch.setattr(trainer.ray, "get", lambda ref: ref)
        trainer.commit_weight_push(["A"], is_main_rank=False)
        trainer.commit_weight_push([], is_main_rank=True)
        assert recorded == []
        trainer.commit_weight_push(["A"], is_main_rank=True)
        assert recorded == [["A"]]
