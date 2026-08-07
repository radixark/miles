"""Trainer verbs for tinker control operations: slot-sorted execution, veto
propagation, publish staging, state-op validation, logprob gathering, and the
push-selection/commit plumbing — all with fakes (collectives are GPU E2E)."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

from pathlib import Path
from types import SimpleNamespace

import pytest

import miles.backends.megatron_utils.tinker_backend.trainer as trainer
from miles.ray.tinker_backend.config import AdapterRun, AdapterRunConfig


def make_run(name="X", slot=0, step=3, save="/tmp/tinker-trainer-test"):
    config = AdapterRunConfig(rank=8, alpha=16, save=Path(save) / name if save else None)
    return AdapterRun(name=name, config=config, slot=slot, step=step, registration_id="reg1")


def control_op(kind, name="X", slot=0, op_id="op1", payload=None, step=3, serving_version=1):
    return dict(
        operation_id=op_id,
        name=name,
        slot=slot,
        kind=kind,
        payload=payload,
        step=step,
        serving_version=serving_version,
    )


@pytest.fixture()
def harness(monkeypatch):
    """execute_controls with the collective pieces faked out."""
    calls = SimpleNamespace(step_args=None, saved=[], loaded=[], backups=0)

    def fake_step(optimizer, model, adam_params_by_slot):
        calls.step_args = adam_params_by_slot
        vetoed = {slot for slot, adam in adam_params_by_slot.items() if (adam or {}).get("veto")}
        return {slot: 1.25 for slot in adam_params_by_slot if slot not in vetoed}, vetoed

    monkeypatch.setattr(trainer, "step_adapter_slots", fake_step)
    monkeypatch.setattr(trainer, "save_slot_state", lambda *a, **k: calls.saved.append(k) or Path("/saved"))
    monkeypatch.setattr(trainer, "load_slot_state", lambda *a, base=None, **k: 42 if "good" in str(base) else None)

    loaded = {"X": make_run()}
    pending: set = set()
    backuper = SimpleNamespace(backup=lambda tag: setattr(calls, "backups", calls.backups + 1))

    def run(operations):
        return trainer.execute_controls(SimpleNamespace(), None, None, loaded, pending, backuper, operations)

    return SimpleNamespace(run=run, calls=calls, loaded=loaded, pending=pending)


class TestExecuteControls:
    def test_optim_steps_apply_per_call_adam_and_report_norms(self, harness):
        results = harness.run([control_op("optim_step", payload={"adam_params": {"learning_rate": 3e-4}})])
        assert harness.calls.step_args == {0: {"learning_rate": 3e-4}}
        assert results["op1"] == dict(ok=True, result=dict(grad_norm=1.25, learning_rate=3e-4))

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
        assert results["op1"] == dict(ok=True, result=dict(step=42, path="/good/state"))
        assert harness.pending == {"X"}  # engines must not keep pre-restore weights
        assert harness.calls.backups == 1

        results = harness.run([control_op("load_state", op_id="op2", payload={"path": "/missing"})])
        assert results["op2"]["ok"] is False and results["op2"]["category"] == "user"

    def test_unknown_kind_fails_every_leftover(self, harness):
        results = harness.run([control_op("compile_model")])
        assert results["op1"]["ok"] is False and "no executor" in results["op1"]["error"]


class TestGatherAndCommit:
    def test_gather_groups_rows_per_operation_in_order(self):
        rollout_data = {
            "tinker_logprob_collector": {(0, 1): [-2.0], (0, 0): [-1.0], (3, 0): [-9.0]},
            "operation_by_slot": {0: "fb1", 3: "fb2", 5: None},
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

        monkeypatch.setattr(trainer, "get_tinker_controller", lambda: FakeController)
        monkeypatch.setattr(trainer.ray, "get", lambda ref: ref)
        monkeypatch.setattr(
            "miles.backends.megatron_utils.initialize.is_first_replica_megatron_main_rank", lambda: True
        )

        rollout_data = {
            "adapter_name_by_slot": {0: "A", 3: "B"},
            "operation_by_slot": {0: "fb1", 3: None},
            "tinker_logprob_collector": {(0, 0): [-1.0]},
        }
        trainer.commit_batch(rollout_data, pending_push=set())
        assert committed["accumulated"] == ["A", "B"]
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

        monkeypatch.setattr(trainer, "get_tinker_controller", lambda: FakeController)
        monkeypatch.setattr(trainer.ray, "get", lambda ref: ref)
        trainer.commit_weight_push(["A"], is_main_rank=False)
        trainer.commit_weight_push([], is_main_rank=True)
        assert recorded == []
        trainer.commit_weight_push(["A"], is_main_rank=True)
        assert recorded == [["A"]]


def test_serving_name_is_registration_scoped():
    run = make_run()
    assert run.serving_name == "__miles_adapter_X_reg1"
