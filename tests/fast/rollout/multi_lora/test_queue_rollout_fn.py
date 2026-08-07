"""Queue child rollout fn: one claimed forward_backward operation becomes one
stamped batch; bad payloads fail their own operation and the child keeps
serving; the operation directives ride the output metadata."""

from types import SimpleNamespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import asyncio

import pytest

import miles.rollout.multi_lora.queue_rollout_fn as queue_module
from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles.rollout.multi_lora.queue_rollout_fn import QueueChildRolloutFn, ThinkerOperationSource
from miles.utils.adapter_config import AdapterRun, AdapterRunConfig


def make_run(name="X", reg="rx", slot=3, version=2) -> AdapterRun:
    config = AdapterRunConfig(input_mode="thinker", metadata={"team": "t1"})
    return AdapterRun(name=name, config=config, slot=slot, version=version, registration_id=reg)


def make_child(run: AdapterRun) -> QueueChildRolloutFn:
    source = ThinkerOperationSource(SimpleNamespace(), run)
    return QueueChildRolloutFn(RolloutFnConstructorInput(args=source.args, data_source=source))


def sample_payload(n=2) -> dict:
    return {
        "batch_id": "batch-7",
        "samples": [
            {"prompt": "p", "tokens": [1, 2, 3, 4], "response_length": 2, "loss_mask": [1, 1]} for _ in range(n)
        ],
        "loss": {"loss_fn": "cross_entropy"},
    }


class _FakeController:
    """Scripted claim results; records failures."""

    def __init__(self, claims):
        self._claims = list(claims)
        self.failed: list[tuple] = []
        self.claim_data_operation = SimpleNamespace(remote=lambda name, reg: self._next_claim())
        self.fail_operation = SimpleNamespace(remote=lambda *args: self.failed.append(args))

    def _next_claim(self):
        return self._claims.pop(0) if self._claims else None


@pytest.fixture()
def fake_ray(monkeypatch):
    monkeypatch.setattr(queue_module, "ray", SimpleNamespace(get=lambda ref: ref))
    monkeypatch.setattr(queue_module, "_CLAIM_POLL_S", 0.01)

    def install(controller):
        import miles.ray.multi_lora.controller as controller_module

        monkeypatch.setattr(controller_module, "get_multi_lora_controller", lambda: controller)

    return install


def op(op_id="op1", kind="forward_backward", payload=None, ordinal=1):
    return dict(
        operation_id=op_id,
        name="X",
        registration_id="rx",
        ordinal=ordinal,
        kind=kind,
        payload=sample_payload() if payload is None else payload,
        state="CLAIMED",
        result=None,
        error=None,
        error_category=None,
    )


def test_one_operation_becomes_one_stamped_batch(fake_ray):
    controller = _FakeController([op()])
    fake_ray(controller)
    child = make_child(make_run())
    output = asyncio.run(child(RolloutFnTrainInput(rollout_id=0)))

    assert len(output.samples) == 2 and all(len(group) == 1 for group in output.samples)
    stamped = output.samples[0][0]
    assert (stamped.adapter.name, stamped.adapter.registration_id) == ("X", "rx")
    assert stamped.adapter.serving_version == 2 and stamped.adapter.slot == 3
    assert stamped.metadata["team"] == "t1"  # run metadata merged in
    assert stamped.status == stamped.Status.COMPLETED
    assert output.metadata == dict(
        operation_id="op1",
        operation_kind="forward_backward",
        batch_id="batch-7",
        step_after_backward=False,
        loss_spec={"loss_fn": "cross_entropy"},
    )


def test_child_waits_for_a_claim(fake_ray):
    # None, None, then an operation: the child polls until work arrives.
    controller = _FakeController([None, None, op()])
    fake_ray(controller)
    child = make_child(make_run())
    output = asyncio.run(child(RolloutFnTrainInput(rollout_id=0)))
    assert output.metadata["operation_id"] == "op1"


def test_bad_payload_fails_its_operation_and_the_child_continues(fake_ray):
    controller = _FakeController([op("bad", payload={"samples": []}), op("good")])
    fake_ray(controller)
    child = make_child(make_run())
    output = asyncio.run(child(RolloutFnTrainInput(rollout_id=0)))

    assert output.metadata["operation_id"] == "good"
    [(failed_id, error, category)] = controller.failed
    assert failed_id == "bad" and category == "user" and "no samples" in error


def test_forward_operations_build_batches_too(fake_ray):
    controller = _FakeController(
        [
            op(
                "fwd",
                kind="forward",
                payload={"samples": [{"prompt": "p", "tokens": [1, 2], "response_length": 1, "loss_mask": [1]}]},
            )
        ]
    )
    fake_ray(controller)
    child = make_child(make_run())
    output = asyncio.run(child(RolloutFnTrainInput(rollout_id=0)))
    assert output.metadata["operation_kind"] == "forward"
    assert output.metadata["step_after_backward"] is False
    assert controller.failed == []
