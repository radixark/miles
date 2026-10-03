"""Full-model admission, accumulated Adam updates, and immutable sampler versions."""

import asyncio
from contextlib import suppress
from types import SimpleNamespace

import httpx
import pytest
import torch
from tests.fast.tinker.harness import ADAM, await_settled, datum, make_service, model_payload

from miles.backends.megatron_utils.full_training.optimizer import GradientAccumulator
from miles.tinker.arguments import configure_tinker_args
from miles.tinker.core.future import DONE
from miles.tinker.core.types import UserInputError
from miles.tinker.core.utils import resolve_sampler_checkpoint
from miles.tinker.full_training import FullTrainingBackend
from miles.tinker.runtime import MilesBackend
from miles.tinker.server.app import build_app


@pytest.fixture
async def full_service(tmp_path):
    service = make_service(tmp_path, full_training=True, n_slots=1, trains_unembed=True)
    task = asyncio.create_task(service.run())
    try:
        yield service
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


async def _create_full(service):
    request, model = service.create_model("tenant", model_payload(service, parameterization={"type": "full"}))
    assert (await await_settled(service, "tenant", request)).state == DONE
    return model


async def test_full_creation_through_http(full_service):
    transport = httpx.ASGITransport(app=build_app(full_service))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://gateway", headers={"X-API-Key": "tenant"}
    ) as http:
        session = (await http.post("/api/v1/create_session", json={})).json()["session_id"]
        payload = {
            "session_id": session,
            "model_seq_id": 0,
            "base_model": "base",
            "parameterization": {"type": "full"},
        }
        bad = await http.post("/api/v1/create_model", json=payload | {"parameterization": {"type": "lora"}})
        assert bad.status_code == 400
        bad = await http.post("/api/v1/create_model", json=payload | {"unexpected": True})
        assert bad.status_code == 400
        created = await http.post("/api/v1/create_model", json=payload)
        assert created.status_code == 200, created.text
        body = created.json()
        assert (await await_settled(full_service, "tenant", body["request_id"])).state == DONE
        info = await http.post("/api/v1/get_info", json={"model_id": body["model_id"]})
        assert info.json()["is_lora"] is False


async def test_full_model_capacity_and_parameterization(full_service):
    with pytest.raises(UserInputError, match="lora_config"):
        full_service.create_model("tenant", model_payload(full_service, lora_config={"rank": 8}))
    model = await _create_full(full_service)
    assert full_service.models[model].lora_rank is None
    assert full_service.backend.named("load_slot")[0]["rank"] is None
    with pytest.raises(UserInputError, match="one active model"):
        await _create_full(full_service)


async def test_full_checkpoints_restore_without_lora_metadata(full_service):
    model = await _create_full(full_service)
    request = full_service.submit(
        "tenant", "save_state", {"model_id": model, "seq_id": 1, "name": "saved", "overwrite": False}
    )
    future = await await_settled(full_service, "tenant", request)
    assert future.state == DONE
    path = future.result["path"]
    assert full_service.weights_info("tenant", path)["is_lora"] is False
    request = full_service.submit(
        "tenant", "load_state", {"model_id": model, "seq_id": 2, "path": path, "optimizer": True}
    )
    assert (await await_settled(full_service, "tenant", request)).state == DONE
    assert full_service.backend.named("load_slot")[-1]["load_optimizer"] is True


async def test_sampler_rejects_cross_parameterization(full_service):
    model = await _create_full(full_service)
    request = full_service.submit(
        "tenant", "save_weights_for_sampler", {"model_id": model, "seq_id": 1, "sampler_path": "v1"}
    )
    future = await await_settled(full_service, "tenant", request)
    assert future.state == DONE
    path = future.result["path"]
    resolve_sampler_checkpoint(full_service.config.checkpoint_root, "tenant", path, "base", is_lora=False)
    with pytest.raises(UserInputError, match="parameterization"):
        resolve_sampler_checkpoint(full_service.config.checkpoint_root, "tenant", path, "base")


async def test_full_runtime_preserves_targets_and_omits_adapter_routing():
    backend = MilesBackend(None, "unused", dp_size=2, full_training=True)

    async def execute(method, batch_id, data):
        assert "adapter_slots" not in data
        assert data["target_tokens"] == [[1, 2, 3], [1, 2, 3]]
        assert data["loss_masks"] == [[1, 1, 1], [0, 0, 0]]
        return [{"per_datum": [{"sample_index": 0, "loss": 1, "logprobs": torch.zeros(3)}]}]

    backend._call_trainer = execute
    assert len(await backend.forward_backward(0, [(0, datum())], "cross_entropy", {})) == 1


class NativeOptimizerShape:
    """The native optimizer interface over a real CPU Adam and fresh per-pass gradients."""

    def __init__(self, parameter):
        self.parameter = parameter
        self.inner = torch.optim.Adam([parameter])
        self.param_groups = self.inner.param_groups
        self.next_gradient = None

    def get_parameters(self):
        return [self.parameter]

    def prepare_grads(self):
        self.parameter.grad = self.next_gradient.clone()
        return False

    def get_grad_norm(self):
        return self.parameter.grad.norm()

    def step_with_ready_grads(self):
        self.inner.step()
        return True

    def zero_grad(self):
        self.inner.zero_grad(set_to_none=True)


@pytest.mark.parametrize("clip", [0.0, 0.5])
def test_accumulation_matches_one_adam_update_on_the_sum(clip):
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    reference = torch.nn.Parameter(p.detach().clone())
    native = NativeOptimizerShape(p)
    accumulator = GradientAccumulator(native)
    adam = dict(ADAM, grad_clip_norm=clip, learning_rate=0.01, weight_decay=0.1)
    ref_optimizer = torch.optim.Adam(
        [reference], lr=adam["learning_rate"], betas=(adam["beta1"], adam["beta2"]), eps=adam["eps"], weight_decay=0.1
    )
    for _ in range(2):
        for gradient in (torch.tensor([3.0, -2.0]), torch.tensor([-1.0, 7.0])):
            native.next_gradient = gradient
            accumulator.add()
            native.zero_grad()
        reference.grad = torch.tensor([2.0, 5.0])
        expected_norm = float(reference.grad.norm())
        if clip:
            torch.nn.utils.clip_grad_norm_([reference], clip)
        ref_optimizer.step()
        outcome = accumulator.step(adam)
        assert outcome["grad_norm"] == pytest.approx(expected_norm)
        torch.testing.assert_close(p, reference, rtol=0, atol=0)
        assert not accumulator.gradients and accumulator.num_batches == 0
        assert "error" in accumulator.step(adam)


def test_nonfinite_gradients_do_not_change_weights_or_adam_state():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    native = NativeOptimizerShape(p)
    accumulator = GradientAccumulator(native)
    native.next_gradient = torch.tensor([float("nan")])
    accumulator.add()
    assert accumulator.step(ADAM) == {"skipped_nonfinite": 1}
    assert p.item() == 1.0 and not native.inner.state
    assert accumulator.num_batches == 0


class Engine:
    def __init__(self):
        self.version = None
        self.loads = []

    async def update_weights_from_disk(self, model_path, weight_version, load_format):
        assert load_format == "auto"
        self.version = weight_version
        self.loads.append((model_path, weight_version))
        return {"success": True}

    async def get_weight_version(self):
        return self.version


class Controller:
    def __init__(self):
        self.engines = [Engine(), Engine()]
        self.hashes = {"cell": "first"}
        self.aborted = False

    async def start_update_weights(self):
        return SimpleNamespace(rollout_engines=self.engines, snapshot_cell_id_to_hashes=dict(self.hashes))

    async def end_update_weights(self, hashes):
        pass

    async def abort_update_weights(self):
        self.aborted = True


async def test_snapshot_switch_loads_every_replica_and_rechecks_replacements():
    controller = Controller()
    backend = FullTrainingBackend(None, "unused", inference_controller=controller, base_checkpoint="/base")
    await backend.pin_snapshot("v1", "/v1")
    await backend.pin_snapshot("v1", "/v1")
    assert [len(e.loads) for e in controller.engines] == [1, 1]
    controller.hashes["cell"] = "replacement"
    controller.engines[1] = Engine()
    await backend.pin_snapshot("v1", "/v1")
    assert [e.version for e in controller.engines] == ["v1", "v1"]
    await backend.pin_snapshot(None, None)
    assert all(e.loads[-1] == ("/base", "tinker-base") for e in controller.engines)


async def test_cancelled_sample_finishes_before_weights_can_change(monkeypatch):
    controller = Controller()
    backend = FullTrainingBackend(None, "unused", inference_controller=controller, base_checkpoint="/base")
    started, finish = asyncio.Event(), asyncio.Event()

    async def generate(self, payload, lora_name, lora_path=None):
        assert lora_name is None and lora_path is None
        version = controller.engines[0].version
        started.set()
        await finish.wait()
        assert all(e.version == version for e in controller.engines)
        return {"sequences": []}

    monkeypatch.setattr(MilesBackend, "sample", generate)
    first = asyncio.create_task(backend.sample({}, "v1", "/v1"))
    await started.wait()
    first.cancel()
    second = asyncio.create_task(backend.sample({}, "v2", "/v2"))
    await asyncio.sleep(0)
    assert all(e.version == "v1" for e in controller.engines)
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    await second
    assert all(e.version == "v2" for e in controller.engines)


async def test_failed_replica_load_never_reaches_generation(monkeypatch):
    controller = Controller()
    backend = FullTrainingBackend(None, "unused", inference_controller=controller, base_checkpoint="/base")

    async def failed(**kwargs):
        return {"success": False, "message": "active requests"}

    controller.engines[1].update_weights_from_disk = failed
    result = await backend.sample({}, "v1", "/v1")
    assert "weight load failed" in result["error"]
    assert controller.aborted and backend._loaded_snapshot is None


@pytest.mark.parametrize(
    "unsupported",
    [
        "lora_rank",
        "calculate_per_token_loss",
        "overlap_param_gather",
        "use_precision_aware_optimizer",
        "debug_disable_optimizer",
    ],
)
def test_full_launch_rejects_incompatible_training_modes(unsupported):
    args = SimpleNamespace(
        train_backend="megatron",
        exclude_modules=None,
        tinker_full_training=True,
        multi_lora_n_adapters=0,
        lora_rank=0,
        lora_adapter_path=None,
        bf16=True,
        fp16=False,
        loss_scale=None,
        optimizer="adam",
        calculate_per_token_loss=False,
        context_parallel_size=1,
        offload_train=False,
        colocate=False,
        overlap_param_gather=False,
        use_precision_aware_optimizer=False,
        use_dynamic_batch_size=True,
        micro_batch_size=1,
    )
    configure_tinker_args(args)
    setattr(args, unsupported, True)
    with pytest.raises(AssertionError):
        configure_tinker_args(args)


async def test_cancelled_waiter_does_not_switch_the_fleet():
    controller = Controller()
    backend = FullTrainingBackend(None, "unused", inference_controller=controller, base_checkpoint="/base")
    async with backend._sampling_lock:
        waiting = asyncio.create_task(backend.sample({}, "v1", "/v1"))
        await asyncio.sleep(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
    assert all(not engine.loads for engine in controller.engines)
