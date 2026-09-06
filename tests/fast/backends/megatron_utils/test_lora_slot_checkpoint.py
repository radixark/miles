"""CPU Adam regression coverage for slot checkpoint precision and isolation."""

import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


class MasterAdam:
    """CPU model/master pair matching Megatron's mixed-precision ownership."""

    def __init__(self, values):
        self.main = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))
        self.model = torch.nn.Parameter(self.main.detach().to(torch.bfloat16))
        self.optimizer = torch.optim.Adam([self.main], lr=0.001)

    def get_parameters(self):
        return [self.main]

    def reload_model_params(self):
        with torch.no_grad():
            self.main.copy_(self.model)

    def step(self):
        self.main.grad = torch.tensor([0.5, -0.25])
        self.optimizer.step()
        self.optimizer.zero_grad()
        with torch.no_grad():
            self.model.copy_(self.main)


class SlotModel:
    def __init__(self, children):
        self.children = children
        self.exposed_slot = None

    def named_parameters(self):
        return [("layer.adapter.weight", self.children[self.exposed_slot].model)]


@pytest.fixture
def rig(monkeypatch):
    # Only framework imports are stubbed: checkpoint code and torch Adam run.
    def stub(name, **attrs):
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        return module

    stub("megatron")
    stub("megatron.core")
    stub("megatron.core.distributed", DistributedDataParallel=object)
    stub("megatron.core.optimizer", MegatronOptimizer=object, get_megatron_optimizer=None)
    stub("megatron.core.optimizer.optimizer", MegatronOptimizer=object)
    stub("megatron.core.optimizer.optimizer_config", OptimizerConfig=object)
    stub("megatron.core.optimizer.layer_wise_optimizer", LayerWiseDistributedOptimizer=object)
    stub("megatron.core.optimizer.clip_grads", clip_grad_by_total_norm_fp32=None, get_grad_norm_fp32=None)
    stub("megatron.core.process_groups_config", ProcessGroupCollection=object)
    stub("miles.backends.training_utils.parallel", get_parallel_state=None)
    stub("miles.utils.distributed_utils", get_gloo_group=None)
    stub(
        "miles.backends.megatron_utils.lora.slots", adapter_shard_topology=lambda: (True, ()), megatron_shard_name=None
    )

    root = Path(__file__).resolve().parents[4] / "miles/backends/megatron_utils/lora"

    def load(name):
        full_name = f"miles.backends.megatron_utils.lora.{name}"
        spec = importlib.util.spec_from_file_location(full_name, root / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, full_name, module)
        spec.loader.exec_module(module)
        return module

    optimizer_module = load("optimizer")
    checkpoint = load("checkpoint")
    monkeypatch.setattr(checkpoint, "_weight_shard_name", lambda: "weights.pt")
    monkeypatch.setattr(checkpoint, "_barrier", lambda: None)
    children = [MasterAdam([0.5009, 1.0017]), MasterAdam([0.7509, 1.5017])]
    model = SlotModel(children)
    optimizer = SimpleNamespace(chained_optimizers=children, miles_slot_child_indices={0: [0], 1: [1]})
    optimizer.reload_model_params = lambda: [child.reload_model_params() for child in children]

    @contextmanager
    def expose_adapter_slot(models, slot):
        model.exposed_slot = slot
        yield
        model.exposed_slot = None

    def load_adapter(models, slot, state):
        with torch.no_grad():
            children[slot].model.copy_(state["layer.adapter.weight"])
        return 1

    stub("megatron.bridge.peft.multi_lora_layers", expose_adapter_slot=expose_adapter_slot, load_adapter=load_adapter)
    return SimpleNamespace(
        checkpoint=checkpoint,
        optimizer_module=optimizer_module,
        children=children,
        model=[model],
        optimizer=optimizer,
        stub=stub,
        load=load,
    )


def test_resume_preserves_uninterrupted_master_and_adam_update(rig, tmp_path):
    child = rig.children[0]
    child.step()
    saved_master = child.main.detach().clone()
    assert not torch.equal(saved_master, child.model.float()), "exercise precision absent from BF16 weights"
    rig.checkpoint.save_slot(rig.model, rig.optimizer, 0, str(tmp_path / "checkpoint"))
    child.step()
    expected_master, expected_model = child.main.detach().clone(), child.model.detach().clone()
    expected_moments = {key: value.clone() for key, value in child.optimizer.state[child.main].items()}

    rig.checkpoint.load_slot(rig.model, rig.optimizer, 0, str(tmp_path / "checkpoint"), load_optimizer=True)
    assert torch.equal(child.main, saved_master)
    child.step()

    assert torch.equal(child.main, expected_master)
    assert torch.equal(child.model, expected_model)
    for key, value in expected_moments.items():
        assert torch.equal(child.optimizer.state[child.main][key], value)


@pytest.mark.parametrize("load_optimizer", [True, False])
def test_loading_one_slot_preserves_other_slot_master_and_pending_gradient(rig, tmp_path, load_optimizer):
    other = rig.children[1]
    other.step()
    other.main.grad = torch.tensor([2.0, 3.0])
    expected_master, expected_grad = other.main.detach().clone(), other.main.grad.clone()
    rig.checkpoint.save_slot(rig.model, rig.optimizer, 0, str(tmp_path / "checkpoint"))

    rig.checkpoint.load_slot(rig.model, rig.optimizer, 0, str(tmp_path / "checkpoint"), load_optimizer=load_optimizer)

    assert torch.equal(other.main, expected_master)
    assert torch.equal(other.main.grad, expected_grad)


def test_weights_only_load_rebuilds_only_target_masters(rig, tmp_path):
    child = rig.children[0]
    rig.checkpoint.save_slot(rig.model, rig.optimizer, 0, str(tmp_path / "checkpoint"))
    child.step()
    moments = {key: value.clone() for key, value in child.optimizer.state[child.main].items()}

    rig.checkpoint.load_slot(rig.model, rig.optimizer, 0, str(tmp_path / "checkpoint"), load_optimizer=False)

    assert torch.equal(child.main, child.model.float())
    for key, value in moments.items():
        assert torch.equal(child.optimizer.state[child.main][key], value)


def test_legacy_checkpoint_warns_and_restores_moments_from_rounded_weights(rig, tmp_path, caplog):
    child = rig.children[0]
    child.step()
    path = tmp_path / "checkpoint"
    rig.checkpoint.save_slot(rig.model, rig.optimizer, 0, str(path))
    state_path = path / "optim_rank0.pt"
    saved = torch.load(state_path, weights_only=True)
    for child_state in saved["children"]:
        child_state.pop("main_params", None)
    torch.save(saved, state_path)
    expected_moments = {key: value.clone() for key, value in child.optimizer.state[child.main].items()}
    child.step()

    rig.checkpoint.load_slot(rig.model, rig.optimizer, 0, str(path), load_optimizer=True)

    assert torch.equal(child.main, child.model.float())
    assert "master parameters" in caplog.text
    for key, value in expected_moments.items():
        assert torch.equal(child.optimizer.state[child.main][key], value)


def test_same_rank_checkpoint_can_restore_into_another_slot_without_copying_slot_tags(rig, tmp_path):
    source, target = rig.children
    source.step()
    expected_master = source.main.detach().clone()
    for slot, child in enumerate(rig.children):
        child.optimizer.param_groups[0]["miles_multi_lora_slot"] = slot
    rig.checkpoint.save_slot(rig.model, rig.optimizer, 0, str(tmp_path / "checkpoint"))

    rig.checkpoint.load_slot(rig.model, rig.optimizer, 1, str(tmp_path / "checkpoint"), load_optimizer=True)

    assert torch.equal(target.main, expected_master)
    assert torch.equal(source.main, expected_master)
    assert target.optimizer.param_groups[0]["miles_multi_lora_slot"] == 1


def test_empty_owned_child_can_round_trip(rig):
    child = SimpleNamespace(optimizer=SimpleNamespace(param_groups=[], state={}), get_parameters=lambda: [])
    optimizer = SimpleNamespace(chained_optimizers=[child], miles_slot_child_indices={0: [0]})
    saved = rig.checkpoint._optimizer_slot_state(optimizer, 0)
    rig.checkpoint._load_optimizer_slot_state(optimizer, 0, saved)


@pytest.mark.parametrize("action", ["load_slot", "unload_slot"])
def test_slot_lifecycle_keeps_other_masters_and_pending_gradients(rig, monkeypatch, action):
    rig.stub("miles.backends.megatron_utils.model", run_forward_backward_pass=None, setup_train_iteration_config=None)
    rig.stub("miles.backends.training_utils.data", get_data_iterator=None)
    rig.stub("miles.backends.training_utils.log_utils", aggregate_train_losses=None)
    rig.stub(
        "miles.backends.training_utils.loss_hub.tinker_losses",
        drain_per_datum_outputs=None,
        start_per_datum_outputs=None,
    )
    rig.stub("miles.utils.dumper_utils", DumperMegatronUtil=None, DumperPhase=None)
    rig.stub("miles.utils.types", RolloutBatch=dict)
    monkeypatch.setattr(rig.optimizer_module, "zero_adapter_slot_grads", lambda *args: None)
    slots = sys.modules["miles.backends.megatron_utils.lora.slots"]
    monkeypatch.setattr(slots, "zero_optimizer_state_for_adapter", lambda *args: None, raising=False)
    bridge = sys.modules["megatron.bridge.peft.multi_lora_layers"]

    def initialize(models, slot, **kwargs):
        with torch.no_grad():
            rig.children[slot].model.fill_(0.25)

    monkeypatch.setattr(bridge, "init_adapter_slot", initialize, raising=False)
    monkeypatch.setattr(bridge, "clear_adapter_slot", initialize, raising=False)
    executor = rig.load("executor")
    other = rig.children[1]
    other.main.grad = torch.tensor([2.0, 3.0])
    expected_master, expected_grad = other.main.detach().clone(), other.main.grad.clone()
    kwargs = {"rank": 8, "alpha": 16} if action == "load_slot" else {}

    getattr(executor, action)(rig.model, rig.optimizer, slot=0, **kwargs)

    assert torch.equal(rig.children[0].main, rig.children[0].model.float())
    assert torch.equal(other.main, expected_master)
    assert torch.equal(other.main.grad, expected_grad)
