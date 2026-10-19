import importlib
import sys
from argparse import Namespace
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from tests.fast.utils.test_utils.fault_injector.fakes import _arm_marker_hook

from miles.backends.training_utils.model_companion import ModelCompanion
from miles.utils.audit_utils.event_logger import logger as event_logger_module
from miles.utils.audit_utils.event_logger.logger import EventLogger, read_events
from miles.utils.audit_utils.event_logger.models import FaultHookEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.tensor_backper import TensorBackuper
from miles.utils.test_utils.fault_injector.actions.base import FaultHookContext
from miles.utils.test_utils.fault_injector.controller import reach_fault_hook
from miles.utils.test_utils.fault_injector.models import FaultHookName, FaultHookStatus
from miles.utils.types import SampleLineage

_ACTOR_MODULE_NAME = "miles.backends.megatron_utils.actor"


@pytest.fixture(scope="module")
def actor_module() -> Iterator[ModuleType]:
    """Import the Megatron actor with its unavailable native memory dependency stubbed."""
    package = importlib.import_module("miles.backends.megatron_utils")
    missing = object()
    saved_module = sys.modules.get(_ACTOR_MODULE_NAME, missing)
    saved_saver = sys.modules.get("torch_memory_saver", missing)
    saved_package_attr = getattr(package, "actor", missing)

    saver_module = ModuleType("torch_memory_saver")
    saver_module.torch_memory_saver = Mock()
    sys.modules["torch_memory_saver"] = saver_module
    sys.modules.pop(_ACTOR_MODULE_NAME, None)
    if saved_package_attr is not missing:
        delattr(package, "actor")

    try:
        yield importlib.import_module(_ACTOR_MODULE_NAME)
    finally:
        sys.modules.pop(_ACTOR_MODULE_NAME, None)
        if saved_module is not missing:
            sys.modules[_ACTOR_MODULE_NAME] = saved_module
        if saved_package_attr is missing:
            if hasattr(package, "actor"):
                delattr(package, "actor")
        else:
            package.actor = saved_package_attr
        if saved_saver is missing:
            sys.modules.pop("torch_memory_saver", None)
        else:
            sys.modules["torch_memory_saver"] = saved_saver


class TestCriticValuesValueSpec:
    def test_critic_values_are_shipped_as_a_typed_ragged_field(self, actor_module: ModuleType) -> None:
        """Variable-length critic sequences require the typed ragged object-store codec."""
        assert actor_module.CRITIC_VALUES_VALUE_SPEC["values"].codec == "typed_ragged"


class TestMaterializeCriticValues:
    @pytest.mark.parametrize("tensor_input", [False, True])
    def test_ragged_values_become_owned_float32_tensors(self, actor_module: ModuleType, tensor_input: bool) -> None:
        """Both store codecs yield owned FP32 tensors independent of released storage."""
        sequences = [[1.5, -2.0], [], [3.0]]
        values = [torch.tensor(value, dtype=torch.float32) for value in sequences] if tensor_input else sequences

        result = actor_module._materialize_critic_values(values=values, device=torch.device("cpu"))
        values[0][0] = 99.0

        assert [value.tolist() for value in result] == [[1.5, -2.0], [], [3.0]]
        assert all(value.dtype == torch.float32 for value in result)
        assert all(value.device == torch.device("cpu") for value in result)


def test_actor_ref_actor_switch_restores_model_companion(
    actor_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The actor backup tag restores its companion after a ref model switch."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    real_empty_like = torch.empty_like
    monkeypatch.setattr(
        torch,
        "empty_like",
        lambda tensor, **kwargs: real_empty_like(
            tensor, **{key: value for key, value in kwargs.items() if key != "pin_memory"}
        ),
    )
    train_actor = object.__new__(actor_module.MegatronTrainRayActor)
    train_actor.args = SimpleNamespace(colocate=False, keep_old_actor=False, megatron_to_hf_mode="bridge")
    train_actor.with_ref = True
    train_actor.with_opd_teacher = False
    train_actor.model = [torch.nn.Module()]
    train_actor.model[0].add_module(
        "model_companion", ModelCompanion(pipeline_rank=0, chunk_index=0, replica_id=(0, 0, 0))
    )
    weight = torch.nn.Parameter(torch.tensor([1.0]))
    train_actor.model[0].register_parameter("weight", weight)
    train_actor.weights_backuper = TensorBackuper.create(
        source_getter=partial(train_actor._named_actor_weights, include_model_companion=True)
    )
    train_actor._active_model_tag = "actor"
    train_actor._asleep = False
    actor_sample = SampleLineage(source_sample_index=7, output_index=0, output_count=1)
    ref_sample = SampleLineage(source_sample_index=8, output_index=0, output_count=1)
    train_actor.model[0].model_companion.record_sample_consumptions([actor_sample])
    train_actor.weights_backuper.backup("actor")
    weight.data.fill_(2)
    train_actor.model[0].model_companion.record_sample_consumptions([ref_sample])
    train_actor.weights_backuper.backup("ref")
    train_actor._active_model_tag = "ref"

    assert set(train_actor._get_actor_weights()) == {"vp_stages.0.weight"}

    train_actor._switch_model("actor")

    assert train_actor.model[0].model_companion.snapshot_sample_consumptions(is_skipped=False) == {actor_sample: 1}
    assert torch.equal(weight, torch.tensor([1.0]))
    assert set(train_actor._get_actor_weights()) == {"vp_stages.0.weight"}

    train_actor._switch_model("ref")

    assert train_actor.model[0].model_companion.snapshot_sample_consumptions(is_skipped=False) == {
        actor_sample: 1,
        ref_sample: 1,
    }
    assert torch.equal(weight, torch.tensor([2.0]))


class TestSendCheckpoint:
    def test_healing_before_the_first_train_step_is_refused_without_sending(
        self, actor_module: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Healing before the first train step is refused without transferring a checkpoint."""
        train_actor = object.__new__(actor_module.MegatronTrainRayActor)
        train_actor.args = Namespace(keep_old_actor=False)
        train_actor._last_rollout_id = None
        train_actor.model = object()
        train_actor.optimizer = object()
        train_actor.opt_param_scheduler = object()
        checkpoint_transfer_attempted = False

        def record_checkpoint_transfer(**_kwargs: object) -> None:
            nonlocal checkpoint_transfer_attempted
            checkpoint_transfer_attempted = True

        monkeypatch.setattr(actor_module, "get_parallel_state", lambda: SimpleNamespace(indep_dp=object()))
        monkeypatch.setattr(actor_module, "_send_ckpt", record_checkpoint_transfer)

        with pytest.raises(AssertionError, match="healing before the first train step is unsupported"):
            train_actor.send_ckpt(dst_rank=1)

        assert not checkpoint_transfer_attempted


@pytest.mark.parametrize(
    "enabled,role,outcome,representative,published",
    [
        (True, "actor", "NORMAL", True, True),
        (False, "actor", "NORMAL", True, False),
        (True, "critic", "NORMAL", True, False),
        (True, "actor", "DISCARDED_SHOULD_RETRY", True, False),
        (True, "actor", "NORMAL", False, False),
    ],
)
def test_model_companion_info_records_only_normal_actor_representatives(
    actor_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    role: str,
    outcome: str,
    representative: bool,
    published: bool,
) -> None:
    """Only a normal actor result on the local representative records model companion info."""
    actor = object.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(enable_sample_ownership_checker=enabled)
    actor.role = role
    actor.model = []
    actor._cell_index = 2
    calls = []
    monkeypatch.setattr(actor_module, "is_local_replica_megatron_main_rank", lambda: representative)
    monkeypatch.setattr(
        actor_module.SampleOwnershipRecorder,
        "publish_model_companion_info",
        lambda *args, **kwargs: calls.append(kwargs),
    )

    actor._publish_model_companion_info(
        rollout_id=7,
        attempt=3,
        result=actor_module.TrainStepOutput(outcome=actor_module.TrainStepOutcome[outcome]),
    )

    assert bool(calls) is published
    if published:
        assert calls[0]["cell_index"] == 2
        assert calls[0]["rollout_id"] == 7
        assert calls[0]["attempt"] == 3


class _FakeWeightUpdater:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.weight_versions: list[int] = []
        self.conn_status = SimpleNamespace(needs_reconnect=lambda snapshot: False)
        self.protocol = SimpleNamespace(cell_updaters_of_cell_id={})

    def update_weights(self, *, weight_version: int) -> None:
        self.weight_versions.append(weight_version)
        reach_fault_hook(FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_SEND)
        if self.error is not None:
            raise self.error


def _make_updating_actor(
    actor_module: ModuleType, monkeypatch: pytest.MonkeyPatch, updater: _FakeWeightUpdater
) -> object:
    monkeypatch.setattr(actor_module, "print_memory", lambda *args, **kwargs: None)
    train_actor = object.__new__(actor_module.MegatronTrainRayActor)
    train_actor.args = Namespace(
        debug_train_only=False,
        debug_rollout_only=False,
        offload_train=False,
        debug_skip_weight_update=False,
        ci_test=False,
        keep_old_actor=False,
        rematerialize_param_from_master_weight=False,
    )
    train_actor._heartbeat = Mock()
    train_actor._asleep = False
    train_actor.weight_updater = updater
    train_actor._get_actor_weight_version = lambda: 7
    return train_actor


def _update(train_actor: object, *, rollout_id: int | None) -> object:
    info = SimpleNamespace(
        rollout_engines=[],
        snapshot_cell_id_to_hashes={"rollout-0": "hash-a"},
        engine_gpu_counts=[],
        engine_gpu_offsets=[],
        engine_cell_ids=[],
    )
    return train_actor.update_weights(info=info, debug_weight_update_id="update-1", rollout_id=rollout_id)


class TestUpdateWeightsFaultHookContext:
    def test_the_send_hook_inherits_the_rollout_version_and_update_of_this_update(
        self, actor_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A hook reached inside the update must match on, and record, the update that reached it."""
        log: list[object] = []
        hooks = _arm_marker_hook(
            monkeypatch,
            log=log,
            hook_name=FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_SEND,
            rollout_id=3,
            weight_version=7,
        )
        monkeypatch.setattr(actor_module, "fault_hook_controller", hooks)
        monkeypatch.setattr(
            event_logger_module,
            "_event_logger",
            EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="main")),
        )
        updater = _FakeWeightUpdater()

        output = _update(_make_updating_actor(actor_module, monkeypatch, updater), rollout_id=3)

        assert log == [("hook", FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_SEND.value)]
        assert updater.weight_versions == [7]
        assert output.weight_version == 7
        [fired] = [
            event.record
            for event in read_events(tmp_path)
            if isinstance(event, FaultHookEvent) and event.record.status == FaultHookStatus.FIRED
        ]
        assert fired.context == FaultHookContext(
            rollout_id=3,
            weight_version=7,
            debug_weight_update_id="update-1",
            snapshot_cell_id_to_hashes={"rollout-0": "hash-a"},
        )

    @pytest.mark.parametrize("filters", [{"rollout_id": 4}, {"weight_version": 8}])
    def test_a_hook_armed_for_another_rollout_or_version_does_not_fire(
        self, actor_module: ModuleType, monkeypatch: pytest.MonkeyPatch, filters: dict[str, int]
    ) -> None:
        """A request for another step's update must not fire during this one."""
        log: list[object] = []
        hooks = _arm_marker_hook(
            monkeypatch, log=log, hook_name=FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_SEND, **filters
        )
        monkeypatch.setattr(actor_module, "fault_hook_controller", hooks)

        _update(_make_updating_actor(actor_module, monkeypatch, _FakeWeightUpdater()), rollout_id=3)

        assert log == []

    def test_the_context_is_dropped_when_the_update_fails(
        self, actor_module: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed update must not leave its version for a later hook to match against."""
        log: list[object] = []
        hooks = _arm_marker_hook(
            monkeypatch, log=log, hook_name=FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_ALL_GATHER, weight_version=7
        )
        monkeypatch.setattr(actor_module, "fault_hook_controller", hooks)
        updater = _FakeWeightUpdater(error=RuntimeError("send failed"))

        with pytest.raises(RuntimeError, match="send failed"):
            _update(_make_updating_actor(actor_module, monkeypatch, updater), rollout_id=3)
        reach_fault_hook(FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_ALL_GATHER)

        assert log == []
