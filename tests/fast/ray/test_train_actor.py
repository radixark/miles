import os
import socket
import sys
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from tests.fast.fixtures.args_fixtures import make_trainer_config, parse_megatron_test_config
from tests.fast.train_parallel_config_utils import make_train_parallel_config

from miles.ray import placement_group, train_actor
from miles.ray.specs.train import compute_trainer_configs
from miles.ray.train_actor import TrainRayActor, WeightUpdateOutput
from miles.utils import distributed_utils, logging_utils
from miles.utils.args.trainer_utils import compute_trainer_config
from miles.utils.dp_schedule import TrainParallelConfig
from miles.utils.init_once import InitOnce
from miles.utils.workers.env_vars import CELL_INDEX_ENV_VAR, SUBPROCESS_INDEX_ENV_VAR


def _unimplemented_actor_operation(*args: object, **kwargs: object) -> None:
    raise NotImplementedError


class _ActorWithoutReloadSupport(TrainRayActor):
    init = _unimplemented_actor_operation
    sleep = _unimplemented_actor_operation
    wake_up = _unimplemented_actor_operation
    train = _unimplemented_actor_operation
    save_model = _unimplemented_actor_operation
    update_weights = _unimplemented_actor_operation
    _get_parallel_config = _unimplemented_actor_operation


def _inited_guard() -> InitOnce:
    guard = InitOnce("TrainRayActor")
    with guard.guarding():
        pass
    return guard


class TestConstructorSignature:
    def test_positional_constructor_arguments_are_rejected(self):
        """Workers are built from a spec's kwargs, so silently shifted positional args must not construct one."""
        with pytest.raises(TypeError):
            TrainRayActor(SimpleNamespace(), 2, 1, "10.0.0.1:1234", "actor", 0)


class TestProposeMasterAddrAndPort:
    def test_the_proposal_steps_past_a_port_that_is_already_taken(self, monkeypatch: pytest.MonkeyPatch):
        """A cell rendezvouses on the proposing worker's own node, on a port no other process already holds."""
        monkeypatch.setattr(train_actor, "get_current_node_ip", lambda: "10.0.0.3")

        with socket.socket() as occupied:
            occupied.bind(("", train_actor.get_free_port(start_port=20500)))
            occupied.listen(1)
            taken_port = occupied.getsockname()[1]
            monkeypatch.setattr(train_actor.random, "randint", lambda _low, _high: taken_port)

            addr, port = TrainRayActor.__new__(TrainRayActor).propose_master_addr_and_port()

        assert addr == "10.0.0.3"
        assert port > taken_port
        with socket.socket() as probe:
            probe.bind(("", port))


class TestKillSelf:
    def test_kill_self_exits_with_a_failure_status(self, monkeypatch: pytest.MonkeyPatch):
        """A worker asked to die must leave no survivor and must not look like a clean shutdown."""
        exit_statuses: list[int] = []
        monkeypatch.setattr(train_actor.os, "_exit", exit_statuses.append)

        TrainRayActor.__new__(TrainRayActor).kill_self()

        assert exit_statuses == [1]


class TestLoadState:
    def test_an_actor_without_reload_support_refuses_in_place_state_loading(self):
        """An unsupported backend must require a restart instead of pretending to reload in place."""
        actor = _ActorWithoutReloadSupport.__new__(_ActorWithoutReloadSupport)

        with pytest.raises(
            NotImplementedError,
            match="_ActorWithoutReloadSupport cannot reload its state without restarting",
        ):
            actor.load_state(ckpt_load_config=None)


class TestConfigureMasterAddrAndPort:
    def _make_actor(self) -> TrainRayActor:
        return TrainRayActor.__new__(TrainRayActor)

    def test_the_master_addr_and_port_land_in_the_environment(self, monkeypatch: pytest.MonkeyPatch):
        """The driver-assigned addr/port must reach the env vars that torch's env:// init reads."""
        monkeypatch.delenv("MASTER_ADDR", raising=False)
        monkeypatch.delenv("MASTER_PORT", raising=False)

        self._make_actor().configure_master_addr_and_port(master_addr="10.0.0.1", master_port=20001)

        assert os.environ["MASTER_ADDR"] == "10.0.0.1"
        assert os.environ["MASTER_PORT"] == "20001"

    def test_a_stale_master_addr_and_port_are_overwritten(self, monkeypatch: pytest.MonkeyPatch):
        """A worker inheriting another run's env must end up on the addr/port the driver assigned."""
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", "1")

        self._make_actor().configure_master_addr_and_port(master_addr="10.0.0.2", master_port=20002)

        assert os.environ["MASTER_ADDR"] == "10.0.0.2"
        assert os.environ["MASTER_PORT"] == "20002"


class TestTrainParallelConfigWiring:
    async def test_the_driver_passes_the_resolved_actor_config_to_the_rollout_executor(self, monkeypatch):
        """The driver resolves the actor config before handing it to the rollout executor."""
        train_parallel_config = make_train_parallel_config(dp_size=4)
        args = parse_megatron_test_config()
        [trainer_config] = compute_trainer_configs(args)

        class FakeActorHandle:
            async def get_train_parallel_config(self) -> TrainParallelConfig | None:
                return train_parallel_config

        class FakeRolloutExecutor:
            def __init__(self):
                self.received_config = None

            async def set_train_parallel_config(self, config: TrainParallelConfig | None) -> None:
                assert not isinstance(config, FakeActorHandle)
                self.received_config = config

            async def load(self, rollout_id):
                pass

        actor_handle = FakeActorHandle()
        rollout_executor = FakeRolloutExecutor()
        monkeypatch.setattr(placement_group, "compute_trainer_configs", lambda args: [trainer_config])
        monkeypatch.setattr(placement_group, "take_over_trainers", lambda args, *, handles: _return(False))
        monkeypatch.setattr(
            placement_group,
            "create_training_model",
            lambda *, handle, trainer_id, request, requested_start_rollout_id, resumed: _return(
                placement_group.TrainerInfo(handle=actor_handle, restored_rollout_id=0, start_rollout_id=0)
            ),
        )

        await placement_group.create_training_models(
            args,
            rollout_executor=rollout_executor,
            handles={trainer_config.trainer_id: actor_handle},
        )

        assert rollout_executor.received_config is train_parallel_config


async def _return(value):
    return value


def _actor_with(guard: InitOnce) -> TrainRayActor:
    actor = TrainRayActor.__new__(TrainRayActor)
    actor._init_once = guard
    return actor


class TestInitRunsExactlyOnce:
    def test_init_rebinds_reporting_after_distributed_rank_is_final(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The reporter must receive the rank and world size discovered by process-group initialization."""
        run = parse_megatron_test_config("--distributed-backend", "nccl", "--distributed-timeout-minutes", "1")
        args = compute_trainer_config(run, compute_trainer_configs(run)[0])
        actor = _ActorWithoutReloadSupport.__new__(_ActorWithoutReloadSupport)
        actor._init_once = InitOnce("TrainRayActor")
        actor._heartbeat = SimpleNamespace(bump=lambda: None)
        rebound: list[object] = []
        monkeypatch.setattr(train_actor.torch.cuda, "set_device", lambda _device: None)
        monkeypatch.setattr(train_actor.dist, "init_process_group", lambda **_kwargs: None)
        monkeypatch.setattr(train_actor.dist, "get_rank", lambda: 3)
        monkeypatch.setattr(train_actor.dist, "get_world_size", lambda: 8)
        monkeypatch.setattr(train_actor, "init_gloo_group", lambda: None)
        monkeypatch.setattr(train_actor, "rebind_env_reporting", rebound.append)
        monkeypatch.setattr(train_actor.torch.version, "hip", "test")

        actor._init_common(args=args, role="actor")

        assert (args.backend.rank, args.backend.world_size) == (3, 8)
        assert rebound == [args]

    @pytest.mark.parametrize("indep_dp", [False, True])
    @pytest.mark.parametrize("original_dump", [None, "1"])
    def test_only_independent_trainers_ignore_cross_cell_dump_signals(
        self, monkeypatch: pytest.MonkeyPatch, indep_dp: bool, original_dump: str | None
    ) -> None:
        """Recoverable cross-cell failures cannot terminate the default process group via a dump signal."""
        run = parse_megatron_test_config(
            "--use-dynamic-batch-size", "--max-tokens-per-gpu", "1024", *(["--indep-dp"] if indep_dp else [])
        )
        args = compute_trainer_config(run, compute_trainer_configs(run)[0])
        actor = _ActorWithoutReloadSupport.__new__(_ActorWithoutReloadSupport)
        actor._init_once = InitOnce("TrainRayActor")
        actor._heartbeat = SimpleNamespace(bump=lambda: None)
        observed: list[str | None] = []
        if original_dump is None:
            monkeypatch.delenv("TORCH_NCCL_DUMP_ON_TIMEOUT", raising=False)
        else:
            monkeypatch.setenv("TORCH_NCCL_DUMP_ON_TIMEOUT", original_dump)
        monkeypatch.setattr(train_actor.torch.cuda, "set_device", lambda _device: None)
        monkeypatch.setattr(
            train_actor.dist,
            "init_process_group",
            lambda **_kwargs: observed.append(os.environ.get("TORCH_NCCL_DUMP_ON_TIMEOUT")),
        )
        monkeypatch.setattr(train_actor.dist, "get_rank", lambda: 0)
        monkeypatch.setattr(train_actor.dist, "get_world_size", lambda: 2)
        monkeypatch.setattr(
            train_actor, "init_gloo_group", lambda: observed.append(os.environ.get("TORCH_NCCL_DUMP_ON_TIMEOUT"))
        )
        monkeypatch.setattr(train_actor, "rebind_env_reporting", lambda _args: None)
        monkeypatch.setattr(train_actor.torch.version, "hip", "test")

        actor._init_common(args=args, role="actor")

        assert observed == ["0" if indep_dp else original_dump, original_dump]
        assert os.environ.get("TORCH_NCCL_DUMP_ON_TIMEOUT") == original_dump

    def test_a_second_init_is_refused(self):
        """A worker that already initialized is a stale process; reusing it must fail loudly, not train on."""
        actor = _actor_with(_inited_guard())

        with pytest.raises(AssertionError, match="stale worker"):
            actor._init_common(None, "actor")

    def test_a_worker_that_never_ran_init_reports_itself_uninitialized(self):
        """A restarted script asks a worker it found running whether to initialize it or to resume it."""
        assert _actor_with(InitOnce("TrainRayActor")).is_initialized() is False

    def test_a_worker_that_ran_init_reports_itself_initialized(self):
        """The take-over path has to see the worker the previous script built as built."""
        assert _actor_with(_inited_guard()).is_initialized() is True


class TestTheLocalGpuIsFoundWithoutRay:
    def test_a_supervised_rank_reads_its_index_rather_than_asking_ray(self, monkeypatch):
        """A served worker is a plain process, not an actor, so ray owns no assignment and answers []."""
        monkeypatch.setenv(CELL_INDEX_ENV_VAR, "0")
        monkeypatch.setenv(SUBPROCESS_INDEX_ENV_VAR, "3")
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.setattr(train_actor.ray, "get_gpu_ids", lambda: [])

        assert train_actor.get_local_gpu_id() == 3

    def test_a_pod_running_one_worker_needs_no_supervisor_to_know_its_card(self, monkeypatch):
        """A cell one worker wide is launched without the supervisor, so the index it would set is absent."""
        monkeypatch.setenv(CELL_INDEX_ENV_VAR, "0")
        monkeypatch.delenv(SUBPROCESS_INDEX_ENV_VAR, raising=False)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.setattr(train_actor.ray, "get_gpu_ids", lambda: [])

        assert train_actor.get_local_gpu_id() == 0

    def test_a_ray_placed_actor_still_reads_its_assignment_from_ray(self, monkeypatch):
        """Every existing run takes this path, where ray does own the assignment."""
        monkeypatch.delenv(CELL_INDEX_ENV_VAR, raising=False)
        monkeypatch.delenv(SUBPROCESS_INDEX_ENV_VAR, raising=False)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
        monkeypatch.setattr(train_actor.ray, "get_gpu_ids", lambda: [6])

        assert train_actor.get_local_gpu_id() == 2

    def test_a_ray_actor_without_a_visible_device_list_still_reads_its_assignment(self, monkeypatch):
        """The other ray shape: no mask, so the assignment is the id itself."""
        monkeypatch.delenv(CELL_INDEX_ENV_VAR, raising=False)
        monkeypatch.delenv(SUBPROCESS_INDEX_ENV_VAR, raising=False)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.setattr(train_actor.ray, "get_gpu_ids", lambda: [5])

        assert train_actor.get_local_gpu_id() == 5


# PCI bus numbers of the eight GPUs on an HGX H200 board, in NVML (physical) index order.
_PHYSICAL_GPU_PCI_BUSES = [0x18, 0x2A, 0x3A, 0x5D, 0x9A, 0xAB, 0xBA, 0xDB]


class _FakeCuda:
    """torch.cuda under a device mask: local device i is the i-th physical GPU of the mask."""

    def __init__(self, *, visible_physical_gpus: list[int]) -> None:
        self._visible_physical_gpus = visible_physical_gpus
        self._current_device: int | None = None

    def set_device(self, device: str) -> None:
        self._current_device = int(device.removeprefix("cuda:"))

    def current_device(self) -> int:
        return self._current_device

    def get_device_properties(self, device: int) -> SimpleNamespace:
        bus = _PHYSICAL_GPU_PCI_BUSES[self._visible_physical_gpus[device]]
        return SimpleNamespace(pci_domain_id=0, pci_bus_id=bus, pci_device_id=0)


class _FakeNvml:
    """NVML indexes every GPU on the host physically, whatever CUDA_VISIBLE_DEVICES says.

    Bus ids are matched in the canonical form NVML itself reports (nvmlPciInfo_t.busId).
    """

    def __init__(self) -> None:
        self._bus_ids = [f"00000000:{bus:02X}:00.0" for bus in _PHYSICAL_GPU_PCI_BUSES]
        self.affinity_set_for: list[int] = []

    def nvmlInit(self) -> None:
        pass

    def nvmlShutdown(self) -> None:
        pass

    def nvmlDeviceGetHandleByIndex(self, index: int) -> int:
        return index

    def nvmlDeviceGetHandleByPciBusId(self, bus_id: str) -> int:
        return self._bus_ids.index(bus_id)

    def nvmlDeviceGetIndex(self, handle: int) -> int:
        return handle

    def nvmlDeviceSetCpuAffinity(self, handle: int) -> None:
        self.affinity_set_for.append(handle)


class TestNumaAffinity:
    def test_a_masked_rank_takes_the_cpus_of_the_gpu_it_runs_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Under mask 4,5,6,7 local rank 0 runs on GPU 4, so it must take GPU 4's NUMA node, not GPU 0's."""
        args = make_trainer_config(
            debug_deterministic_collective=False,
            distributed_backend="nccl",
            distributed_timeout_minutes=1,
            num_gpus_per_node=4,
        )
        actor = _ActorWithoutReloadSupport.__new__(_ActorWithoutReloadSupport)
        actor._init_once = InitOnce("TrainRayActor")
        actor._heartbeat = SimpleNamespace(bump=lambda: None)
        cuda = _FakeCuda(visible_physical_gpus=[4, 5, 6, 7])
        nvml = _FakeNvml()
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setitem(sys.modules, "pynvml", nvml)
        monkeypatch.setattr(train_actor.torch.version, "hip", None)
        monkeypatch.setattr(train_actor.torch.cuda, "set_device", cuda.set_device)
        monkeypatch.setattr(train_actor.torch.cuda, "current_device", cuda.current_device)
        monkeypatch.setattr(train_actor.torch.cuda, "get_device_properties", cuda.get_device_properties)
        monkeypatch.setattr(train_actor.dist, "init_process_group", lambda **_kwargs: None)
        monkeypatch.setattr(train_actor.dist, "get_rank", lambda: 0)
        monkeypatch.setattr(train_actor.dist, "get_world_size", lambda: 4)
        monkeypatch.setattr(train_actor, "init_gloo_group", lambda: None)
        monkeypatch.setattr(train_actor, "rebind_env_reporting", lambda _args: None)

        actor._init_common(args=args, role="actor")

        assert nvml.affinity_set_for == [4]


class TestWeightUpdateOutputMerge:
    def test_merging_nothing_answers_an_empty_report(self):
        """An update window with no engines runs no trainer cell, and that is not a failure."""
        assert WeightUpdateOutput.merge([]) == WeightUpdateOutput(weight_version=None, failed_cell_ids=())

    def test_trainer_cells_that_all_succeeded_publish_one_version_and_no_failure(self):
        """The driver publishes this version to the executor, so a merge must not invent a second one."""
        outputs = [
            WeightUpdateOutput(weight_version=5, failed_cell_ids=()),
            WeightUpdateOutput(weight_version=5, failed_cell_ids=()),
        ]

        assert WeightUpdateOutput.merge(outputs) == WeightUpdateOutput(weight_version=5, failed_cell_ids=())

    def test_the_failed_cell_ids_of_every_trainer_cell_are_concatenated(self):
        """A dropped id leaves an engine serving half-written weights in service."""
        outputs = [
            WeightUpdateOutput(weight_version=5, failed_cell_ids=("rollout-1",)),
            WeightUpdateOutput(weight_version=5, failed_cell_ids=("rollout-3", "rollout-4")),
        ]

        merged = WeightUpdateOutput.merge(outputs)

        assert merged.weight_version == 5
        assert set(merged.failed_cell_ids) == {"rollout-1", "rollout-3", "rollout-4"}

    def test_one_rollout_cell_blamed_by_two_trainer_cells_is_rejected(self):
        """The targets are split disjointly, so the same id twice means the split leaked."""
        outputs = [
            WeightUpdateOutput(weight_version=5, failed_cell_ids=("rollout-1",)),
            WeightUpdateOutput(weight_version=5, failed_cell_ids=("rollout-1",)),
        ]

        with pytest.raises(AssertionError, match="more than one trainer cell"):
            WeightUpdateOutput.merge(outputs)

    def test_every_target_failing_still_merges_into_one_report(self):
        """The caller decides what a total failure means, so merge itself must not raise on it."""
        outputs = [
            WeightUpdateOutput(weight_version=None, failed_cell_ids=("rollout-1",)),
            WeightUpdateOutput(weight_version=None, failed_cell_ids=("rollout-2",)),
        ]

        merged = WeightUpdateOutput.merge(outputs)

        assert merged.weight_version is None
        assert set(merged.failed_cell_ids) == {"rollout-1", "rollout-2"}

    def test_a_trainer_cell_that_published_no_version_does_not_erase_a_surviving_ones(self):
        """A dead trainer's report carries no version, and answering None would republish nothing to the engines."""
        outputs = [
            WeightUpdateOutput(weight_version=None, failed_cell_ids=("rollout-1",)),
            WeightUpdateOutput(weight_version=9, failed_cell_ids=()),
        ]

        assert WeightUpdateOutput.merge(outputs).weight_version == 9

    def test_trainer_cells_that_disagree_on_the_version_are_rejected(self):
        """Two live versions mean the engines now serve different weights under one version number."""
        outputs = [
            WeightUpdateOutput(weight_version=5, failed_cell_ids=()),
            WeightUpdateOutput(weight_version=6, failed_cell_ids=()),
        ]

        with pytest.raises(AssertionError, match="disagree on the weight version"):
            WeightUpdateOutput.merge(outputs)

    def test_a_skipped_broadcast_from_every_trainer_cell_answers_no_version(self):
        """--debug-skip-weight-update returns None everywhere, which must reach the driver as None."""
        outputs = [
            WeightUpdateOutput(weight_version=None, failed_cell_ids=()),
            WeightUpdateOutput(weight_version=None, failed_cell_ids=()),
        ]

        assert WeightUpdateOutput.merge(outputs) == WeightUpdateOutput(weight_version=None, failed_cell_ids=())


class TestBackendProcessGroupSdk:
    @pytest.mark.parametrize(
        ("backend", "offload", "cpu_backend", "expected"),
        [
            ("megatron", False, None, "nccl"),
            ("fsdp", True, "gloo", "cpu:gloo,cuda:nccl"),
            ("fsdp", False, "gloo", "nccl"),
            ("fsdp", True, None, "nccl"),
            ("torchtitan", True, "gloo", "cpu:gloo,cuda:nccl"),
        ],
    )
    def test_backend_and_timeout_reach_real_common_initialization_sdk(
        self, monkeypatch: pytest.MonkeyPatch, backend: str, offload: bool, cpu_backend: str | None, expected: str
    ) -> None:
        """Common worker initialization selects hybrid CPU offload only when configured and preserves the backend timeout."""
        args = make_trainer_config(
            train_backend=backend,
            distributed_backend="nccl",
            distributed_timeout_minutes=17,
            optimizer_cpu_offload=offload,
            fsdp_cpu_backend=cpu_backend,
            debug_deterministic_collective=False,
        )
        actor = _ActorWithoutReloadSupport.__new__(_ActorWithoutReloadSupport)
        actor._init_once = InitOnce("TrainRayActor")
        actor._heartbeat = SimpleNamespace(bump=lambda: None)
        initialize_group, gloo_group = Mock(), Mock()
        monkeypatch.setattr(train_actor.torch.cuda, "set_device", lambda _device: None)
        monkeypatch.setattr(train_actor.dist, "init_process_group", initialize_group)
        monkeypatch.setattr(train_actor.dist, "new_group", gloo_group)
        monkeypatch.setattr(train_actor.dist, "get_rank", lambda: 3)
        monkeypatch.setattr(train_actor.dist, "get_world_size", lambda: 8)
        monkeypatch.setattr(train_actor.torch.version, "hip", "test")
        monkeypatch.setattr(distributed_utils, "GLOO_GROUP", None)
        monkeypatch.setattr(logging_utils, "_ENV_REPORTER", None)

        actor._init_common(args=args, role="actor")

        initialize_group.assert_called_once_with(backend=expected, timeout=timedelta(minutes=17))
        gloo_group.assert_called_once_with(backend="gloo")
        assert (args.backend.rank, args.backend.world_size) == (3, 8)

    @pytest.mark.parametrize("valid", [False, True])
    def test_deterministic_collective_backend_is_checked_before_world_initialization(
        self, monkeypatch: pytest.MonkeyPatch, valid: bool
    ) -> None:
        """Deterministic training requires the registered backend before creating the world process group."""
        from miles.utils.test_utils import det_process_group

        backend = train_actor.DET_NCCL_BACKEND_NAME if valid else "nccl"
        args = make_trainer_config(
            distributed_backend=backend,
            distributed_timeout_minutes=19,
            debug_deterministic_collective=True,
        )
        actor = _ActorWithoutReloadSupport.__new__(_ActorWithoutReloadSupport)
        actor._init_once = InitOnce("TrainRayActor")
        actor._heartbeat = SimpleNamespace(bump=lambda: None)
        initialize_group, register_backend = Mock(), Mock()
        monkeypatch.setattr(train_actor.torch.cuda, "set_device", lambda _device: None)
        monkeypatch.setattr(train_actor.dist, "init_process_group", initialize_group)
        monkeypatch.setattr(train_actor.dist, "new_group", Mock())
        monkeypatch.setattr(train_actor.dist, "get_rank", lambda: 0)
        monkeypatch.setattr(train_actor.dist, "get_world_size", lambda: 1)
        monkeypatch.setattr(train_actor.dist.Backend, "register_backend", register_backend)
        monkeypatch.setattr(train_actor.torch.version, "hip", "test")
        monkeypatch.setattr(distributed_utils, "GLOO_GROUP", None)
        monkeypatch.setattr(logging_utils, "_ENV_REPORTER", None)
        monkeypatch.setattr(det_process_group, "_backend_registered", False)

        if valid:
            actor._init_common(args=args, role="actor")
            initialize_group.assert_called_once_with(backend=backend, timeout=timedelta(minutes=19))
        else:
            with pytest.raises(AssertionError):
                actor._init_common(args=args, role="actor")
            initialize_group.assert_not_called()

        assert register_backend.call_count == 1
        assert register_backend.call_args.args[0] == train_actor.DET_NCCL_BACKEND_NAME
