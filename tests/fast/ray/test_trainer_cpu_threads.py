import logging
import sys
from types import SimpleNamespace

import pytest
from tests.fast.fixtures.args_fixtures import (
    make_trainer_config,
    parse_fsdp_test_config,
    parse_megatron_test_config,
    replace_config_values,
)
from tests.fast.fixtures.capability_fixtures import FakeBackendCapability
from tests.fast.ray.test_train_actor import _ActorWithoutReloadSupport, _FakeCuda, _FakeNvml

from miles.ray import train_actor
from miles.ray.specs.train import TrainerSpec
from miles.ray.train_actor import resolve_trainer_cpu_threads
from miles.utils import distributed_utils, logging_utils
from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.args.trainer_utils import compute_trainer_config
from miles.utils.init_once import InitOnce
from miles.utils.workers.connection_config import WorkerPodMetadata
from miles.utils.workers.env_vars import (
    CELL_INDEX_ENV_VAR,
    POD_INDEX_ENV_VAR,
    SUBPROCESS_INDEX_ENV_VAR,
    WORKER_METADATA_ENV_VAR,
)
from miles.utils.workers.ray_worker_manager import RayWorkerManager, _BaseActorManager, _CellManager, _ctor_context
from miles.utils.workers.serving.worker_identity import read_worker_identity
from miles.utils.workers.worker_spec import PortInfo, WorkerLaunchContext


def _make_actor(*, num_workers_per_node: int | None = None) -> _ActorWithoutReloadSupport:
    actor = _ActorWithoutReloadSupport.__new__(_ActorWithoutReloadSupport)
    actor._init_once = InitOnce("TrainRayActor")
    actor._heartbeat = SimpleNamespace(bump=lambda: None)
    actor._num_workers_per_node = num_workers_per_node
    return actor


def _patch_common_init(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rank: int = 0,
    world_size: int = 4,
    device_type: str = "cuda",
    nvml_numa_affinity: bool = False,
) -> list[int]:
    recorded_threads: list[int] = []
    monkeypatch.setattr(
        train_actor,
        "current_platform",
        SimpleNamespace(
            bind_device_before_process_group=lambda _rank: None,
            bind_device_after_process_group=lambda: None,
            nvml_numa_affinity=nvml_numa_affinity,
            device_type=device_type,
        ),
    )
    monkeypatch.setattr(train_actor.dist, "init_process_group", lambda **_kwargs: None)
    monkeypatch.setattr(train_actor.dist, "new_group", lambda **_kwargs: object())
    monkeypatch.setattr(train_actor.dist, "get_rank", lambda: rank)
    monkeypatch.setattr(train_actor.dist, "get_world_size", lambda: world_size)
    monkeypatch.setattr(train_actor.torch, "set_num_threads", recorded_threads.append)
    monkeypatch.setattr(distributed_utils, "GLOO_GROUP", None)
    monkeypatch.setattr(logging_utils, "_ENV_REPORTER", None)
    return recorded_threads


def _spec_and_launch_ctx(*, trainer_cpu_threads: int | None = None, train_env_vars: dict[str, str] | None = None):
    args = replace_config_values(
        parse_megatron_test_config("--actor-num-gpus-per-node", "4"),
        offload_train=False,
        trainer_cpu_threads=trainer_cpu_threads,
        train_env_vars=train_env_vars or {},
    )
    (spec,) = [TrainerSpec.create(cfg) for cfg in TrainerSpec.slice_configs(args)]
    ctx = WorkerLaunchContext(
        args=spec.args,
        cell_index=0,
        worker_in_cell_index=0,
        num_workers_per_cell=4,
        gpu_ids=[0],
        num_workers_per_node=4,
    )
    return spec, ctx


class TestResolveTrainerCpuThreads:
    def test_default_without_cpu_offload_leaves_process_threads_untouched(self) -> None:
        """Colocated GPU runs rely on Ray's 1-thread actor default unless CPU offload or explicit threads are set."""
        assert (
            resolve_trainer_cpu_threads(
                trainer_cpu_threads=None,
                optimizer_cpu_offload=False,
                train_env_vars={},
                avail_cpus=128,
                workers_on_node=8,
            )
            is None
        )

    @pytest.mark.parametrize(
        ("avail_cpus", "workers_on_node", "numa_cpus", "expected"),
        [
            (180, 4, None, 45),
            (192, 4, None, 48),
            (256, 4, None, 48),
            (128, 8, 64, 16),
            (128, 8, 12, 12),
            (2, 8, None, 1),
        ],
    )
    def test_optimizer_cpu_offload_auto_sizes_from_host_affinity_and_local_workers(
        self, avail_cpus: int, workers_on_node: int, numa_cpus: int | None, expected: int
    ) -> None:
        """CPU offload divides available host CPUs across local trainer workers and clamps to [1, 48]."""
        assert (
            resolve_trainer_cpu_threads(
                trainer_cpu_threads=None,
                optimizer_cpu_offload=True,
                train_env_vars={},
                avail_cpus=avail_cpus,
                workers_on_node=workers_on_node,
                numa_cpus=numa_cpus,
            )
            == expected
        )

    def test_explicit_trainer_cpu_threads_overrides_default_and_auto_sizing(self) -> None:
        """An explicit --trainer-cpu-threads value wins whether --optimizer-cpu-offload is enabled or not."""
        for offload in (False, True):
            assert (
                resolve_trainer_cpu_threads(
                    trainer_cpu_threads=12,
                    optimizer_cpu_offload=offload,
                    train_env_vars={},
                    avail_cpus=192,
                    workers_on_node=4,
                )
                == 12
            )

    def test_user_omp_num_threads_in_train_env_vars_is_not_overridden_by_auto_sizing(self) -> None:
        """An operator who pinned OMP_NUM_THREADS in --train-env-vars keeps that setting under CPU offload."""
        assert (
            resolve_trainer_cpu_threads(
                trainer_cpu_threads=None,
                optimizer_cpu_offload=True,
                train_env_vars={"OMP_NUM_THREADS": "8"},
                avail_cpus=192,
                workers_on_node=4,
            )
            is None
        )


class TestTrainerCpuThreadsCliAndEnvVars:
    def test_cli_parses_trainer_cpu_threads_and_propagates_to_trainer_config(self) -> None:
        """The CLI flag parses on both Megatron and FSDP backends and reaches the sliced TrainerConfig."""
        megatron_cfg = parse_megatron_test_config("--trainer-cpu-threads", "24")
        fsdp_cfg = parse_fsdp_test_config("--trainer-cpu-threads", "16")

        trainer_cfg = compute_trainer_config(megatron_cfg, megatron_cfg.raw_megatron.trainers[0])
        assert megatron_cfg.trainer_cpu_threads == 24
        assert trainer_cfg.trainer_cpu_threads == 24
        assert fsdp_cfg.trainer_cpu_threads == 16

    @pytest.mark.parametrize("bad_value", ["0", "-4"])
    def test_non_positive_trainer_cpu_threads_is_rejected(self, bad_value: str) -> None:
        """Zero or negative thread counts would break torch.set_num_threads at worker startup."""
        with pytest.raises(ValueError, match="--trainer-cpu-threads must be >= 1"):
            parse_megatron_test_config("--trainer-cpu-threads", bad_value)

    def test_env_vars_omit_omp_num_threads_by_default(self) -> None:
        """Without --trainer-cpu-threads, driver-side env rendering does not inject a driver-sized OMP_NUM_THREADS."""
        spec, ctx = _spec_and_launch_ctx()
        assert "OMP_NUM_THREADS" not in spec.env_var(ctx)

    def test_env_vars_set_omp_num_threads_when_trainer_cpu_threads_is_set(self) -> None:
        """Explicit --trainer-cpu-threads also exports OMP_NUM_THREADS for OpenMP runtimes before worker import."""
        spec, ctx = _spec_and_launch_ctx(trainer_cpu_threads=20)
        assert spec.env_var(ctx)["OMP_NUM_THREADS"] == "20"

    def test_env_vars_preserve_explicit_train_env_vars_omp_num_threads(self) -> None:
        """--train-env-vars OMP_NUM_THREADS is never clobbered by compute_trainer_env_vars."""
        spec, ctx = _spec_and_launch_ctx(trainer_cpu_threads=20, train_env_vars={"OMP_NUM_THREADS": "6"})
        assert spec.env_var(ctx)["OMP_NUM_THREADS"] == "6"


class TestTrainRayActorCpuThreadsInit:
    def test_init_common_leaves_threads_unchanged_without_cpu_offload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default non-offload trainer initialization never calls torch.set_num_threads."""
        args = make_trainer_config(
            train_backend="megatron",
            optimizer_cpu_offload=False,
            trainer_cpu_threads=None,
            train_env_vars={},
            distributed_backend="nccl",
            distributed_timeout_minutes=1,
            debug_deterministic_collective=False,
        )
        recorded = _patch_common_init(monkeypatch, world_size=8)
        monkeypatch.setattr(train_actor.os, "sched_getaffinity", lambda _pid: set(range(128)), raising=False)

        _make_actor(num_workers_per_node=8)._init_common(args=args, role="actor")

        assert recorded == []

    @pytest.mark.parametrize(
        ("backend", "device_type", "distributed_backend"),
        [
            ("megatron", "cuda", "nccl"),
            ("fsdp", "cuda", "nccl"),
            ("torchtitan", "tpu", "tpu_dist"),
        ],
    )
    def test_init_common_auto_sizes_threads_across_all_backends_on_multi_node(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        backend: str,
        device_type: str,
        distributed_backend: str,
    ) -> None:
        """Multi-node CPU offload divides worker-host affinity by local workers on that node, not global world_size."""
        extra = {"fsdp_cpu_backend": "gloo"} if backend in ("fsdp", "torchtitan") else {}
        args = make_trainer_config(
            train_backend=backend,
            optimizer_cpu_offload=True,
            trainer_cpu_threads=None,
            train_env_vars={},
            distributed_backend=distributed_backend,
            distributed_timeout_minutes=1,
            debug_deterministic_collective=False,
            **extra,
        )
        recorded = _patch_common_init(monkeypatch, rank=0, world_size=16, device_type=device_type)
        monkeypatch.setattr(train_actor.os, "sched_getaffinity", lambda _pid: set(range(180)), raising=False)

        with caplog.at_level(logging.INFO, logger=train_actor.__name__):
            _make_actor(num_workers_per_node=4)._init_common(args=args, role="actor")

        assert recorded == [45]
        assert "Configured PyTorch intra-op CPU threads to 45" in caplog.text

    def test_init_common_samples_host_cpus_before_nvml_numa_pinning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On a 2-socket 128-CPU 8-GPU host, NVML pins each rank to 64 socket CPUs; auto-sizing still picks 128 // 8 = 16."""
        args = make_trainer_config(
            train_backend="megatron",
            optimizer_cpu_offload=True,
            trainer_cpu_threads=None,
            train_env_vars={},
            distributed_backend="nccl",
            distributed_timeout_minutes=1,
            debug_deterministic_collective=False,
        )
        recorded = _patch_common_init(monkeypatch, rank=1, world_size=8, device_type="cuda", nvml_numa_affinity=True)
        affinity = {"cpus": set(range(128))}
        cuda = _FakeCuda(visible_physical_gpus=list(range(8)))

        class _SocketPinningNvml(_FakeNvml):
            def nvmlDeviceSetCpuAffinity(self, handle: int) -> None:
                super().nvmlDeviceSetCpuAffinity(handle)
                affinity["cpus"] = set(range(64))

        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.setitem(sys.modules, "pynvml", _SocketPinningNvml())
        monkeypatch.setattr(train_actor.torch.version, "hip", None)
        monkeypatch.setattr(train_actor.torch.cuda, "set_device", cuda.set_device)
        monkeypatch.setattr(train_actor.torch.cuda, "current_device", lambda: 1)
        monkeypatch.setattr(train_actor.torch.cuda, "get_device_properties", cuda.get_device_properties)
        monkeypatch.setattr(train_actor.os, "sched_getaffinity", lambda _pid: affinity["cpus"], raising=False)

        _make_actor(num_workers_per_node=8)._init_common(args=args, role="actor")

        assert recorded == [16]

    def test_init_common_falls_back_to_cpu_count_without_sched_getaffinity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Platforms without sched_getaffinity fall back cleanly to os.cpu_count()."""
        args = make_trainer_config(
            train_backend="fsdp",
            optimizer_cpu_offload=True,
            fsdp_cpu_backend="gloo",
            trainer_cpu_threads=None,
            train_env_vars={},
            distributed_backend="nccl",
            distributed_timeout_minutes=1,
            debug_deterministic_collective=False,
        )
        recorded = _patch_common_init(monkeypatch, rank=0, world_size=4)

        def _raise_os_error(_pid: int) -> set[int]:
            raise OSError("sched_getaffinity unsupported")

        monkeypatch.setattr(train_actor.os, "sched_getaffinity", _raise_os_error, raising=False)
        monkeypatch.setattr(train_actor.os, "cpu_count", lambda: 64)

        _make_actor(num_workers_per_node=4)._init_common(args=args, role="actor")

        assert recorded == [16]

    def test_init_common_respects_user_omp_num_threads_in_train_env_vars(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit OMP_NUM_THREADS in --train-env-vars suppresses worker auto-sizing."""
        args = make_trainer_config(
            train_backend="torchtitan",
            optimizer_cpu_offload=True,
            fsdp_cpu_backend="gloo",
            trainer_cpu_threads=None,
            train_env_vars={"OMP_NUM_THREADS": "6"},
            distributed_backend="tpu_dist",
            distributed_timeout_minutes=1,
            debug_deterministic_collective=False,
        )
        recorded = _patch_common_init(monkeypatch, rank=0, world_size=4, device_type="tpu")
        monkeypatch.setattr(train_actor.os, "sched_getaffinity", lambda _pid: set(range(192)), raising=False)

        _make_actor(num_workers_per_node=4)._init_common(args=args, role="actor")

        assert recorded == []


class TestWorkersPerNodePropagation:
    def test_ray_launch_context_propagates_per_role_gpus_per_node_to_trainer_ctor_kwargs(self) -> None:
        """Actor and critic pools forward their respective per-node worker counts through Ray launch/ctor contexts."""
        args = replace_config_values(
            parse_megatron_test_config(
                "--actor-num-nodes",
                "2",
                "--actor-num-gpus-per-node",
                "8",
                "--advantage-estimator",
                "ppo",
                "--critic-num-nodes",
                "2",
                "--critic-num-gpus-per-node",
                "4",
            ),
            offload_train=False,
        )
        actor_spec, critic_spec = [TrainerSpec.create(cfg) for cfg in TrainerSpec.slice_configs(args)]
        manager = object.__new__(RayWorkerManager)
        manager.scaling = ScalingConfig.model_construct(
            actor_num_nodes=2,
            actor_num_gpus_per_node=8,
            critic_num_nodes=2,
            critic_num_gpus_per_node=4,
        )
        manager.pgs = {"actor": SimpleNamespace(pg_reordered_gpu_ids=list(range(16)))}

        for spec, expected_per_node, expected_world_size in ((actor_spec, 8, 16), (critic_spec, 4, 8)):
            cell = _CellManager(manager=manager, cell_index=0, spec=spec, actors=None)
            actor_mgr = _BaseActorManager(
                manager=manager,
                parent=cell,
                worker_in_cell_index=0,
                spec=spec,
                actor_handle=None,
                gpu_slot_index=0,
            )
            kwargs = spec.ctor_kwargs(_ctor_context(actor_mgr.launch_context))
            assert kwargs["world_size"] == expected_world_size
            assert kwargs["num_workers_per_node"] == expected_per_node

    def test_kubernetes_worker_identity_propagates_workers_per_pod(self) -> None:
        """Served Kubernetes pods forward workers_per_pod as num_workers_per_node."""
        metadata = WorkerPodMetadata(
            category="trainer_engine",
            workers_per_pod=4,
            pods_per_cell=2,
            gpu_slots_per_worker=1,
            dynamic_pool=True,
            worker_class="miles.backends.torchtitan_utils.actor.TorchtitanTrainRayActor",
            port_infos=[PortInfo(name="rpc", static_port=8000)],
            static_meta=TrainerSpec.model_construct().partial_static_meta,
        )
        environ = {
            WORKER_METADATA_ENV_VAR: metadata.model_dump_json(),
            CELL_INDEX_ENV_VAR: "0",
            POD_INDEX_ENV_VAR: "1",
            SUBPROCESS_INDEX_ENV_VAR: "2",
        }
        identity = read_worker_identity(environ)
        ctx = identity.ctor_context(args=SimpleNamespace(trainer_role="actor"), capability=FakeBackendCapability())
        kwargs = TrainerSpec.model_construct().ctor_kwargs(ctx)

        assert kwargs["world_size"] == 8
        assert kwargs["rank"] == 6
        assert kwargs["num_workers_per_node"] == 4
