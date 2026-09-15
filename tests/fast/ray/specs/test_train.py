import builtins
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
from tests.fast.fixtures.args_fixtures import parser_defaults
from tests.fast.fixtures.capability_fixtures import FakeBackendCapability
from tests.fast.fixtures.megatron_config_fixtures import write_megatron_config, write_megatron_config_trainers

from miles.backends.megatron_utils.megatron_config import compute_trainer_args
from miles.ray.placement_group import _get_placement_group_layout
from miles.ray.specs import train as train_specs
from miles.ray.specs.train import (
    TRAINER_CONCURRENCY_GROUPS,
    TRAINER_CONTROLLER_WORKER_CLASS,
    TRAINER_METHOD_CONCURRENCY_GROUPS,
    compute_actor_args,
    compute_trainer_configs,
    compute_trainer_controller_pool_id,
    compute_trainer_pool_id,
    specs_trainer,
    specs_trainer_controller,
    trainer_controller_cell_id,
    trainer_controller_worker_name,
)
from miles.ray.train_actor import TrainRayActor
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.builder import build_values
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import SECTION_OF_CATEGORY, LaunchPlan
from miles.utils.workers.rpc.common.metadata import _find_rpc_config
from miles.utils.workers.worker_spec import WorkerCtorContext


def _make_args(**overrides) -> SimpleNamespace:
    args = SimpleNamespace(
        actor_num_nodes=1,
        actor_num_gpus_per_node=4,
        critic_num_nodes=1,
        critic_num_gpus_per_node=4,
        use_critic=False,
        indep_dp=False,
        train_backend="megatron",
        use_fault_tolerance=False,
        kl_coef=0,
        use_kl_loss=False,
        use_opd=False,
        opd_type="megatron",
        train_env_vars={},
        dumper_source_patcher_config_train=None,
        offload_train=False,
        offload_train_target="cpu",
        offload_train_disk_dir="/tmp/offload",
        offload_train_disk_chunk_mb=64,
        megatron_config=None,
        trainer_model_id=None,
        advantage_estimator="grpo",
        lr=1e-6,
        optimizer="adam",
        use_distributed_optimizer=True,
        debug_disable_optimizer=False,
        save=None,
        load=None,
        megatron_to_hf_mode="core",
        ref_load=None,
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
        ref_ckpt_step=None,
        ckpt_step=None,
        start_rollout_id=None,
        lr_warmup_iters=None,
        eps_clip=0.2,
        disable_param_buffers_cpu_backup=False,
        critic_load=None,
        critic_save=None,
        critic_lr=None,
        critic_lr_warmup_iters=None,
    )
    args = SimpleNamespace(**{**parser_defaults(), **vars(args)})
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _make_context(**overrides) -> WorkerCtorContext:
    kwargs = dict(cell_index=0, worker_in_cell_index=0, gpu_ids=[0], capability=FakeBackendCapability())
    kwargs.update(overrides)
    return WorkerCtorContext(**kwargs)


def _install_fake_torch_memory_saver(monkeypatch, get_binary_path: MagicMock) -> MagicMock:
    package = ModuleType("torch_memory_saver")
    package.__path__ = []
    utils = ModuleType("torch_memory_saver.utils")
    utils.get_binary_path_from_package = get_binary_path
    monkeypatch.setitem(sys.modules, "torch_memory_saver", package)
    monkeypatch.setitem(sys.modules, "torch_memory_saver.utils", utils)
    monkeypatch.setattr(Path, "read_bytes", lambda self: b"TMS_INIT_ENABLE_DISK_BACKUP")
    return get_binary_path


def _unavailable_fp8_probe() -> str:
    raise RuntimeError("the fp8 hardware default is unavailable here")


class TestSpecSet:
    def test_only_the_actor_is_declared_without_a_critic(self):
        """Most runs have no critic, so no idle critic workers may be scheduled."""
        specs = specs_trainer(_make_args())

        assert [spec.name for spec in specs] == [compute_trainer_pool_id("actor")]

    def test_the_critic_gets_its_own_spec(self):
        """Actor and critic are separate worker sets even though they share GPUs."""
        specs = specs_trainer(_make_args(use_critic=True))

        assert [spec.name for spec in specs] == [
            compute_trainer_pool_id("actor"),
            compute_trainer_pool_id("critic"),
        ]

    def test_the_critic_args_are_neutralized(self):
        """A critic must not apply the actor's KL or on-policy distillation settings."""
        specs = specs_trainer(_make_args(use_critic=True, kl_coef=0.1, use_kl_loss=True, use_opd=True))

        critic_args = specs[1].ctor_kwargs(_make_context())["args"]
        assert (critic_args.kl_coef, critic_args.use_opd) == (0, False)


class TestScheduling:
    def test_actor_and_critic_share_one_placement_group(self):
        """Shared actor/critic PPO puts both roles on the same GPUs."""
        specs = specs_trainer(_make_args(use_critic=True))

        assert {spec.scheduling.pg_name for spec in specs} == {"actor"}

    def test_one_worker_per_gpu_without_independent_dp(self):
        """The trainer world is one rank per GPU in a single cell."""
        (spec,) = specs_trainer(_make_args(actor_num_gpus_per_node=8))

        assert (spec.scheduling.num_cells, spec.scheduling.num_workers_per_cell) == (1, 8)

    def test_independent_dp_splits_the_world_into_cells(self, monkeypatch):
        """Each independent-DP replica becomes one cell the manager can restart alone."""
        monkeypatch.setattr("miles.ray.specs.train.compute_megatron_world_size_except_dp", lambda _args: 2)

        (spec,) = specs_trainer(_make_args(actor_num_gpus_per_node=8, indep_dp=True))

        assert (spec.scheduling.num_cells, spec.scheduling.num_workers_per_cell) == (4, 2)

    def test_independent_dp_critic_cells_use_the_critic_gpu_shape(self, monkeypatch):
        """A critic sized differently from the actor must be split by its own GPU count."""
        monkeypatch.setattr("miles.ray.specs.train.compute_megatron_world_size_except_dp", lambda _args: 2)
        monkeypatch.setattr("miles.ray.specs.train._create_indep_dp_store_addr", lambda: "10.0.0.1:1234")

        _actor_spec, critic_spec = specs_trainer(
            _make_args(
                use_critic=True,
                indep_dp=True,
                actor_num_nodes=3,
                actor_num_gpus_per_node=8,
                critic_num_nodes=2,
                critic_num_gpus_per_node=4,
            )
        )

        assert (critic_spec.scheduling.num_cells, critic_spec.scheduling.num_workers_per_cell) == (4, 2)

    def test_a_nondivisible_independent_dp_trainer_layout_is_rejected(self, monkeypatch):
        """A GPU count that cannot be split into equal cells must fail loudly instead of dropping ranks."""
        monkeypatch.setattr("miles.ray.specs.train.compute_megatron_world_size_except_dp", lambda _args: 2)

        with pytest.raises(AssertionError, match="must be divisible"):
            specs_trainer(_make_args(indep_dp=True, actor_num_nodes=1, actor_num_gpus_per_node=5))

    def test_a_cell_spanning_several_nodes_is_deployed_as_one_pod_per_node(self):
        """A trainer rank owns one gpu, so a two-node cell has to be two pods of a node's worth."""
        (spec,) = specs_trainer(_make_args(actor_num_nodes=2, actor_num_gpus_per_node=8))

        assert (spec.scheduling.pods_per_cell(), spec.scheduling.workers_per_pod()) == (2, 8)

    def test_each_role_packs_its_pods_with_its_own_per_node_count(self):
        """A critic on smaller nodes must not inherit the actor's packing, or every name and port shifts."""
        args = _make_args(
            use_critic=True,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            critic_num_nodes=2,
            critic_num_gpus_per_node=2,
        )

        actor_spec, critic_spec = specs_trainer(args)

        assert (actor_spec.scheduling.pods_per_cell(), actor_spec.scheduling.workers_per_pod()) == (1, 8)
        assert (critic_spec.scheduling.pods_per_cell(), critic_spec.scheduling.workers_per_pod()) == (2, 2)

    def test_a_cell_smaller_than_a_node_fits_into_one_pod(self, monkeypatch):
        """An independent-DP cell of two ranks must not claim a pod bigger than the cell itself."""
        monkeypatch.setattr("miles.ray.specs.train.compute_megatron_world_size_except_dp", lambda _args: 2)

        (spec,) = specs_trainer(_make_args(actor_num_gpus_per_node=8, indep_dp=True))

        assert spec.scheduling.num_workers_per_cell == 2
        assert (spec.scheduling.pods_per_cell(), spec.scheduling.workers_per_pod()) == (1, 2)

    def test_a_worker_reserves_a_fraction_of_its_gpu(self):
        """The rollout engine shares the same GPU slot, so the trainer must not claim it whole."""
        (spec,) = specs_trainer(_make_args())

        assert spec.scheduling.num_gpus_per_worker == 0.4
        assert spec.scheduling.num_gpu_slots_per_worker == 1

    def test_a_trainer_worker_reserves_matching_fractional_cpu_and_gpu_resources(self):
        """Claiming a whole CPU per worker would let Ray refuse to co-schedule the rollout engine."""
        (spec,) = specs_trainer(_make_args())

        assert spec.scheduling.num_cpus_per_worker == 0.4
        assert spec.scheduling.num_cpus_per_worker == spec.scheduling.num_gpus_per_worker

    def test_a_policy_parallelism_override_reshapes_only_its_own_cells(self, tmp_path, monkeypatch):
        """The cell count is computed from that trainer's own args, so an override must reshape only its pool."""
        monkeypatch.setattr(
            "miles.ray.specs.train.compute_megatron_world_size_except_dp",
            lambda args: args.tensor_model_parallel_size,
        )
        args = _make_args(
            indep_dp=True,
            actor_num_gpus_per_node=8,
            tensor_model_parallel_size=1,
            megatron_config=write_megatron_config_trainers(
                tmp_path, [{"model_id": "a", "overrides": {"tensor_model_parallel_size": 2}}, {"model_id": "b"}]
            ),
        )

        spec_a, spec_b = specs_trainer(args)

        assert (spec_a.scheduling.num_cells, spec_b.scheduling.num_cells) == (4, 8)

    def test_a_critic_is_sized_by_the_critic_node_counts(self):
        """The critic reads critic_num_*, so sizing it by the actor's counts would reserve the wrong pool."""
        _actor_spec, critic_spec = specs_trainer(
            _make_args(use_critic=True, actor_num_gpus_per_node=8, critic_num_nodes=1, critic_num_gpus_per_node=2)
        )

        assert critic_spec.scheduling.num_workers_per_cell == 2
        assert critic_spec.scheduling.num_gpus_per_node == 2


class TestConstructorArguments:
    def test_each_worker_learns_its_own_rank(self):
        """Ranks come from the spec now that no worker asks rank 0 for them."""
        (spec,) = specs_trainer(_make_args(actor_num_gpus_per_node=2))

        ranks = [spec.ctor_kwargs(_make_context(worker_in_cell_index=i))["rank"] for i in range(2)]
        assert ranks == [0, 1]

    def test_the_world_size_is_the_cell_size(self):
        """A rank joins the process group of its own cell, not of the whole job."""
        (spec,) = specs_trainer(_make_args(actor_num_gpus_per_node=4))

        assert spec.ctor_kwargs(_make_context())["world_size"] == 4

    def test_no_quorum_store_address_is_baked_into_the_spec(self, monkeypatch):
        """Every pod recomputes the spec, so an address minted here would give each pod its own quorum."""
        monkeypatch.setattr("miles.ray.specs.train.compute_megatron_world_size_except_dp", lambda _args: 2)

        (spec,) = specs_trainer(_make_args(actor_num_gpus_per_node=8, indep_dp=True))

        assert "indep_dp_store_addr" not in spec.ctor_kwargs(_make_context())

    def test_the_backend_selects_the_worker_class(self):
        """A run must not start Megatron workers for an fsdp job."""
        (megatron_spec,) = specs_trainer(_make_args(train_backend="megatron"))
        (fsdp_spec,) = specs_trainer(_make_args(train_backend="fsdp"))

        assert megatron_spec.worker_class.endswith("MegatronTrainRayActor")
        assert fsdp_spec.worker_class.endswith("FSDPTrainRayActor")


class TestConcurrencyGroups:
    def test_the_heartbeat_rpc_is_always_isolated(self):
        """A heartbeat queued behind a train step reads as a dead cell."""
        (spec,) = specs_trainer(_make_args(use_fault_tolerance=True))

        assert spec.concurrency_groups == {"heartbeat_status": 1, "default": 1, "fault_injector": 1, "kill_self": 1}

    def test_the_isolated_methods_travel_with_the_groups(self):
        """Declaring groups without routing any method to them leaves the heartbeat behind a train step."""
        (spec,) = specs_trainer(_make_args(use_fault_tolerance=True))

        assert spec.method_concurrency_groups == {
            "get_heartbeat_status": "heartbeat_status",
            "inject_fault": "fault_injector",
            "kill_self": "kill_self",
        }

    def test_a_run_without_fault_tolerance_gets_a_plain_actor(self):
        """A threaded trainer actor runs NCCL setup off the main thread and deadlocked a non-FT run."""
        (spec,) = specs_trainer(_make_args())

        assert (spec.concurrency_groups, spec.method_concurrency_groups) == (None, None)

    def test_the_actor_is_not_annotated_statically(self):
        """A static @ray.method(concurrency_group=...) makes Ray reject the plain non-FT actor."""
        annotations: list[str | None] = [
            getattr(getattr(TrainRayActor, name), "__ray_concurrency_group__", None)
            for name in TRAINER_METHOD_CONCURRENCY_GROUPS
        ]

        assert annotations == [None, None, None]

    def test_every_routed_method_exists_on_the_trainer_actor(self):
        """A routed name the actor never defines only blows up when a fault-tolerant run launches."""
        methods = [getattr(TrainRayActor, name, None) for name in TRAINER_METHOD_CONCURRENCY_GROUPS]

        assert all(callable(method) for method in methods)

    def test_both_trainer_roles_follow_the_same_gate(self):
        """A critic threaded while its actor is not would deadlock exactly the run the gate protects."""
        fault_tolerant_specs = specs_trainer(_make_args(use_critic=True, use_fault_tolerance=True))
        plain_specs = specs_trainer(_make_args(use_critic=True))

        assert [spec.concurrency_groups is None for spec in fault_tolerant_specs] == [False, False]
        assert [spec.concurrency_groups is None for spec in plain_specs] == [True, True]

    def test_every_routed_group_is_declared(self):
        """Ray rejects an actor whose method names a concurrency group the class never declares."""
        (spec,) = specs_trainer(_make_args(use_fault_tolerance=True))

        assert set(spec.method_concurrency_groups.values()) <= set(TRAINER_CONCURRENCY_GROUPS)

    def test_the_isolated_methods_are_annotated_on_the_actor(self):
        """Dropping an @rpc concurrency group would silently queue that call behind a train step."""
        declared = declared_concurrency_groups(TrainRayActor)

        assert {name: declared.get(name) for name in ("get_heartbeat_status", "inject_fault", "kill_self")} == {
            "get_heartbeat_status": "heartbeat_status",
            "inject_fault": "fault_injector",
            "kill_self": "kill_self",
        }

    def test_every_annotated_group_is_declared(self):
        """Ray rejects an actor whose method names a concurrency group the class never declares."""
        annotated_groups: set[str] = set(declared_concurrency_groups(TrainRayActor).values())

        assert annotated_groups
        assert annotated_groups <= set(TRAINER_CONCURRENCY_GROUPS)

    def test_the_same_methods_are_isolated_under_rpc_communication(self):
        """The rpc server has its own group per method, and an unannotated one queues behind a train step."""
        groups = {
            name: _find_rpc_config(getattr(TrainRayActor, name)).concurrency_group
            for name in ("get_heartbeat_status", "inject_fault", "kill_self")
        }

        assert groups == {
            "get_heartbeat_status": "heartbeat_status",
            "inject_fault": "fault_injector",
            "kill_self": "kill_self",
        }


class TestEnvironmentVariables:
    def test_user_env_vars_are_forwarded(self):
        """--train-env-vars must reach the worker process."""
        (spec,) = specs_trainer(_make_args(train_env_vars={"MY_VAR": "1"}))

        assert spec.env_var(_make_context())["MY_VAR"] == "1"

    def test_user_train_env_vars_override_framework_defaults(self, monkeypatch):
        """A user who overrides a framework default must win, otherwise the flag is unusable."""
        monkeypatch.setenv("NCCL_CUMEM_ENABLE", "0")
        monkeypatch.setenv("NVSHMEM_DISABLE_NCCL", "1")
        monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "0")

        (spec,) = specs_trainer(
            _make_args(
                train_env_vars={
                    "NCCL_CUMEM_ENABLE": "1",
                    "NVSHMEM_DISABLE_NCCL": "0",
                    "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
                }
            )
        )
        env_vars = spec.env_var(_make_context())

        assert (
            env_vars["NCCL_CUMEM_ENABLE"],
            env_vars["NVSHMEM_DISABLE_NCCL"],
            env_vars["NVTE_FP8_BLOCK_SCALING_FP32_SCALES"],
        ) == ("1", "0", "1")

    def test_the_fp8_scaling_default_is_captured_when_the_spec_is_built(self, monkeypatch):
        """The environment is rendered later inside the gpu-less worker manager, which would decide the wrong default."""
        monkeypatch.delenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", raising=False)
        monkeypatch.setattr(train_specs, "default_fp8_block_scaling_fp32_scales", lambda: "0")
        (spec,) = specs_trainer(_make_args())
        monkeypatch.setattr(train_specs, "default_fp8_block_scaling_fp32_scales", lambda: "1")

        assert spec.env_var(_make_context())["NVTE_FP8_BLOCK_SCALING_FP32_SCALES"] == "0"

    def test_an_explicit_fp8_scaling_env_is_still_forwarded(self, monkeypatch):
        """An operator who pinned the value in the launcher environment must still reach the trainer."""
        monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "0")

        (spec,) = specs_trainer(_make_args())

        assert spec.env_var(_make_context())["NVTE_FP8_BLOCK_SCALING_FP32_SCALES"] == "0"

    def test_a_later_fp8_scaling_env_change_does_not_reach_the_trainer(self, monkeypatch):
        """The environment is rendered in another process, so a value looked up at render time is the wrong one."""
        monkeypatch.delenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", raising=False)
        monkeypatch.setattr(train_specs, "default_fp8_block_scaling_fp32_scales", lambda: "0")

        (spec,) = specs_trainer(_make_args())
        monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")

        assert spec.env_var(_make_context())["NVTE_FP8_BLOCK_SCALING_FP32_SCALES"] == "0"

    def test_an_explicit_fp8_scaling_env_is_forwarded_without_probing_the_hardware(self, monkeypatch):
        """A pinned value must reach the trainer even where the hardware default cannot be computed at all."""
        monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")
        monkeypatch.setattr(train_specs, "default_fp8_block_scaling_fp32_scales", _unavailable_fp8_probe)

        (spec,) = specs_trainer(_make_args())

        assert spec.env_var(_make_context())["NVTE_FP8_BLOCK_SCALING_FP32_SCALES"] == "1"

    def test_an_empty_fp8_scaling_env_is_forwarded_verbatim(self, monkeypatch):
        """An empty pin disables fp32 scales, so substituting the hardware default would silently re-enable them."""
        monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "")
        monkeypatch.setattr(train_specs, "default_fp8_block_scaling_fp32_scales", lambda: "1")

        (spec,) = specs_trainer(_make_args())

        assert spec.env_var(_make_context())["NVTE_FP8_BLOCK_SCALING_FP32_SCALES"] == ""

    def test_train_env_vars_override_the_captured_fp8_scaling_default(self, monkeypatch):
        """The captured default is written into the same dict and must not shadow a --train-env-vars entry."""
        monkeypatch.delenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", raising=False)
        monkeypatch.setattr(train_specs, "default_fp8_block_scaling_fp32_scales", lambda: "0")

        (spec,) = specs_trainer(_make_args(train_env_vars={"NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1"}))

        assert spec.env_var(_make_context())["NVTE_FP8_BLOCK_SCALING_FP32_SCALES"] == "1"

    def test_the_critic_spec_captures_the_fp8_scaling_default_as_well(self, monkeypatch):
        """The critic trainer imports transformer engine like the actor, so its spec needs the same captured value."""
        monkeypatch.delenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", raising=False)
        monkeypatch.setattr(train_specs, "default_fp8_block_scaling_fp32_scales", lambda: "0")

        specs = specs_trainer(_make_args(use_critic=True))
        monkeypatch.setattr(train_specs, "default_fp8_block_scaling_fp32_scales", lambda: "1")

        assert [spec.env_var(_make_context())["NVTE_FP8_BLOCK_SCALING_FP32_SCALES"] for spec in specs] == ["0", "0"]

    def test_disk_offload_forwards_backend_flags_and_nondefault_chunk_size(self, monkeypatch):
        """The disk backend must be switched on in place of the cpu one and use the requested chunk size."""
        _install_fake_torch_memory_saver(monkeypatch, MagicMock(return_value=Path("/opt/tms.so")))
        args = _make_args(offload_train=True, offload_train_target="disk", offload_train_disk_chunk_mb=128)

        (spec,) = specs_trainer(args)
        env_vars = spec.env_var(_make_context())

        assert (
            env_vars["TMS_INIT_ENABLE_CPU_BACKUP"],
            env_vars["TMS_INIT_ENABLE_DISK_BACKUP"],
            env_vars["TMS_DISK_BACKUP_CHUNK_MB"],
        ) == ("0", "1", "128")

    def test_disk_offload_gets_a_directory_per_worker(self, monkeypatch):
        """Two ranks sharing one directory would overwrite each other's offloaded weights."""
        _install_fake_torch_memory_saver(monkeypatch, MagicMock(return_value=Path("/opt/tms.so")))
        args = _make_args(offload_train=True, offload_train_target="disk")

        (spec,) = specs_trainer(args)

        directories = [
            spec.env_var(_make_context(cell_index=1, worker_in_cell_index=i))["TMS_DISK_BACKUP_DIR"] for i in range(2)
        ]
        assert directories == ["/tmp/offload/cell1_rank0", "/tmp/offload/cell1_rank1"]

    def test_a_library_without_the_disk_backend_is_rejected(self, monkeypatch):
        """Launching disk offload against a library that cannot write to disk would
        silently lose the offloaded weights."""
        _install_fake_torch_memory_saver(monkeypatch, MagicMock(return_value=Path("/opt/tms.so")))
        monkeypatch.setattr(Path, "read_bytes", lambda self: b"built without the disk backend")

        (spec,) = specs_trainer(_make_args(offload_train=True, offload_train_target="disk"))

        with pytest.raises(AssertionError, match="has no disk backend"):
            spec.env_var(_make_context())

    def test_no_disk_directory_without_disk_offload(self):
        """The cpu backup path must not be told to write to disk."""
        (spec,) = specs_trainer(_make_args(offload_train=False))

        assert "TMS_DISK_BACKUP_DIR" not in spec.env_var(_make_context())


class TestTorchMemorySaverPreload:
    def test_the_preload_library_is_resolved_from_the_package(self, monkeypatch):
        """The hook must be preloaded from the installed package, not a hardcoded path."""
        expected_path = Path("/opt/torch_memory_saver_hook_mode_preload_cu13.abi3.so")
        get_binary_path = _install_fake_torch_memory_saver(monkeypatch, MagicMock(return_value=expected_path))

        (spec,) = specs_trainer(_make_args(offload_train=True, offload_train_target="cpu"))
        env_vars = spec.env_var(_make_context())

        get_binary_path.assert_called_once_with("torch_memory_saver_hook_mode_preload")
        assert env_vars["LD_PRELOAD"] == str(expected_path)
        assert env_vars["TMS_INIT_ENABLE"] == "1"
        assert env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] == "1"

    def test_fsdp_offload_does_not_enable_the_megatron_preload(self, monkeypatch):
        """fsdp has its own offload implementation, so preloading the hook only breaks its allocator."""
        original_import = builtins.__import__

        def reject_torch_memory_saver_import(
            name: str,
            globals_: dict[str, object] | None = None,
            locals_: dict[str, object] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> ModuleType:
            if name.partition(".")[0] == "torch_memory_saver":
                raise AssertionError("FSDP offload must not import torch_memory_saver")
            return original_import(name, globals_, locals_, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", reject_torch_memory_saver_import)

        (spec,) = specs_trainer(_make_args(train_backend="fsdp", offload_train=True, offload_train_target="cpu"))
        env_vars = spec.env_var(_make_context())

        assert "LD_PRELOAD" not in env_vars
        assert "TMS_INIT_ENABLE" not in env_vars

    def test_a_missing_preload_library_is_not_swallowed(self, monkeypatch):
        """Silently launching without the hook would make offload corrupt weights."""
        _install_fake_torch_memory_saver(monkeypatch, MagicMock(side_effect=RuntimeError("missing preload library")))

        (spec,) = specs_trainer(_make_args(offload_train=True, offload_train_target="cpu"))

        with pytest.raises(RuntimeError, match="missing preload library"):
            spec.env_var(_make_context())


class TestPorts:
    def test_the_master_port_is_shared_across_the_cell(self):
        """All ranks of a cell rendezvous on one address, so it is a master port."""
        (spec,) = specs_trainer(_make_args())

        (master,) = [port for port in spec.port_infos if port.name == "master"]
        assert master.mode == "master"
        assert master.allow_dynamic is True


@pytest.mark.parametrize("role", ["actor", "critic"])
def test_the_pool_name_encodes_the_role(role):
    """Spec names identify trainer cells apart from inference cells."""
    assert compute_trainer_pool_id(role) == f"trainer-engine-{role}"


class _FakeStaticProvider:
    def __init__(self) -> None:
        self.handles: list[str] = []

    def get_handle(self, worker_name: str) -> object:
        self.handles.append(worker_name)
        return self


def _controller_layout() -> LaunchPlan:
    return LaunchPlan(
        run_id="260101-000000-000",
        release="miles-run-260101",
        namespace="rl",
        state_file="/cluster-storage/miles_data/miles-runs/run/state/orchestrator-260101-000000-000001.state",
        orchestrator_command=["python", "/repo/train.py"],
        worker_argv=["--actor-num-nodes", "1"],
    )


def _controller_context(capability: FakeBackendCapability) -> WorkerCtorContext:
    return WorkerCtorContext(cell_index=0, worker_in_cell_index=0, gpu_ids=[], capability=capability)


def _controller_providers() -> FakeBackendCapability:
    return FakeBackendCapability(
        cells_provider=object(), static_provider=_FakeStaticProvider(), cell_operations=object()
    )


class TestSpecTrainerController:
    def test_one_controller_per_trainer_role(self):
        """Each controller owns exactly one trainer pool, so a critic run needs a second one."""
        assert specs_trainer_controller(_make_args())[0].name == "trainer-controller-actor"
        assert specs_trainer_controller(_make_args(use_critic=True))[1].name == "trainer-controller-critic"

    def test_it_is_a_gpuless_worker_on_both_backends(self):
        """A gpu request would reserve a whole trainer slot for a process that only sends rpcs."""
        spec = specs_trainer_controller(_make_args())[0]

        assert (spec.scheduling.num_cells, spec.scheduling.num_workers_per_cell) == (1, 1)
        assert spec.scheduling.num_gpus_per_worker == 0
        assert spec.scheduling.num_gpu_slots_per_worker == 0

    def test_the_worker_class_is_the_controller_itself(self):
        """The spec names the class a pod or actor constructs, so it must be the real implementation."""
        spec = specs_trainer_controller(_make_args())[0]

        assert spec.worker_class == TRAINER_CONTROLLER_WORKER_CLASS

    def test_the_worker_and_cell_names_are_stable(self):
        """The driver looks the controller up by name, so these names are part of the release's contract."""
        assert trainer_controller_worker_name("actor") == "trainer-controller-actor-0-0"
        assert trainer_controller_cell_id("actor") == "trainer-controller-actor-0"

    def test_it_renders_into_static_workers_with_its_rpc_port(self):
        """The release has to contain the controller pod, or the address book would point at nothing."""
        spec = specs_trainer_controller(_make_args())[0]

        values = build_values([spec], _controller_layout()).as_values()

        (entry,) = values["run"]["staticWorkers"]
        assert SECTION_OF_CATEGORY[spec.category] == "staticWorkers"
        assert entry["name"] == "trainer-controller-actor"
        assert entry["ports"] == [{"name": "rpc", "port": 8000}]
        assert "--pool-id" in entry["command"]
        assert entry["command"][entry["command"].index("--pool-id") + 1] == "trainer-controller-actor"
        assert spec.worker_class == TRAINER_CONTROLLER_WORKER_CLASS
        assert "resources" not in entry

    def test_it_asks_for_a_provider_over_its_own_trainer_pool(self):
        """A controller that watched both pools would try to heal the other role's cells."""
        capability = _controller_providers()

        args = _make_args(use_critic=True)
        specs = specs_trainer_controller(args)
        kwargs = [spec.ctor_kwargs(_controller_context(capability)) for spec in specs]

        assert capability.requested_pool_ids == [["trainer-engine-actor"], ["trainer-engine-critic"]]
        assert [entry["cell_provider"] for entry in kwargs] == [capability.cells_provider] * 2

    def test_only_the_actor_controller_drives_the_inference_controller(self):
        """Weight updates flow from the actor; a critic asking for engines would fight it."""
        capability = _controller_providers()

        args = _make_args(use_critic=True)
        actor_spec, critic_spec = specs_trainer_controller(args)[0], specs_trainer_controller(args)[1]
        actor_kwargs = actor_spec.ctor_kwargs(_controller_context(capability))
        critic_kwargs = critic_spec.ctor_kwargs(_controller_context(capability))

        assert capability.requested_static_pool_ids == ["inference-controller"]
        assert actor_kwargs["inference_controller"] is capability.static_provider
        assert critic_kwargs["inference_controller"] is None

    def test_the_run_shape_flags_are_resolved_by_the_spec(self):
        """These are functions of args, so the worker can answer them from the argv it parses itself."""
        capability = _controller_providers()

        spec = specs_trainer_controller(_make_args(kl_coef=0.1, use_opd=True, opd_type="megatron"))[0]
        kwargs = spec.ctor_kwargs(_controller_context(capability))

        assert (kwargs["trainer_id"], kwargs["with_ref"], kwargs["with_opd_teacher"]) == ("actor", True, True)

    def test_a_policy_that_switches_off_its_kl_loss_gets_no_reference_cells(self, tmp_path):
        """with_ref is read off that trainer's own args, so one policy may need reference cells while another does not."""
        args = _make_args(
            use_kl_loss=True,
            megatron_config=write_megatron_config_trainers(
                tmp_path, [{"model_id": "a", "overrides": {"use_kl_loss": False}}, {"model_id": "b"}]
            ),
        )

        spec_a, spec_b = specs_trainer_controller(args)

        assert spec_a.ctor_kwargs(_controller_context(_controller_providers()))["with_ref"] is False
        assert spec_b.ctor_kwargs(_controller_context(_controller_providers()))["with_ref"] is True

    def test_the_critic_controller_gets_no_reference_or_teacher_cells(self):
        """A critic controller must not hand its cells the actor's KL and OPD settings."""
        spec = specs_trainer_controller(_make_args(use_critic=True, kl_coef=0.1, use_kl_loss=True, use_opd=True))[1]
        critic_kwargs = spec.ctor_kwargs(_controller_context(_controller_providers()))

        assert (critic_kwargs["with_ref"], critic_kwargs["with_opd_teacher"]) == (False, False)

    def test_no_args_are_frozen_into_the_controller_at_spec_time(self):
        """The spec is built before the driver finishes deriving args, so a captured copy would be stale."""
        actor_kwargs = specs_trainer_controller(_make_args())[0].ctor_kwargs(
            _controller_context(_controller_providers())
        )

        assert "args" not in actor_kwargs

    def test_the_controller_pool_name_encodes_the_role(self):
        """The two controllers of a critic run must not collide in the address book."""
        assert compute_trainer_controller_pool_id("critic") == "trainer-controller-critic"


class TestTrainerConfigs:
    def test_a_single_policy_run_names_its_trainers_actor_and_critic(self):
        """Every existing pool name, worker name and checkpoint path is written against these two ids."""
        configs = compute_trainer_configs(_make_args(use_critic=True))

        assert [config.trainer_id for config in configs] == ["actor", "critic"]
        assert [config.model_id for config in configs] == [None, None]
        assert [config.role for config in configs] == ["actor", "critic"]

    def test_every_policy_of_a_multi_policy_run_gets_a_trainer_of_its_own(self, tmp_path):
        """One TrainerController per trainer id is what keeps two policies from sharing cells."""
        args = _make_args(megatron_config=write_megatron_config(tmp_path, "alpha", "beta"))

        configs = compute_trainer_configs(args)

        assert [config.trainer_id for config in configs] == ["alpha-actor", "beta-actor"]
        assert [config.model_id for config in configs] == ["alpha", "beta"]
        assert [config.role for config in configs] == ["actor", "actor"]

    def test_the_pool_ids_of_two_policies_do_not_collide(self, tmp_path):
        """Pool ids become Kubernetes names, so a collision would put both policies in one pool."""
        args = _make_args(megatron_config=write_megatron_config(tmp_path, "alpha", "beta"))

        specs = specs_trainer(args)

        assert [spec.name for spec in specs] == ["trainer-engine-alpha-actor", "trainer-engine-beta-actor"]

    def test_each_policy_gets_its_own_slice_of_the_placement_group(self, tmp_path):
        """Two policies sharing the same slots would be scheduled onto the same gpus."""
        args = _make_args(megatron_config=write_megatron_config(tmp_path, "alpha", "beta"))

        specs = specs_trainer(args)

        assert [spec.scheduling.pg_slot_offset for spec in specs] == [0, 4]

    def test_the_slice_of_each_policy_follows_the_trainer_size(self, tmp_path):
        """The stride is one trainer's gpu count; a hardcoded one overlaps as soon as a policy grows."""
        args = _make_args(
            megatron_config=write_megatron_config(tmp_path, "alpha", "beta", "gamma"),
            actor_num_nodes=2,
            actor_num_gpus_per_node=4,
        )

        specs = specs_trainer(args)

        assert [spec.scheduling.pg_slot_offset for spec in specs] == [0, 8, 16]

    def test_a_critic_shares_the_first_slice_with_its_actor(self):
        """Actor and critic sit in one placement group on purpose; changing that must be deliberate."""
        specs = specs_trainer(_make_args(use_critic=True))

        assert [(spec.name, spec.scheduling.pg_slot_offset) for spec in specs] == [
            ("trainer-engine-actor", 0),
            ("trainer-engine-critic", 0),
        ]

    def test_every_slice_fits_inside_the_placement_group(self, tmp_path):
        """A slot past the end of the group is a pending bundle nobody ever schedules."""
        megatron_config = write_megatron_config(tmp_path, "alpha", "beta", "gamma")
        args = _make_args(megatron_config=megatron_config, actor_num_nodes=2, actor_num_gpus_per_node=4)
        _, actor_num_gpus = _get_placement_group_layout(
            SimpleNamespace(
                actor_num_nodes=2,
                actor_num_gpus_per_node=4,
                rollout_num_gpus=8,
                eval_num_gpus=0,
                debug_train_only=False,
                debug_rollout_only=False,
                rollout_external=False,
                colocate=False,
                use_critic=False,
                megatron_config=megatron_config,
            )
        )

        specs = specs_trainer(args)

        for spec in specs:
            reserved = spec.scheduling.num_cells * spec.scheduling.gpus_per_cell()
            assert spec.scheduling.pg_slot_offset + reserved <= actor_num_gpus

    def test_each_policy_controller_drives_the_inference_controller(self, tmp_path):
        """A policy that cannot reach the engines can never publish its weights."""
        args = _make_args(megatron_config=write_megatron_config(tmp_path, "alpha", "beta"))

        specs = specs_trainer_controller(args)
        kwargs = [spec.ctor_kwargs(_controller_context(_controller_providers())) for spec in specs]

        assert [entry["trainer_id"] for entry in kwargs] == ["alpha-actor", "beta-actor"]
        assert all(entry["inference_controller"] is not None for entry in kwargs)
        assert [entry["role"] for entry in kwargs] == ["actor", "actor"]

    def test_a_worker_is_told_which_policy_it_serves_through_its_args(self, tmp_path):
        """The worker namespaces its logs and metrics by this id; a shared one merges the two runs."""
        args = _make_args(megatron_config=write_megatron_config(tmp_path, "alpha", "beta"))

        kwargs = [spec.ctor_kwargs(_make_context()) for spec in specs_trainer(args)]

        assert [entry["args"].trainer_model_id for entry in kwargs] == ["alpha", "beta"]
        assert [entry["role"] for entry in kwargs] == ["actor", "actor"]

    def test_a_worker_of_every_policy_still_gets_the_plain_actor_role(self, tmp_path):
        """The worker layer branches on the role literal, so a policy specific role would break training."""
        args = _make_args(megatron_config=write_megatron_config(tmp_path, "alpha", "beta"))

        [kwargs] = [spec.ctor_kwargs(_make_context()) for spec in specs_trainer(_make_args())]

        assert (kwargs["args"].trainer_model_id, kwargs["role"]) == (None, "actor")
        assert [spec.name for spec in specs_trainer(args)] == [
            "trainer-engine-alpha-actor",
            "trainer-engine-beta-actor",
        ]

    def test_a_policy_overlays_its_own_megatron_args_onto_its_worker(self, tmp_path):
        """A per-policy --lr that never reaches the worker would train both policies identically."""
        args = _make_args(
            megatron_config=write_megatron_config_trainers(
                tmp_path, [{"model_id": "alpha", "overrides": {"lr": 5e-7}}, {"model_id": "beta"}]
            ),
        )

        kwargs = [spec.ctor_kwargs(_make_context()) for spec in specs_trainer(args)]

        assert [entry["args"].lr for entry in kwargs] == [5e-7, 1e-6]

    def test_the_critic_of_a_configured_policy_is_a_trainer_of_that_policy(self, tmp_path):
        """A critic left without a model id would be built from raw args, ignoring the policy overlay."""
        args = _make_args(
            use_critic=True,
            megatron_config=write_megatron_config_trainers(
                tmp_path, [{"model_id": "alpha", "overrides": {"lr": 5e-7}}]
            ),
        )

        configs = compute_trainer_configs(args)

        assert [(config.trainer_id, config.role, config.model_id) for config in configs] == [
            ("alpha-actor", "actor", "alpha"),
            ("alpha-critic", "critic", "alpha"),
        ]

    def test_the_critic_args_stack_the_policy_overlay_and_the_critic_neutralization(self, tmp_path):
        """The two transforms are not exclusive: the critic of a policy needs that policy's overlay too."""
        args = _make_args(
            use_critic=True,
            kl_coef=0.1,
            use_opd=True,
            critic_lr=2e-6,
            megatron_config=write_megatron_config_trainers(
                tmp_path, [{"model_id": "alpha", "overrides": {"eps_clip": 0.3, "lr": 5e-7}}]
            ),
        )

        critic_args = compute_trainer_args(args, compute_trainer_configs(args)[1])

        assert (critic_args.eps_clip, critic_args.kl_coef, critic_args.use_opd) == (0.3, 0, False)
        assert critic_args.lr == 2e-6

    def test_the_critic_pool_of_a_configured_policy_is_named_after_its_trainer_id(self, tmp_path):
        """The critic pool must not collide with the policy's own pool nor with a plain 'critic' pool."""
        args = _make_args(use_critic=True, megatron_config=write_megatron_config(tmp_path, "alpha"))

        assert [spec.name for spec in specs_trainer(args)] == [
            "trainer-engine-alpha-actor",
            "trainer-engine-alpha-critic",
        ]
        assert [spec.name for spec in specs_trainer_controller(args)] == [
            "trainer-controller-alpha-actor",
            "trainer-controller-alpha-critic",
        ]
