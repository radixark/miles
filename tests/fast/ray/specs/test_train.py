import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from miles.ray.specs import train as train_specs
from miles.ray.specs.train import compute_trainer_pool_id, specs_trainer
from miles.utils.workers.worker_spec import WorkerLaunchContext


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
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _make_context(**overrides) -> WorkerLaunchContext:
    kwargs = dict(cell_index=0, worker_in_cell_index=0, gpu_ids=[0])
    kwargs.update(overrides)
    return WorkerLaunchContext(**kwargs)


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

    def test_a_worker_reserves_a_fraction_of_its_gpu(self):
        """The rollout engine shares the same GPU slot, so the trainer must not claim it whole."""
        (spec,) = specs_trainer(_make_args())

        assert spec.scheduling.num_gpus_per_worker == 0.4
        assert spec.scheduling.num_gpu_slots_per_worker == 1


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

    def test_the_backend_selects_the_worker_class(self):
        """A run must not start Megatron workers for an fsdp job."""
        (megatron_spec,) = specs_trainer(_make_args(train_backend="megatron"))
        (fsdp_spec,) = specs_trainer(_make_args(train_backend="fsdp"))

        assert megatron_spec.worker_class.endswith("MegatronTrainRayActor")
        assert fsdp_spec.worker_class.endswith("FSDPTrainRayActor")


class TestConcurrencyGroups:
    def test_fault_tolerance_isolates_the_heartbeat_rpc(self):
        """A heartbeat queued behind a train step reads as a dead cell."""
        (spec,) = specs_trainer(_make_args(use_fault_tolerance=True))

        assert spec.concurrency_groups == {"heartbeat_status": 1, "default": 1, "fault_injector": 1}

    def test_no_groups_without_fault_tolerance(self):
        """Runs without fault tolerance keep ray's default scheduling."""
        (spec,) = specs_trainer(_make_args())

        assert spec.concurrency_groups is None


class TestEnvironmentVariables:
    def test_user_env_vars_are_forwarded(self):
        """--train-env-vars must reach the worker process."""
        (spec,) = specs_trainer(_make_args(train_env_vars={"MY_VAR": "1"}))

        assert spec.env_var(_make_context())["MY_VAR"] == "1"

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

    def test_disk_offload_gets_a_directory_per_worker(self):
        """Two ranks sharing one directory would overwrite each other's offloaded weights."""
        args = _make_args(offload_train=True, offload_train_target="disk")

        (spec,) = specs_trainer(args)

        directories = [
            spec.env_var(_make_context(cell_index=1, worker_in_cell_index=i))["TMS_DISK_BACKUP_DIR"] for i in range(2)
        ]
        assert directories == ["/tmp/offload/cell1_rank0", "/tmp/offload/cell1_rank1"]

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
    assert compute_trainer_pool_id(role) == f"trainer-{role}"
