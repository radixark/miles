import pytest

pytest.importorskip("monarch")

from miles.utils.workers.env_vars import WORKER_GPU_IDS_ENV_VAR
from miles.utils.workers.monarch.env_vars import MONARCH_RUN_ENV_VARS_ENV_VAR
from miles.utils.workers.monarch.runtime import _compute_env, read_run_env
from miles.utils.workers.worker_manager.runtime import GpuSlotRef, WorkerLaunchRequest


def _launch(*, env_vars: dict[str, str], gpu: bool) -> WorkerLaunchRequest:
    return WorkerLaunchRequest(
        pool_id="actor",
        name="actor-0-0",
        actor_class=object,
        ctor_kwargs={},
        env_vars=env_vars,
        num_cpus=1,
        num_gpus=1 if gpu else 0,
        gpu_slots=GpuSlotRef(layout_name="actor", first_slot_index=0, num_slots=1) if gpu else None,
        pin_to_head=False,
    )


class TestTheRunEnvOfAWorker:
    def test_the_driver_reads_the_variables_the_launcher_named(self, monkeypatch: pytest.MonkeyPatch):
        """Only the launcher's variables travel; the rest of the driver's environment is the head node's own."""
        monkeypatch.setenv(MONARCH_RUN_ENV_VARS_ENV_VAR, "PYTHONPATH,NCCL_DEBUG,UNSET_ONE")
        monkeypatch.setenv("PYTHONPATH", "/shared/miles:/shared/Megatron-LM")
        monkeypatch.setenv("NCCL_DEBUG", "INFO")
        monkeypatch.delenv("UNSET_ONE", raising=False)

        assert read_run_env() == {"PYTHONPATH": "/shared/miles:/shared/Megatron-LM", "NCCL_DEBUG": "INFO"}

    def test_a_driver_the_launcher_named_nothing_for_passes_nothing_on(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv(MONARCH_RUN_ENV_VARS_ENV_VAR, raising=False)

        assert read_run_env() == {}

    def test_every_worker_gets_the_run_env_under_its_own_variables(self):
        """A worker on another host would otherwise miss e.g. the Megatron PYTHONPATH; its own variables win."""
        run_env = {"PYTHONPATH": "/shared/miles", "NCCL_DEBUG": "INFO"}

        env = _compute_env(_launch(env_vars={"NCCL_DEBUG": "WARN"}, gpu=True), run_env=run_env, gpu_ids=[3])

        assert env == {"PYTHONPATH": "/shared/miles", "NCCL_DEBUG": "WARN", WORKER_GPU_IDS_ENV_VAR: "3"}

    def test_a_gpu_worker_learns_the_physical_devices_it_was_placed_on(self):
        """It sees the whole host, so without them every worker of a host would pick device 0."""
        env = _compute_env(_launch(env_vars={}, gpu=True), run_env={}, gpu_ids=[4, 5])

        assert env[WORKER_GPU_IDS_ENV_VAR] == "4,5"

    def test_a_cpu_worker_still_sees_no_gpu(self):
        env = _compute_env(_launch(env_vars={}, gpu=False), run_env={"CUDA_VISIBLE_DEVICES": "0,1"}, gpu_ids=None)

        assert env["CUDA_VISIBLE_DEVICES"] == ""
