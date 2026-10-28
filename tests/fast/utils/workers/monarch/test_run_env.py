import pytest

pytest.importorskip("monarch")

from miles.utils.workers.env_vars import WORKER_GPU_IDS_ENV_VAR
from miles.utils.workers.monarch.runtime import _compute_env
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
    def test_a_gpu_worker_learns_the_physical_devices_it_was_placed_on(self):
        """It sees the whole host, so without them every worker of a host would pick device 0."""
        env = _compute_env(_launch(env_vars={"NCCL_DEBUG": "WARN"}, gpu=True), gpu_ids=[4, 5])

        assert env == {"NCCL_DEBUG": "WARN", WORKER_GPU_IDS_ENV_VAR: "4,5"}

    def test_a_cpu_worker_still_sees_no_gpu(self):
        env = _compute_env(_launch(env_vars={}, gpu=False), gpu_ids=None)

        assert env["CUDA_VISIBLE_DEVICES"] == ""
