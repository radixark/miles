import pytest

pytest.importorskip("monarch")

from miles.utils.workers.monarch import runtime
from miles.utils.workers.monarch.runtime import (
    GpuLayoutRequest,
    GpuSlot,
    ProbedHost,
    compute_monarch_gpu_layout,
    visible_gpu_ids,
)


def _request(
    *, num_gpus: int, rollout_offset: int = 0, use_critic: bool = False, num_gpus_per_node: int = 4
) -> GpuLayoutRequest:
    return GpuLayoutRequest(
        num_gpus=num_gpus, rollout_offset=rollout_offset, use_critic=use_critic, num_gpus_per_node=num_gpus_per_node
    )


def _hosts(*ips: str, gpu_ids: list[int] | None = None) -> list[ProbedHost]:
    return [ProbedHost(ip=ip, gpu_ids=gpu_ids if gpu_ids is not None else [0, 1, 2, 3]) for ip in ips]


class TestComputeMonarchGpuLayout:
    def test_slots_pack_hosts_in_mesh_order(self):
        """Like a PACK placement group: the first host fills up before the second gets a slot."""
        layout = compute_monarch_gpu_layout(request=_request(num_gpus=6), hosts=_hosts("10.0.0.2", "10.0.0.1"))

        assert layout.slots_of_layout["actor"] == [
            GpuSlot(host_rank=0, gpu_id=0),
            GpuSlot(host_rank=0, gpu_id=1),
            GpuSlot(host_rank=0, gpu_id=2),
            GpuSlot(host_rank=0, gpu_id=3),
            GpuSlot(host_rank=1, gpu_id=0),
            GpuSlot(host_rank=1, gpu_id=1),
        ]

    def test_rollout_slots_start_at_the_rollout_offset(self):
        """A disaggregated run gives the engines the slots after the trainer's, as the rollout placement group does."""
        layout = compute_monarch_gpu_layout(request=_request(num_gpus=8, rollout_offset=4), hosts=_hosts("a", "b"))

        assert [slot.host_rank for slot in layout.slots_of_layout["rollout"]] == [1, 1, 1, 1]

    def test_colocated_rollout_shares_the_trainer_slots(self):
        layout = compute_monarch_gpu_layout(request=_request(num_gpus=2), hosts=_hosts("a"))

        assert layout.slots_of_layout["rollout"] == layout.slots_of_layout["actor"]

    def test_a_critic_shares_the_actor_slots(self):
        layout = compute_monarch_gpu_layout(request=_request(num_gpus=2, use_critic=True), hosts=_hosts("a"))

        assert layout.slots_of_layout["critic"] == layout.slots_of_layout["actor"]

    def test_more_gpus_than_the_hosts_offer_is_refused(self):
        with pytest.raises(AssertionError, match="offer 4 each"):
            compute_monarch_gpu_layout(request=_request(num_gpus=5), hosts=_hosts("a"))

    def test_slots_carry_the_physical_ids_the_host_reports(self):
        """Workers map their gpu ids back through CUDA_VISIBLE_DEVICES, so ids must be physical, not indices."""
        layout = compute_monarch_gpu_layout(request=_request(num_gpus=4), hosts=_hosts("a", gpu_ids=[2, 3, 4, 5]))

        assert [slot.gpu_id for slot in layout.slots_of_layout["actor"]] == [2, 3, 4, 5]

    def test_a_host_offers_no_more_gpus_than_the_run_uses_per_node(self):
        """--num-gpus-per-node caps a host, as the ray placement group asks each node for that many gpus."""
        layout = compute_monarch_gpu_layout(
            request=_request(num_gpus=4, num_gpus_per_node=2), hosts=_hosts("a", "b", gpu_ids=[0, 1, 2, 3])
        )

        assert [(slot.host_rank, slot.gpu_id) for slot in layout.slots_of_layout["actor"]] == [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]


class TestVisibleGpuIds:
    def test_the_visible_devices_list_names_the_physical_ids(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3,4,5")

        assert visible_gpu_ids() == [2, 3, 4, 5]

    def test_gpus_named_by_uuid_are_refused(self, monkeypatch: pytest.MonkeyPatch):
        """The layout places workers by device index, which a uuid does not give."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-8d1f0c2e")

        with pytest.raises(AssertionError, match="not by uuid"):
            visible_gpu_ids()

    def test_an_empty_visible_devices_list_hides_every_gpu(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

        assert visible_gpu_ids() == []

    def test_the_probe_reads_the_list_the_workers_remap_through(self, monkeypatch: pytest.MonkeyPatch):
        """The first list that names any gpu, as to_local_gpu_id reads it, so both see the same devices."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")

        assert visible_gpu_ids() == [0, 1]

    def test_without_a_list_every_device_node_is_a_gpu(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
        monkeypatch.setattr(
            runtime.os, "listdir", lambda _: ["nvidia1", "nvidiactl", "nvidia0", "loop7", "nvidia-uvm"]
        )

        assert visible_gpu_ids() == [0, 1]
