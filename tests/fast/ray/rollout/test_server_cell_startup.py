from __future__ import annotations

from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

import miles.ray.rollout.server_cell as server_cell_module
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.utils.workers.worker_spec import HostAndPort


def _startup_meta(*, needs_offload: bool, update_weights: bool) -> ServerCellMetadata:
    return ServerCellMetadata(
        model_id="default",
        worker_type="regular",
        cell_id="cell-0",
        num_gpus_per_engine=1,
        gpu_offset=0,
        sglang_api_key=None,
        worker_name="worker-0",
        needs_offload=needs_offload,
        update_weights=update_weights,
        workers_hash="hash-0",
    )


async def _run_add(
    *,
    needs_offload: bool = True,
    update_weights: bool = True,
    check_weight_update_equal: bool = True,
) -> list[str]:
    args = Namespace(
        rollout_external=False,
        use_miles_router=False,
        check_weight_update_equal=check_weight_update_equal,
        check_weight_update_skip_list=[],
    )
    router_api_client = MagicMock()
    router_api_client.add_worker = AsyncMock()
    cell = ServerCell(
        args=args,
        meta=_startup_meta(needs_offload=needs_offload, update_weights=update_weights),
        router_api_client=router_api_client,
    )

    calls: list[str] = []
    client = MagicMock()
    client.check_weights = AsyncMock(side_effect=lambda **kwargs: calls.append(f"check_weights:{kwargs['action']}"))
    client.release_memory_occupation = AsyncMock(side_effect=lambda **kwargs: calls.append("release"))
    client.resume_memory_occupation = AsyncMock(side_effect=lambda **kwargs: calls.append("resume"))

    with (
        patch.object(ServerCell, "api_client", property(lambda self: client)),
        patch.object(server_cell_module, "RayWorkerProvider", _FakeProviderFactory()),
        patch.object(server_cell_module, "wait_server_healthy", new=AsyncMock()),
    ):
        await cell.add()

    assert cell.is_alive
    router_api_client.add_worker.assert_awaited_once()
    return calls


class _FakeProviderFactory:
    def create(self):
        provider = MagicMock()
        provider.get_addrs = AsyncMock(return_value={"primary": HostAndPort(host="10.0.0.1", port=30000)})
        return provider


class TestServerCellStartupSequence:
    async def test_the_weight_baseline_is_snapshotted_before_memory_is_released(self):
        """The checker's baseline must be the freshly loaded checkpoint, not the remapped weight storage."""
        assert await _run_add() == ["check_weights:snapshot", "release", "resume"]

    async def test_a_colocated_cell_releases_all_memory_and_resumes_only_the_weights(self):
        """Startup memory handover for an offloading cell lives on the cell, not on the controller."""
        assert await _run_add(check_weight_update_equal=False) == ["release", "resume"]

    async def test_a_resident_cell_touches_neither_memory_nor_the_checker(self):
        """A cell that never offloads keeps its memory, so startup issues no engine calls."""
        assert await _run_add(needs_offload=False, check_weight_update_equal=False) == []

    async def test_a_resident_cell_still_snapshots_its_weight_baseline(self):
        """The baseline is about the checkpoint, so it is taken whether or not the cell offloads."""
        assert await _run_add(needs_offload=False) == ["check_weights:snapshot"]

    async def test_a_frozen_model_gets_no_baseline(self):
        """A model that never receives weight updates would always mismatch the checker."""
        assert await _run_add(update_weights=False) == ["release", "resume"]

    async def test_the_snapshot_is_taken_over_the_whole_model_without_a_skip_list(self):
        """The baseline must match what the controller-side reset and comparison later cover."""
        calls: list[dict] = []
        args = Namespace(
            rollout_external=False,
            use_miles_router=False,
            check_weight_update_equal=True,
            check_weight_update_skip_list=["lm_head"],
        )
        router_api_client = MagicMock()
        router_api_client.add_worker = AsyncMock()
        cell = ServerCell(
            args=args,
            meta=_startup_meta(needs_offload=True, update_weights=True),
            router_api_client=router_api_client,
        )

        client = MagicMock()
        client.check_weights = AsyncMock(side_effect=lambda **kwargs: calls.append(kwargs))
        client.release_memory_occupation = AsyncMock()
        client.resume_memory_occupation = AsyncMock()

        with (
            patch.object(ServerCell, "api_client", property(lambda self: client)),
            patch.object(server_cell_module, "RayWorkerProvider", _FakeProviderFactory()),
            patch.object(server_cell_module, "wait_server_healthy", new=AsyncMock()),
        ):
            await cell.add()

        assert calls == [{"action": "snapshot", "allow_quant_error": False, "selector": "all", "skip_list": None}]
