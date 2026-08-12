import asyncio
from argparse import Namespace
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast import (
    connect_rollout_engines_from_distributed,
)
from miles.ray.rollout import external_engine_provider as external_engine_provider_module
from miles.ray.rollout.external_engine_provider import StaticInferenceEngineWorkerProvider
from miles.ray.rollout.inference_controller import _compute_server_cell_meta_from_info
from miles.ray.rollout.rollout_server import RolloutServer
from miles.utils.context_lock import ContextLock

_BROADCAST_MODULE = "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast"


def _payload(*, num_gpus: int, disaggregation_mode: str) -> dict[str, Any]:
    return dict(
        internal_states=[dict(world_size=num_gpus)],
        disaggregation_mode=disaggregation_mode,
        disaggregation_bootstrap_port=12090 if disaggregation_mode == "prefill" else None,
    )


class _RecordingEngine:
    def __init__(self, calls: list[dict]) -> None:
        self._calls = calls

    def init_weights_update_group(self, master_address, master_port, rank, world_size, group_name, backend):
        self._calls.append(dict(rank=rank, world_size=world_size))
        return None


async def _discovered_server(monkeypatch, *, payloads: dict[str, dict[str, Any]], urls: list[str]) -> RolloutServer:
    async def _fetch(url: str) -> dict[str, Any]:
        return payloads[url]

    monkeypatch.setattr(external_engine_provider_module, "_fetch_server_info_with_retry", _fetch)
    args = make_args(
        rollout_external=True,
        rollout_external_engine_addrs=urls,
        rollout_external_router_pd=any(p["disaggregation_mode"] != "null" for p in payloads.values()),
        rollout_num_gpus=sum(p["internal_states"][0]["world_size"] for p in payloads.values()),
    )
    provider = StaticInferenceEngineWorkerProvider(args=args)
    await provider.init()

    cells = {}
    for info in provider.cell_infos:
        meta = _compute_server_cell_meta_from_info(info)
        cells[info.cell_id] = SimpleNamespace(meta=meta, api_client=f"client-{meta.gpu_offset}")
    return RolloutServer(
        server_cells=cells,
        args=args,
        context_lock=ContextLock("InferenceController"),
        engine_provider=provider,
    )


def _connect(*, engine_gpu_counts: list[int], rollout_num_gpus_per_engine: int) -> list[dict]:
    calls: list[dict] = []
    engines = [_RecordingEngine(calls) for _ in engine_gpu_counts]
    async_utils = SimpleNamespace(submit=lambda coro: coro, wait_futures=lambda futures: None)

    with (
        patch(f"{_BROADCAST_MODULE}.ray"),
        patch(f"{_BROADCAST_MODULE}.async_utils", async_utils),
        patch(f"{_BROADCAST_MODULE}.init_process_group"),
    ):
        connect_rollout_engines_from_distributed(
            Namespace(rollout_num_gpus_per_engine=rollout_num_gpus_per_engine),
            "miles-pp_0",
            engines,
            engine_gpu_counts=engine_gpu_counts,
        )
    return calls


class TestExternalPdFleetWeightUpdateLayout:
    @pytest.mark.asyncio
    async def test_the_discovered_gpu_counts_reach_the_update_group_unchanged(self, monkeypatch):
        """Every layer knows how to carry per-engine gpu counts, so the only way an external PD fleet
        hangs in the rendezvous is one layer quietly substituting the uniform argument. Prove the whole
        chain with one heterogeneous fleet instead of one value per layer."""
        urls = ["prefill:8000", "decode:8000"]
        payloads = {
            "http://prefill:8000": _payload(num_gpus=2, disaggregation_mode="prefill"),
            "http://decode:8000": _payload(num_gpus=4, disaggregation_mode="decode"),
        }

        srv = await _discovered_server(monkeypatch, payloads=payloads, urls=urls)

        async with srv.context_lock:
            assert srv.engine_gpu_counts == [2, 4]
            assert srv.engine_gpu_offsets == [0, 2]
            assert srv.api_clients == ["client-0", "client-2"]
            counts = srv.engine_gpu_counts

        calls = _connect(engine_gpu_counts=counts, rollout_num_gpus_per_engine=1)

        assert [call["rank"] for call in calls] == [1, 3]
        assert {call["world_size"] for call in calls} == {7}

    @pytest.mark.asyncio
    async def test_the_slower_engine_does_not_take_the_earlier_rank_range(self, monkeypatch):
        """Discovery runs concurrently, so if cells were built as answers arrive the rank layout would
        depend on network timing and drift silently between runs."""
        urls = ["slow:8000", "fast:8000"]
        payloads = {
            "http://slow:8000": _payload(num_gpus=2, disaggregation_mode="prefill"),
            "http://fast:8000": _payload(num_gpus=4, disaggregation_mode="decode"),
        }
        answered: list[str] = []

        async def _fetch(url: str) -> dict[str, Any]:
            while url == "http://slow:8000" and not answered:
                await asyncio.sleep(0)
            answered.append(url)
            return payloads[url]

        monkeypatch.setattr(external_engine_provider_module, "_fetch_server_info_with_retry", _fetch)
        args = make_args(
            rollout_external=True,
            rollout_external_engine_addrs=urls,
            rollout_external_router_pd=True,
            rollout_num_gpus=6,
        )
        provider = StaticInferenceEngineWorkerProvider(args=args)
        await provider.init()

        assert answered == ["http://fast:8000", "http://slow:8000"]
        assert [info.meta["num_gpus_per_engine"] for info in provider.cell_infos] == [2, 4]
        assert [info.meta["gpu_offset"] for info in provider.cell_infos] == [0, 2]


class TestExternalRegularEngineWeightUpdateLayout:
    @pytest.mark.asyncio
    async def test_one_multi_rank_engine_claims_every_rank_it_reported(self, monkeypatch):
        """The deleted external e2e ran a single multi-gpu engine, which is the only shape where the
        per-engine count and the fleet total disagree without any heterogeneity."""
        urls = ["host1:8000"]
        payloads = {"http://host1:8000": _payload(num_gpus=2, disaggregation_mode="null")}

        srv = await _discovered_server(monkeypatch, payloads=payloads, urls=urls)

        async with srv.context_lock:
            counts = srv.engine_gpu_counts
        assert counts == [2]

        calls = _connect(engine_gpu_counts=counts, rollout_num_gpus_per_engine=1)

        assert [call["rank"] for call in calls] == [1]
        assert {call["world_size"] for call in calls} == {3}
