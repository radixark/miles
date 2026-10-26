import asyncio
from pathlib import Path

import pytest
from tests.fast.ray.rollout.checkpoint_rpc_fixtures import serve_checkpoint_worker

from miles.utils.simple_checkpointer import load_simple_checkpoint
from miles.utils.workers.rpc.common.protocol import HEALTH_PATH, IN_FLIGHT_PATH


class TestResponsiveCheckpointRpc:
    @pytest.mark.parametrize("method", ["save", "load"])
    async def test_slow_checkpoint_keeps_http_responsive_and_producer_atomic(
        self, tmp_path: Path, method: str
    ) -> None:
        """Slow real checkpoint methods must not block HTTP or interleave with the producer."""
        async with serve_checkpoint_worker(tmp_path) as rpc:
            source = rpc.worker.data_source
            if method == "load":
                source.release.set()
                await rpc.handle.save(rollout_id=2)
                source.release.clear()
                source.entered.clear()

            try:
                async with asyncio.timeout(3.0):
                    kwargs = {"load": str(tmp_path)} if method == "load" else {}
                    await rpc.handle.submit_without_result(method, rollout_id=2, **kwargs)
                    assert await asyncio.to_thread(source.entered.wait, 2.0)
                    rpc.worker.loop.call_soon_threadsafe(rpc.worker.produce.set)
                    assert (await rpc.client.get(HEALTH_PATH)).json() == {"status": "ok"}
                    [call_id] = (await rpc.client.get(IN_FLIGHT_PATH)).json()["call_ids"]
                    pending = await rpc.client.get(f"/v1/calls/{call_id}", params={"timeout": 0.0})
                    assert pending.json()["status"] == "pending"
                    assert not rpc.worker.produced.is_set()
            finally:
                source.release.set()

            completed = await rpc.client.get(f"/v1/calls/{call_id}", params={"timeout": 1.0})
            assert completed.json()["status"] == "success", completed.text
            assert await asyncio.to_thread(rpc.worker.produced.wait, 2.0)
            assert source.tokens == [1, 2]
            assert load_simple_checkpoint(tmp_path / "rollout/2/data_source") == [1]

        assert rpc.worker.cancelled.is_set()
        assert rpc.worker.loop.is_closed()
