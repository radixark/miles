"""Controller memory leak test.

Runs 500 ticks with realistic data injection and asserts RSS growth < 50MB.
Validates that MiniPrometheus ring buffer, MiniWandb, and detector chain
don't accumulate memory over time.
"""

from __future__ import annotations

import gc

import psutil
from tests.fast.utils.ft.conftest import ControllerTestHarness, inject_healthy_node, make_test_controller

from miles.utils.ft.controller.detectors import build_detector_chain

_WARMUP_TICKS = 10
_TEST_TICKS = 500
_MAX_RSS_GROWTH_BYTES = 50 * 1024 * 1024  # 50 MB

_NODE_IDS = ["node-0", "node-1", "node-2", "node-3"]


async def _register_n_nodes(
    controller: object,
    run_id: str,
    node_ids: list[str] = _NODE_IDS,
) -> None:
    world_size = len(node_ids)
    for rank, node_id in enumerate(node_ids):
        await controller.register_rank(
            run_id=run_id,
            rank=rank,
            world_size=world_size,
            node_id=node_id,
            exporter_address=f"http://{node_id}:9090",
        )


class TestControllerMemoryLeak:
    async def test_controller_main_loop_no_memory_leak(self) -> None:
        harness = make_test_controller(
            detectors=build_detector_chain(),
        )
        controller = harness.controller

        run_id = "leak-test-run"
        await _register_n_nodes(controller, run_id=run_id)

        for node_id in _NODE_IDS:
            inject_healthy_node(harness.metric_store, node_id=node_id)

        # Warm-up: establish steady state
        for i in range(_WARMUP_TICKS):
            self._inject_tick_data(harness, step=i, run_id=run_id)
            await controller._tick()

        gc.collect()
        process = psutil.Process()
        rss_baseline = process.memory_info().rss

        # Test: run 500 ticks with realistic data
        for i in range(_WARMUP_TICKS, _WARMUP_TICKS + _TEST_TICKS):
            self._inject_tick_data(harness, step=i, run_id=run_id)
            await controller._tick()

        gc.collect()
        rss_final = process.memory_info().rss
        rss_growth = rss_final - rss_baseline

        assert rss_growth < _MAX_RSS_GROWTH_BYTES, (
            f"RSS grew by {rss_growth / 1024 / 1024:.1f} MB "
            f"(limit: {_MAX_RSS_GROWTH_BYTES / 1024 / 1024:.0f} MB) "
            f"over {_TEST_TICKS} ticks — possible memory leak"
        )

    @staticmethod
    def _inject_tick_data(
        harness: ControllerTestHarness,
        step: int,
        run_id: str,
    ) -> None:
        harness.mini_wandb.log_step(
            run_id=run_id,
            step=step,
            metrics={
                "loss": 2.5 - step * 0.001,
                "mfu": 0.45,
                "iteration": float(step),
                "grad_norm": 1.2,
                "iteration_time": 0.8,
            },
        )

        for node_id in _NODE_IDS:
            inject_healthy_node(
                harness.metric_store,
                node_id=node_id,
            )
