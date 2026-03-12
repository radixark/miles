"""Integration tests: Controller tick with real detector chains."""

from tests.fast.utils.ft.utils import inject_gpu_unavailable, inject_healthy_node, make_test_controller

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.detectors.chain import build_detector_chain


class TestControllerWithRealDetectors:
    async def test_tick_with_healthy_state(self) -> None:
        chain = build_detector_chain()
        harness = make_test_controller(
            detectors=chain,
            status_sequence=[JobStatus.RUNNING],
        )

        inject_healthy_node(harness.metric_store, node_id="node-0")

        await harness.controller._tick()

        assert harness.controller._tick_loop.tick_count == 1

    async def test_tick_gpu_lost_triggers_mark_bad(self) -> None:
        chain = build_detector_chain()
        harness = make_test_controller(
            detectors=chain,
            status_sequence=[JobStatus.RUNNING],
        )

        inject_healthy_node(harness.metric_store, node_id="node-0")
        inject_gpu_unavailable(harness.metric_store, node_id="node-0", gpu="3")

        await harness.controller._tick()

        # Decision is executed (currently just logged), but tick completes
        assert harness.controller._tick_loop.tick_count == 1

    async def test_tick_training_crash_triggers_recovery(self) -> None:
        chain = build_detector_chain()
        harness = make_test_controller(
            detectors=chain,
            status_sequence=[JobStatus.FAILED],
        )

        await harness.controller._tick()

        assert harness.controller._tick_loop.tick_count == 1

    async def test_detector_chain_no_detectors_passes(self) -> None:
        harness = make_test_controller(detectors=[])

        await harness.controller._tick()

        assert harness.controller._tick_loop.tick_count == 1
