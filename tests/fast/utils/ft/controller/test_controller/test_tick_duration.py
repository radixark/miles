from __future__ import annotations

import pytest
from tests.fast.utils.ft.conftest import get_sample_value, make_test_controller

import miles.utils.ft.controller.metrics.metric_names as mn


class TestTickDuration:
    @pytest.mark.anyio
    async def test_tick_records_duration_to_exporter(self) -> None:
        harness = make_test_controller()
        registry = harness.controller_exporter._registry

        assert get_sample_value(registry, mn.CONTROLLER_TICK_DURATION_SECONDS + "_count") == 0.0

        await harness.controller._tick()

        count = get_sample_value(registry, mn.CONTROLLER_TICK_DURATION_SECONDS + "_count")
        assert count == 1.0

        total = get_sample_value(registry, mn.CONTROLLER_TICK_DURATION_SECONDS + "_sum")
        assert total is not None
        assert total >= 0.0

    @pytest.mark.anyio
    async def test_tick_records_duration_even_on_error(self) -> None:
        harness = make_test_controller()
        registry = harness.controller_exporter._registry

        async def _raise(*_a: object, **_kw: object) -> None:
            raise RuntimeError("boom")

        harness.main_job.get_job_status = _raise  # type: ignore[assignment]

        await harness.controller._tick()

        count = get_sample_value(registry, mn.CONTROLLER_TICK_DURATION_SECONDS + "_count")
        assert count == 1.0
