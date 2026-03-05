"""Unit tests for FtMegatronAgent.

FtMegatronAgent is responsible only for heartbeat gauges (iteration + phase)
exposed via a Prometheus HTTP exporter, and rank registration with FtController.
Training metrics are forwarded separately by FtTrackingAgent via tracking_utils.
"""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import httpx
import pytest

from miles.utils.ft.agents.core.megatron_agent import FtMegatronAgent


def _parse_gauge(text: str, metric_name: str, labels: dict[str, str]) -> float:
    """Extract a gauge value from Prometheus text exposition format."""
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if metric_name not in line:
            continue
        label_match = all(f'{k}="{v}"' in line for k, v in labels.items())
        if label_match:
            value_str = line.rsplit(" ", 1)[-1]
            return float(value_str)
    raise ValueError(f"{metric_name} with labels {labels} not found in metrics output")


@pytest.fixture()
def agent() -> Iterator[FtMegatronAgent]:
    agent = FtMegatronAgent(rank=0, world_size=4)
    yield agent
    agent.shutdown()


class TestFtMegatronAgentExporter:
    @pytest.mark.anyio
    async def test_exporter_returns_prometheus_format(
        self, agent: FtMegatronAgent
    ) -> None:
        address = agent.get_exporter_address()
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{address}/metrics")

        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")

    @pytest.mark.anyio
    async def test_exporter_address_has_port(
        self, agent: FtMegatronAgent
    ) -> None:
        address = agent.get_exporter_address()
        assert address.startswith("http://localhost:")
        port = int(address.split(":")[-1])
        assert port > 0

    @pytest.mark.anyio
    async def test_initial_gauge_values(
        self, agent: FtMegatronAgent
    ) -> None:
        address = agent.get_exporter_address()
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{address}/metrics")

        text = response.text
        assert "miles_ft_training_iteration" in text
        assert "miles_ft_training_phase" in text
        assert 'rank="0"' in text


class TestFtMegatronAgentStep:
    @pytest.mark.anyio
    async def test_step_updates_iteration_gauge(
        self, agent: FtMegatronAgent
    ) -> None:
        agent.step(iteration=42)

        address = agent.get_exporter_address()
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{address}/metrics")

        text = response.text
        assert "miles_ft_training_iteration" in text
        assert "42.0" in text

    def test_step_does_not_interact_with_controller(
        self, agent: FtMegatronAgent
    ) -> None:
        agent._controller_handle = MagicMock()
        agent.step(iteration=10)
        agent._controller_handle.log_step.remote.assert_not_called()

    def test_step_warns_on_non_increasing_iteration(
        self, agent: FtMegatronAgent, caplog: pytest.LogCaptureFixture
    ) -> None:
        agent.step(iteration=5)
        agent.step(iteration=5)
        assert "non-increasing iteration" in caplog.text
        assert agent._last_iteration == 5

    def test_step_warns_on_decreasing_iteration(
        self, agent: FtMegatronAgent, caplog: pytest.LogCaptureFixture
    ) -> None:
        agent.step(iteration=5)
        agent.step(iteration=3)
        assert "non-increasing iteration" in caplog.text
        assert agent._last_iteration == 5

    @pytest.mark.anyio
    async def test_step_iteration_monotonic_across_phases(
        self, agent: FtMegatronAgent
    ) -> None:
        """Simulate a full rollout cycle with split set_phase/step API."""
        address = agent.get_exporter_address()
        labels = {"rank": "0"}

        agent.set_phase("training")
        for step_id in range(4):
            agent.step(iteration=step_id)

        agent.set_phase("idle")
        agent.set_phase("checkpoint_saving")
        agent.set_phase("idle")

        agent.set_phase("training")
        for step_id in range(4, 8):
            agent.step(iteration=step_id)

        agent.set_phase("idle")

        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{address}/metrics")
        iteration = _parse_gauge(resp.text, "miles_ft_training_iteration", labels)
        phase = _parse_gauge(resp.text, "miles_ft_training_phase", labels)
        assert iteration == 7.0
        assert phase == 0.0


class TestFtMegatronAgentSetPhase:
    @pytest.mark.anyio
    async def test_set_phase_updates_phase_gauge(
        self, agent: FtMegatronAgent
    ) -> None:
        agent.set_phase("checkpoint_saving")

        address = agent.get_exporter_address()
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{address}/metrics")

        labels = {"rank": "0"}
        phase = _parse_gauge(response.text, "miles_ft_training_phase", labels)
        assert phase == 2.0

    @pytest.mark.anyio
    async def test_set_phase_idle_preserves_iteration(
        self, agent: FtMegatronAgent
    ) -> None:
        agent.step(iteration=10)
        agent.set_phase("idle")

        address = agent.get_exporter_address()
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{address}/metrics")

        labels = {"rank": "0"}
        iteration = _parse_gauge(response.text, "miles_ft_training_iteration", labels)
        phase = _parse_gauge(response.text, "miles_ft_training_phase", labels)
        assert iteration == 10.0
        assert phase == 0.0


class TestFtMegatronAgentRegisterRank:
    @patch("miles.utils.ft.agents.core.megatron_agent.FtMegatronAgent._get_controller_handle")
    def test_register_rank_calls_controller(
        self, mock_get_handle: MagicMock
    ) -> None:
        mock_controller = MagicMock()
        mock_ray_get = MagicMock()
        mock_get_handle.return_value = mock_controller

        with patch.dict(
            "os.environ", {"MILES_FT_TRAINING_RUN_ID": "test-run-1"}
        ), patch("ray.get", mock_ray_get):
            agent = FtMegatronAgent(rank=0, world_size=4)
            try:
                mock_controller.register_rank.remote.assert_called_once()
                call_kwargs = mock_controller.register_rank.remote.call_args[1]
                assert call_kwargs["run_id"] == "test-run-1"
                assert call_kwargs["rank"] == 0
                assert call_kwargs["world_size"] == 4
            finally:
                agent.shutdown()

    @patch("miles.utils.ft.agents.core.megatron_agent.FtMegatronAgent._get_controller_handle")
    def test_register_rank_retries_on_failure(
        self, mock_get_handle: MagicMock
    ) -> None:
        mock_controller = MagicMock()
        mock_get_handle.return_value = mock_controller

        call_count = 0

        def ray_get_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("simulated failure")
            return None

        with patch.dict(
            "os.environ", {"MILES_FT_TRAINING_RUN_ID": "test-run-1"}
        ), patch("ray.get", side_effect=ray_get_side_effect), patch(
            "time.sleep"
        ):
            agent = FtMegatronAgent(rank=0, world_size=4)
            try:
                assert call_count == 3
                assert mock_controller.register_rank.remote.call_count == 3
            finally:
                agent.shutdown()

    @patch("miles.utils.ft.agents.core.megatron_agent.FtMegatronAgent._get_controller_handle")
    def test_register_rank_all_attempts_fail_no_exception(
        self, mock_get_handle: MagicMock
    ) -> None:
        mock_controller = MagicMock()
        mock_get_handle.return_value = mock_controller

        with patch.dict(
            "os.environ", {"MILES_FT_TRAINING_RUN_ID": "test-run-1"}
        ), patch(
            "ray.get", side_effect=RuntimeError("always fails")
        ), patch("time.sleep"):
            agent = FtMegatronAgent(rank=2, world_size=4)
            try:
                assert mock_controller.register_rank.remote.call_count == 3
            finally:
                agent.shutdown()

    def test_register_rank_skipped_without_run_id(self) -> None:
        agent = FtMegatronAgent(rank=0, world_size=4)
        try:
            assert agent._run_id == ""
        finally:
            agent.shutdown()

    @patch("miles.utils.ft.agents.core.megatron_agent.FtMegatronAgent._get_controller_handle")
    def test_register_rank_skipped_when_controller_unavailable(
        self, mock_get_handle: MagicMock
    ) -> None:
        mock_get_handle.return_value = None

        with patch.dict("os.environ", {"MILES_FT_TRAINING_RUN_ID": "test-run-1"}):
            agent = FtMegatronAgent(rank=0, world_size=4)
            try:
                assert agent._run_id == "test-run-1"
            finally:
                agent.shutdown()

    @patch("miles.utils.ft.agents.core.megatron_agent.FtMegatronAgent._get_controller_handle")
    def test_register_rank_asserts_node_id_and_exporter_address(
        self, mock_get_handle: MagicMock
    ) -> None:
        mock_controller = MagicMock()
        mock_ray_get = MagicMock()
        mock_get_handle.return_value = mock_controller

        with patch.dict(
            "os.environ", {"MILES_FT_TRAINING_RUN_ID": "test-run-1"}
        ), patch("ray.get", mock_ray_get):
            agent = FtMegatronAgent(rank=0, world_size=4)
            try:
                call_kwargs = mock_controller.register_rank.remote.call_args[1]
                assert call_kwargs["node_id"] == agent._node_id
                assert call_kwargs["exporter_address"] == agent.get_exporter_address()
            finally:
                agent.shutdown()

    @patch("miles.utils.ft.agents.core.megatron_agent.FtMegatronAgent._get_controller_handle")
    def test_register_rank_includes_pid(
        self, mock_get_handle: MagicMock
    ) -> None:
        import os as _os

        mock_controller = MagicMock()
        mock_ray_get = MagicMock()
        mock_get_handle.return_value = mock_controller

        with patch.dict(
            "os.environ", {"MILES_FT_TRAINING_RUN_ID": "test-run-1"}
        ), patch("ray.get", mock_ray_get):
            agent = FtMegatronAgent(rank=0, world_size=4)
            try:
                call_kwargs = mock_controller.register_rank.remote.call_args[1]
                assert call_kwargs["pid"] == _os.getpid()
            finally:
                agent.shutdown()


class TestFtMegatronAgentFaultTolerance:
    def test_maybe_create_returns_agent_when_enabled(self) -> None:
        agent = FtMegatronAgent.maybe_create(rank=0, world_size=4, enabled=True)
        try:
            assert agent is not None
            assert isinstance(agent, FtMegatronAgent)
        finally:
            if agent is not None:
                agent.shutdown()

    def test_maybe_create_returns_none_when_disabled(self) -> None:
        agent = FtMegatronAgent.maybe_create(rank=0, world_size=4, enabled=False)
        assert agent is None

    def test_maybe_create_returns_none_on_init_error(self) -> None:
        with patch.object(
            FtMegatronAgent, "__init__", side_effect=RuntimeError("init failed")
        ):
            agent = FtMegatronAgent.maybe_create(rank=0, world_size=4)
            assert agent is None

    def test_maybe_create_without_run_id_still_creates(self) -> None:
        agent = FtMegatronAgent.maybe_create(rank=0, world_size=4)
        try:
            assert agent is not None
            assert agent._run_id == ""
        finally:
            if agent is not None:
                agent.shutdown()

    def test_step_exception_does_not_propagate(self) -> None:
        agent = FtMegatronAgent(rank=0, world_size=4)
        try:
            with patch.object(
                agent, "_iteration_child", **{"set.side_effect": RuntimeError("boom")}
            ):
                agent.step(iteration=1)
        finally:
            agent.shutdown()

