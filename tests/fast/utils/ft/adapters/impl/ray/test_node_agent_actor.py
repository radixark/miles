"""Tests for _FtNodeAgentActorCls registration failure behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from miles.utils.ft.adapters.impl.ray.node_agent_actor import _FtNodeAgentActorCls


def _make_fake_agent() -> MagicMock:
    agent = MagicMock()
    agent._node_id = "node-0"
    agent.get_exporter_address.return_value = "http://localhost:9100"
    agent.metadata = {"k8s_node_name": "k8s-node-0"}
    return agent


class TestRegistrationFailurePropagates:
    @pytest.mark.anyio
    async def test_start_waits_for_exporter_before_registering(self) -> None:
        fake_agent = _make_fake_agent()

        actor = _FtNodeAgentActorCls(
            builder=lambda **kw: fake_agent,
            node_id="node-0",
            ft_id="test",
        )

        with patch.object(actor, "_register_with_controller") as register_mock:
            await actor.start()

        fake_agent.start.assert_awaited_once()
        fake_agent.wait_for_exporter_ready.assert_called_once_with()
        register_mock.assert_called_once_with()

    def test_register_failure_raises_instead_of_silent_success(self) -> None:
        """Registration failure must propagate as an exception from
        _register_with_controller, not be silently swallowed.
        Previously @graceful_degrade caught all exceptions."""
        fake_agent = _make_fake_agent()

        actor = _FtNodeAgentActorCls(
            builder=lambda **kw: fake_agent,
            node_id="node-0",
            ft_id="test",
        )

        with patch("miles.utils.ft.adapters.impl.ray.node_agent_actor.ray") as mock_ray:
            mock_controller = MagicMock()
            mock_ray.get_actor.return_value = mock_controller
            mock_ray.get_runtime_context.return_value.current_actor = MagicMock()
            mock_ray.get.side_effect = RuntimeError("connection refused")

            with pytest.raises(RuntimeError, match="Failed to register node agent"):
                actor._register_with_controller()

    def test_register_succeeds_after_transient_failure(self) -> None:
        """Retry logic should still work — transient failure followed by success."""
        fake_agent = _make_fake_agent()

        actor = _FtNodeAgentActorCls(
            builder=lambda **kw: fake_agent,
            node_id="node-0",
            ft_id="test",
        )

        call_count = 0

        def _mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient error")
            return None

        with patch("miles.utils.ft.adapters.impl.ray.node_agent_actor.ray") as mock_ray:
            mock_controller = MagicMock()
            mock_ray.get_actor.return_value = mock_controller
            mock_ray.get_runtime_context.return_value.current_actor = MagicMock()
            mock_ray.get.side_effect = _mock_get

            actor._register_with_controller()

        assert call_count == 2
