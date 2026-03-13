from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from miles.utils.ft.factories.embedded_agent import _ensure_ray_actor_on_node, ensure_node_agent
from miles.utils.ft.factories.node_agent import build_node_agent


class TestEnsureRayActorOnNode:
    """Tests for the generic _ensure_ray_actor_on_node helper."""

    def test_idempotent_when_actor_already_exists(self) -> None:
        """Second call is a no-op when the actor is already registered."""
        with patch("miles.utils.ft.factories.embedded_agent.ray") as mock_ray:
            mock_ray.get_actor.return_value = MagicMock()
            actor_cls = MagicMock()

            _ensure_ray_actor_on_node(
                actor_cls=actor_cls,
                name="test_actor",
                node_id="node-1",
            )

            mock_ray.get_actor.assert_called_once_with("test_actor")
            actor_cls.options.assert_not_called()

    def test_creates_actor_when_not_exists(self) -> None:
        """Creates and starts the actor when ray.get_actor raises ValueError."""
        with (
            patch("miles.utils.ft.factories.embedded_agent.ray") as mock_ray,
            patch("miles.utils.ft.factories.embedded_agent.NodeAffinitySchedulingStrategy"),
        ):
            mock_ray.get_actor.side_effect = ValueError("not found")
            actor_cls = MagicMock()
            handle = MagicMock()
            actor_cls.options.return_value.remote.return_value = handle

            _ensure_ray_actor_on_node(
                actor_cls=actor_cls,
                name="test_actor",
                node_id="node-1",
                actor_kwargs={"key": "val"},
                start_method="start",
            )

            actor_cls.options.assert_called_once()
            actor_cls.options.return_value.remote.assert_called_once_with(key="val")
            handle.start.remote.assert_called_once()
            mock_ray.get.assert_called_once_with(handle.start.remote.return_value)

    def test_start_method_failure_propagates(self) -> None:
        """Previously start() failures were silently swallowed. Now they
        propagate so the caller knows the node agent failed to start."""
        with (
            patch("miles.utils.ft.factories.embedded_agent.ray") as mock_ray,
            patch("miles.utils.ft.factories.embedded_agent.NodeAffinitySchedulingStrategy"),
        ):
            mock_ray.get_actor.side_effect = ValueError("not found")
            actor_cls = MagicMock()
            handle = MagicMock()
            actor_cls.options.return_value.remote.return_value = handle
            mock_ray.get.side_effect = RuntimeError("start failed")

            with pytest.raises(RuntimeError, match="start failed"):
                _ensure_ray_actor_on_node(
                    actor_cls=actor_cls,
                    name="test_actor",
                    node_id="node-1",
                )

    def test_concurrent_creation_race_logs_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """When another rank creates the actor concurrently, logs info and does not raise."""
        with (
            patch("miles.utils.ft.factories.embedded_agent.ray") as mock_ray,
            patch("miles.utils.ft.factories.embedded_agent.NodeAffinitySchedulingStrategy"),
        ):
            mock_ray.get_actor.side_effect = ValueError("not found")
            actor_cls = MagicMock()
            actor_cls.options.return_value.remote.side_effect = ValueError("actor already exists")

            with caplog.at_level(logging.INFO):
                _ensure_ray_actor_on_node(
                    actor_cls=actor_cls,
                    name="test_actor",
                    node_id="node-1",
                )

            assert "created by another rank concurrently" in caplog.text

    def test_unexpected_exception_propagates(self) -> None:
        """Non-ValueError exceptions propagate instead of being silently swallowed.
        Previously the except-Exception branch just logged a warning and returned,
        which caused the entire node's FT to silently fail."""
        with (
            patch("miles.utils.ft.factories.embedded_agent.ray") as mock_ray,
            patch("miles.utils.ft.factories.embedded_agent.NodeAffinitySchedulingStrategy"),
        ):
            mock_ray.get_actor.side_effect = ValueError("not found")
            actor_cls = MagicMock()
            actor_cls.options.return_value.remote.side_effect = RuntimeError("boom")

            with pytest.raises(RuntimeError, match="boom"):
                _ensure_ray_actor_on_node(
                    actor_cls=actor_cls,
                    name="test_actor",
                    node_id="node-1",
                )

    def test_custom_start_method(self) -> None:
        """Respects a custom start_method parameter."""
        with (
            patch("miles.utils.ft.factories.embedded_agent.ray") as mock_ray,
            patch("miles.utils.ft.factories.embedded_agent.NodeAffinitySchedulingStrategy"),
        ):
            mock_ray.get_actor.side_effect = ValueError("not found")
            actor_cls = MagicMock()
            handle = MagicMock()
            actor_cls.options.return_value.remote.return_value = handle

            _ensure_ray_actor_on_node(
                actor_cls=actor_cls,
                name="test_actor",
                node_id="node-1",
                start_method="initialize",
            )

            handle.initialize.remote.assert_called_once()
            mock_ray.get.assert_called_once_with(handle.initialize.remote.return_value)


class TestEnsureNodeAgent:
    """Tests for the ensure_node_agent wrapper."""

    @patch("miles.utils.ft.factories.embedded_agent._ensure_ray_actor_on_node")
    @patch("miles.utils.ft.factories.embedded_agent.ray")
    @patch("miles.utils.ft.factories.embedded_agent.get_ft_id", return_value="job-42")
    def test_calls_ensure_with_correct_args(
        self,
        mock_get_ft_id: MagicMock,
        mock_ray: MagicMock,
        mock_ensure: MagicMock,
    ) -> None:
        mock_ray.get_runtime_context.return_value.get_node_id.return_value = "node-abc"

        ensure_node_agent()

        mock_ensure.assert_called_once()
        call_kwargs = mock_ensure.call_args
        assert call_kwargs.kwargs["node_id"] == "node-abc"
        assert call_kwargs.kwargs["actor_kwargs"] == {
            "builder": build_node_agent,
            "node_id": "node-abc",
            "ft_id": "job-42",
        }

    @patch("miles.utils.ft.factories.embedded_agent._ensure_ray_actor_on_node")
    @patch("miles.utils.ft.factories.embedded_agent.ray")
    def test_explicit_ft_id_overrides_env(
        self,
        mock_ray: MagicMock,
        mock_ensure: MagicMock,
    ) -> None:
        mock_ray.get_runtime_context.return_value.get_node_id.return_value = "node-abc"

        ensure_node_agent(ft_id="explicit-id")

        call_kwargs = mock_ensure.call_args
        assert call_kwargs.kwargs["actor_kwargs"]["ft_id"] == "explicit-id"

    @patch("miles.utils.ft.factories.embedded_agent._ensure_ray_actor_on_node")
    @patch("miles.utils.ft.factories.embedded_agent.ray")
    @patch("miles.utils.ft.factories.embedded_agent.get_ft_id", return_value="job-42")
    def test_exception_propagates_instead_of_silent_degrade(
        self,
        mock_get_ft_id: MagicMock,
        mock_ray: MagicMock,
        mock_ensure: MagicMock,
    ) -> None:
        """Previously ensure_node_agent was wrapped with @graceful_degrade,
        silently swallowing failures. Now exceptions propagate so callers
        know FT is not actually active on this node."""
        mock_ray.get_runtime_context.side_effect = RuntimeError("no ray")

        with pytest.raises(RuntimeError, match="no ray"):
            ensure_node_agent()
