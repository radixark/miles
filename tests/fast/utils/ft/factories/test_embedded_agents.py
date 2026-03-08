from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from miles.utils.ft.factories.embedded_agent import _ensure_ray_actor_on_node, ensure_node_agent


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
        with patch("miles.utils.ft.factories.embedded_agent.ray") as mock_ray:
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

    def test_concurrent_creation_race_logs_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """When another rank creates the actor concurrently, logs info and does not raise."""
        with patch("miles.utils.ft.factories.embedded_agent.ray") as mock_ray:
            mock_ray.get_actor.side_effect = ValueError("not found")
            actor_cls = MagicMock()
            actor_cls.options.return_value.remote.side_effect = ValueError(
                "actor already exists"
            )

            with caplog.at_level(logging.INFO):
                _ensure_ray_actor_on_node(
                    actor_cls=actor_cls,
                    name="test_actor",
                    node_id="node-1",
                )

            assert "created by another rank concurrently" in caplog.text

    def test_unexpected_exception_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Non-ValueError exceptions are logged as warnings with exc_info."""
        with patch("miles.utils.ft.factories.embedded_agent.ray") as mock_ray:
            mock_ray.get_actor.side_effect = ValueError("not found")
            actor_cls = MagicMock()
            actor_cls.options.return_value.remote.side_effect = RuntimeError("boom")

            with caplog.at_level(logging.WARNING):
                _ensure_ray_actor_on_node(
                    actor_cls=actor_cls,
                    name="test_actor",
                    node_id="node-1",
                )

            assert "Failed to create actor" in caplog.text
            assert "boom" in caplog.text

    def test_custom_start_method(self) -> None:
        """Respects a custom start_method parameter."""
        with patch("miles.utils.ft.factories.embedded_agent.ray") as mock_ray:
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
    def test_graceful_degrade_on_exception(
        self,
        mock_get_ft_id: MagicMock,
        mock_ray: MagicMock,
        mock_ensure: MagicMock,
    ) -> None:
        """ensure_node_agent is wrapped with @graceful_degrade, so exceptions return None."""
        mock_ray.get_runtime_context.side_effect = RuntimeError("no ray")

        result = ensure_node_agent()

        assert result is None
