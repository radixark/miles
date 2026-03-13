"""Unit tests for factories/rollout_agent.py."""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest

import miles.utils.ft.factories.rollout_agent as rollout_agent_mod
from miles.utils.ft.factories.rollout_agent import _register_with_controller


class TestFactoryNamingConsistency:
    def test_build_prefix_used_for_pure_assembly(self) -> None:
        """Factory functions that do pure assembly should use build_* prefix, not create_*."""
        public_functions = [
            name
            for name, obj in inspect.getmembers(rollout_agent_mod, inspect.isfunction)
            if not name.startswith("_")
        ]
        create_functions = [name for name in public_functions if name.startswith("create_")]
        assert create_functions == [], (
            f"Found create_* functions that should use build_* prefix: {create_functions}"
        )


class TestRegisterWithController:
    def test_successful_registration(self) -> None:
        """Happy path: registration completes without error."""
        agent = MagicMock()
        agent.address = "http://localhost:9000"

        mock_controller = MagicMock()
        mock_controller.register_rollout.remote.return_value = "ref"

        with patch("miles.utils.ft.factories.rollout_agent.ray") as mock_ray:
            mock_ray.get_actor.return_value = mock_controller
            mock_ray.get_runtime_context.return_value.current_actor = MagicMock()
            mock_ray.get.return_value = None

            _register_with_controller(agent=agent, ft_id="test-ft")

        mock_controller.register_rollout.remote.assert_called_once()

    def test_raises_on_actor_not_found(self) -> None:
        """Previously, registration failures were silently swallowed by
        @graceful_degrade, leaving the rollout subsystem invisible to the
        controller. Now failures propagate as exceptions."""
        agent = MagicMock()
        agent.address = "http://localhost:9000"

        with patch("miles.utils.ft.factories.rollout_agent.ray") as mock_ray:
            mock_ray.get_actor.side_effect = ValueError("actor not found")

            with pytest.raises(ValueError, match="actor not found"):
                _register_with_controller(agent=agent, ft_id="test-ft")

    def test_raises_after_all_retries_exhausted(self) -> None:
        """Previously, exhausted retries were silently swallowed. Now a
        RuntimeError is raised to prevent the agent from running unregistered."""
        agent = MagicMock()
        agent.address = "http://localhost:9000"

        mock_controller = MagicMock()
        mock_controller.register_rollout.remote.return_value = "ref"

        with patch("miles.utils.ft.factories.rollout_agent.ray") as mock_ray:
            mock_ray.get_actor.return_value = mock_controller
            mock_ray.get_runtime_context.return_value.current_actor = MagicMock()
            mock_ray.get.side_effect = RuntimeError("persistent ray error")

            with pytest.raises(RuntimeError, match="registration failed"):
                _register_with_controller(agent=agent, ft_id="test-ft")

    def test_retry_on_transient_failure(self) -> None:
        """Registration retries on transient Ray errors via retry_sync."""
        agent = MagicMock()
        agent.address = "http://localhost:9000"

        mock_controller = MagicMock()
        call_count = 0

        def _flaky_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("transient ray error")
            return None

        with patch("miles.utils.ft.factories.rollout_agent.ray") as mock_ray:
            mock_ray.get_actor.return_value = mock_controller
            mock_ray.get_runtime_context.return_value.current_actor = MagicMock()
            mock_ray.get.side_effect = _flaky_get
            mock_controller.register_rollout.remote.return_value = "ref"

            _register_with_controller(agent=agent, ft_id="test-ft")

        assert call_count >= 2
