"""Unit tests for factories/rollout_agent.py (P0 item 8)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from miles.utils.ft.factories.rollout_agent import _register_with_controller


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

    def test_graceful_degrade_on_actor_not_found(self) -> None:
        """When controller actor is not found, @graceful_degrade suppresses the error."""
        agent = MagicMock()
        agent.address = "http://localhost:9000"

        with patch("miles.utils.ft.factories.rollout_agent.ray") as mock_ray:
            mock_ray.get_actor.side_effect = ValueError("actor not found")

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
