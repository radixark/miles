"""Unit tests for RayControllerClient (P1 item 11)."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from miles.utils.ft.adapters.impl.ray.controller_client import RayControllerClient


class TestGetHandleCaching:
    def test_caches_handle_after_first_lookup(self) -> None:
        with patch("miles.utils.ft.adapters.impl.ray.controller_client.ray") as mock_ray:
            mock_actor = MagicMock()
            mock_ray.get_actor.return_value = mock_actor

            client = RayControllerClient(ft_id="test-ft")
            handle1 = client._get_handle()
            handle2 = client._get_handle()

        assert handle1 is mock_actor
        assert handle2 is mock_actor
        mock_ray.get_actor.assert_called_once()

    def test_returns_none_on_actor_not_found(self) -> None:
        with patch("miles.utils.ft.adapters.impl.ray.controller_client.ray") as mock_ray:
            mock_ray.get_actor.side_effect = ValueError("actor not found")

            client = RayControllerClient(ft_id="missing-ft")
            handle = client._get_handle()

        assert handle is None


class TestRegisterTrainingRank:
    def test_raises_when_controller_not_available(self) -> None:
        with patch("miles.utils.ft.adapters.impl.ray.controller_client.ray") as mock_ray:
            mock_ray.get_actor.side_effect = ValueError("not found")

            client = RayControllerClient(ft_id="test")
            with pytest.raises(RuntimeError, match="controller not available"):
                client.register_training_rank(
                    run_id="run-1",
                    rank=0,
                    world_size=8,
                    node_id="node-0",
                    exporter_address="http://localhost:9000",
                    pid=1234,
                )

        mock_ray.get.assert_not_called()

    def test_calls_ray_get_with_timeout(self) -> None:
        with patch("miles.utils.ft.adapters.impl.ray.controller_client.ray") as mock_ray:
            mock_controller = MagicMock()
            mock_ray.get_actor.return_value = mock_controller
            mock_controller.register_training_rank.remote.return_value = "ref"

            client = RayControllerClient(ft_id="test")
            client.register_training_rank(
                run_id="run-1",
                rank=0,
                world_size=8,
                node_id="node-0",
                exporter_address="http://localhost:9000",
                pid=1234,
                timeout_seconds=30.0,
            )

        mock_ray.get.assert_called_once_with("ref", timeout=30.0)


class TestLogStep:
    def test_fire_and_forget_does_not_call_ray_get(self) -> None:
        with patch("miles.utils.ft.adapters.impl.ray.controller_client.ray") as mock_ray:
            mock_controller = MagicMock()
            mock_ray.get_actor.return_value = mock_controller

            client = RayControllerClient(ft_id="test")
            client.log_step(run_id="run-1", step=42, metrics={"loss": 2.5})

        mock_controller.log_step.remote.assert_called_once_with(
            run_id="run-1",
            step=42,
            metrics={"loss": 2.5},
        )
        mock_ray.get.assert_not_called()

    def test_skips_silently_when_controller_unavailable(self) -> None:
        with patch("miles.utils.ft.adapters.impl.ray.controller_client.ray") as mock_ray:
            mock_ray.get_actor.side_effect = ValueError("not found")

            client = RayControllerClient(ft_id="test")
            client.log_step(run_id="run-1", step=1, metrics={"loss": 3.0})
