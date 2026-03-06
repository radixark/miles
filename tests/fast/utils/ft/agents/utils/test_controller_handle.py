"""Unit tests for get_controller_handle."""

from unittest.mock import MagicMock, patch

from miles.utils.ft.agents.utils.controller_handle import get_controller_handle


class TestGetControllerHandle:
    @patch("ray.get_actor")
    def test_returns_actor_handle(self, mock_get_actor: MagicMock) -> None:
        mock_handle = MagicMock()
        mock_get_actor.return_value = mock_handle

        result = get_controller_handle("abc123")

        assert result is mock_handle
        mock_get_actor.assert_called_once_with("ft_controller_abc123")

    @patch("ray.get_actor")
    def test_returns_none_on_failure(self, mock_get_actor: MagicMock) -> None:
        mock_get_actor.side_effect = RuntimeError("not found")

        result = get_controller_handle("abc123")

        assert result is None

    @patch("ray.get_actor")
    def test_empty_ft_id_uses_default_name(self, mock_get_actor: MagicMock) -> None:
        mock_get_actor.return_value = MagicMock()

        get_controller_handle("")

        mock_get_actor.assert_called_once_with("ft_controller")
