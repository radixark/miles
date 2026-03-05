from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from miles.utils.ft.platform.controller_factory import _build_notifier
from miles.utils.ft.platform.lark_notifier import LarkWebhookNotifier
from miles.utils.ft.launcher import app
from miles.utils.ft.platform.stubs import StubNotifier

runner = CliRunner()


class TestLauncherCli:
    def test_help_output(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "tick-interval" in result.output
        assert "FT Controller" in result.output

    def test_help_includes_platform_option(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "--platform" in result.output
        assert "--ray-address" in result.output
        assert "--entrypoint" in result.output

    def test_help_includes_metric_store_options(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "--metric-store-backe" in result.output
        assert "--prometheus-url" in result.output
        assert "--controller-exporte" in result.output

    def test_help_includes_as_ray_actor_option(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "--as-ray-actor" in result.output


class TestBuildNotifier:
    def test_webhook_url_returns_lark_notifier(self) -> None:
        with patch.dict("os.environ", {"FT_LARK_WEBHOOK_URL": "https://hook.example.com"}):
            notifier = _build_notifier(platform="stub")
        assert isinstance(notifier, LarkWebhookNotifier)

    def test_stub_mode_without_webhook_returns_stub(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            notifier = _build_notifier(platform="stub")
        assert isinstance(notifier, StubNotifier)

    def test_k8s_ray_mode_without_webhook_returns_none(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            notifier = _build_notifier(platform="k8s-ray")
        assert notifier is None

    def test_empty_webhook_url_treated_as_unset(self) -> None:
        with patch.dict("os.environ", {"FT_LARK_WEBHOOK_URL": "  "}):
            notifier = _build_notifier(platform="stub")
        assert isinstance(notifier, StubNotifier)

    def test_no_webhook_non_stub_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging
        with patch.dict("os.environ", {}, clear=True):
            with caplog.at_level(logging.WARNING):
                notifier = _build_notifier(platform_mode="k8s-ray")
        assert notifier is None
        assert "no_notifier_configured" in caplog.text


class TestLauncherAsRayActor:
    def test_as_ray_actor_creates_detached_actor(self) -> None:
        mock_actor_handle = MagicMock()
        mock_options = MagicMock()
        mock_options.remote.return_value = mock_actor_handle

        with (
            patch("miles.utils.ft.launcher.FtControllerActor") as mock_cls,
            patch("miles.utils.ft.launcher.asyncio.run") as mock_asyncio_run,
        ):
            mock_cls.options.return_value = mock_options
            result = runner.invoke(app, ["--platform", "stub", "--as-ray-actor"])

        assert result.exit_code == 0, result.output
        mock_cls.options.assert_called_once_with(name="ft_controller", lifetime="detached")
        mock_options.remote.assert_called_once()
        mock_actor_handle.run.remote.assert_called_once()
        mock_asyncio_run.assert_not_called()

    def test_inline_mode_calls_asyncio_run(self) -> None:
        with (
            patch("miles.utils.ft.launcher.build_ft_controller") as mock_build,
            patch("miles.utils.ft.launcher.asyncio.run") as mock_asyncio_run,
            patch("miles.utils.ft.launcher.ControllerExporter.start"),
        ):
            mock_controller = MagicMock()
            mock_build.return_value = mock_controller
            result = runner.invoke(app, ["--platform", "stub"])

        assert result.exit_code == 0, result.output
        mock_asyncio_run.assert_called_once_with(mock_controller.run())


class TestLauncherWiring:
    def test_main_uses_build_detector_chain(self) -> None:
        """Verify launcher wires build_detector_chain() into FtController."""
        captured_kwargs: dict = {}

        def fake_controller_init(self: object, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)

        with patch("miles.utils.ft.launcher.FtController.__init__", fake_controller_init), \
             patch("miles.utils.ft.launcher.FtController.run"), \
             patch("miles.utils.ft.launcher.ControllerExporter.start"), \
             patch("miles.utils.ft.launcher.asyncio.run"):
            result = runner.invoke(app, ["--platform", "stub"])

        assert result.exit_code == 0, result.output
        assert "detectors" in captured_kwargs
        assert len(captured_kwargs["detectors"]) > 0
