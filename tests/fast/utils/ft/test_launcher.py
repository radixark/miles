from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from miles.utils.ft.launcher import app
from miles.utils.ft.platform.controller_factory import _build_notifier
from miles.utils.ft.platform.lark_notifier import LarkWebhookNotifier
from miles.utils.ft.platform.stubs import StubNotifier

runner = CliRunner()


# ---------------------------------------------------------------------------
# FT Controller CLI tests
# ---------------------------------------------------------------------------


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

    def test_help_includes_metric_store_options(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "--metric-store-backe" in result.output
        assert "--prometheus-url" in result.output
        assert "--controller-exporte" in result.output

    def test_help_includes_runtime_env_json_option(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "--runtime-env-json" in result.output

    def test_help_includes_ft_id_and_label_suffix_options(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "--ft-id" in result.output
        assert "--k8s-label-suffix" in result.output


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
                notifier = _build_notifier(platform="k8s-ray")
        assert notifier is None
        assert "FT_LARK_WEBHOOK_URL" in caplog.text


class TestLauncherSubmitAndRun:
    def test_inline_mode_calls_submit_and_run(self) -> None:
        with (
            patch("miles.utils.ft.launcher.build_ft_controller") as mock_build,
            patch("miles.utils.ft.launcher.asyncio.run") as mock_asyncio_run,
            patch("miles.utils.ft.controller.metrics.exporter.ControllerExporter.start"),
        ):
            mock_controller = MagicMock()
            mock_build.return_value = mock_controller
            result = runner.invoke(app, ["--platform", "stub", "--", "python3", "train.py"])

        assert result.exit_code == 0, result.output
        mock_asyncio_run.assert_called_once()

    def test_entrypoint_passed_to_config(self) -> None:
        with (
            patch("miles.utils.ft.launcher.build_ft_controller") as mock_build,
            patch("miles.utils.ft.launcher.asyncio.run"),
            patch("miles.utils.ft.controller.metrics.exporter.ControllerExporter.start"),
        ):
            mock_build.return_value = MagicMock()
            result = runner.invoke(app, [
                "--platform", "stub",
                "--", "python3", "train.py", "--lr", "0.001",
            ])

        assert result.exit_code == 0, result.output
        config = mock_build.call_args.kwargs["config"]
        assert "python3" in config.entrypoint
        assert "train.py" in config.entrypoint
        assert "--lr" in config.entrypoint

    def test_ft_id_and_label_suffix_passed_to_config(self) -> None:
        with (
            patch("miles.utils.ft.launcher.build_ft_controller") as mock_build,
            patch("miles.utils.ft.launcher.asyncio.run"),
            patch("miles.utils.ft.controller.metrics.exporter.ControllerExporter.start"),
        ):
            mock_build.return_value = MagicMock()
            result = runner.invoke(app, [
                "--platform", "stub",
                "--ft-id", "myft",
                "--k8s-label-suffix", "sfx",
                "--", "python3", "train.py",
            ])

        assert result.exit_code == 0, result.output
        config = mock_build.call_args.kwargs["config"]
        assert config.ft_id == "myft"
        assert config.k8s_label_suffix == "sfx"

    def test_runtime_env_json_parsed_to_config(self) -> None:
        runtime_env = {"env_vars": {"PYTHONPATH": "/root/Megatron-LM"}}
        with (
            patch("miles.utils.ft.launcher.build_ft_controller") as mock_build,
            patch("miles.utils.ft.launcher.asyncio.run"),
            patch("miles.utils.ft.controller.metrics.exporter.ControllerExporter.start"),
        ):
            mock_build.return_value = MagicMock()
            result = runner.invoke(app, [
                "--platform", "stub",
                "--runtime-env-json", '{"env_vars": {"PYTHONPATH": "/root/Megatron-LM"}}',
                "--", "python3", "train.py",
            ])

        assert result.exit_code == 0, result.output
        config = mock_build.call_args.kwargs["config"]
        assert config.runtime_env == runtime_env


class TestLauncherWiring:
    def test_main_uses_build_detector_chain(self) -> None:
        """Verify launcher wires build_detector_chain() into FtController."""
        captured_kwargs: dict = {}

        def fake_controller_init(self: object, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)

        with patch("miles.utils.ft.launcher.FtController.__init__", fake_controller_init), \
             patch("miles.utils.ft.launcher.FtController.submit_initial_training"), \
             patch("miles.utils.ft.launcher.FtController.run"), \
             patch("miles.utils.ft.controller.metrics.exporter.ControllerExporter.start"), \
             patch("miles.utils.ft.launcher.asyncio.run"):
            result = runner.invoke(app, ["--platform", "stub"])

        assert result.exit_code == 0, result.output
        assert "detectors" in captured_kwargs
        assert len(captured_kwargs["detectors"]) > 0
