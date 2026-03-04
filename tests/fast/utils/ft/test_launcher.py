from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from miles.utils.ft.platform.controller_actor import _build_notifier
from miles.utils.ft.platform.lark_notifier import LarkWebhookNotifier
from miles.utils.ft.platform.launcher import app
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
