from __future__ import annotations

from typer.testing import CliRunner

from miles.utils.ft.platform.launcher import app

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
        assert "--metric-store-backend" in result.output
        assert "--prometheus-url" in result.output
        assert "--controller-exporter-port" in result.output

    def test_invalid_platform_raises_error(self) -> None:
        result = runner.invoke(app, ["--platform", "unknown"])
        assert result.exit_code != 0
        assert "Unknown platform" in result.output

    def test_invalid_metric_store_backend_raises_error(self) -> None:
        result = runner.invoke(app, ["--metric-store-backend", "invalid"])
        assert result.exit_code != 0
        assert "Unknown metric-store-backend" in result.output
