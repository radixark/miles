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
