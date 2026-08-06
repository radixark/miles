import subprocess
import sys


def test_miles_router_module_help_exits_successfully() -> None:
    """The Miles router module exposes its CLI help without starting the server."""
    result: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "-m", "miles.router.router", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--config-json" in result.stdout
