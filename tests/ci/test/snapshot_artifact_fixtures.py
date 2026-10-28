import os
import shlex
import subprocess
from pathlib import Path


def _artifact_name(*, script: str, command: str, output: Path) -> str:
    argv = ["bash", "-e", "-c", script]
    print(f"EXECUTE_COMMAND={shlex.quote(command)} GITHUB_OUTPUT={shlex.quote(str(output))} {shlex.join(argv)}")
    subprocess.run(
        args=argv,
        env=os.environ | {"EXECUTE_COMMAND": command, "GITHUB_OUTPUT": str(output)},
        check=True,
        timeout=5,
    )
    key, name = output.read_text().strip().split("=", maxsplit=1)
    assert key == "name"
    return name
