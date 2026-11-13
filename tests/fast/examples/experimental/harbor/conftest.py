"""Put the example on sys.path so its module imports by bare name, the way a
rollout loads it (``--custom-agent-function-path``)."""

import sys

import pytest

from . import EXAMPLE_DIR

if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))


@pytest.fixture
def fake_aenv(tmp_path, monkeypatch):
    """Empty the PATH; calling the fixture puts an `aenv` there whose `--version` prints the given text."""
    bin_dir = tmp_path / "path-bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", str(bin_dir))

    def install(version_output: str, *, exit_code: int = 0) -> None:
        aenv = bin_dir / "aenv"
        aenv.write_text(f"#!/bin/sh\necho '{version_output}'\nexit {exit_code}\n")
        aenv.chmod(0o755)

    return install
