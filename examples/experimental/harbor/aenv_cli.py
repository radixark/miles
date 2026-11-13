"""The `aenv` CLI that Harbor's agentenv environment builds task Dockerfiles with.

Harbor looks it up on the PATH of each rollout worker; the launcher can check
only its own host.
"""

import re
import shutil
import subprocess

# older releases panic on HTTPS endpoints (kvcache-ai/AgentENV#317)
_MIN_VERSION = (0, 2, 3)
# estimate: --version only prints the version compiled into the binary
_VERSION_TIMEOUT_S = 30
_INSTALL_HINT = (
    f"agentenv needs aenv {'.'.join(map(str, _MIN_VERSION))} or later on the PATH of every rollout worker; "
    "install it with `curl -fsSL https://github.com/kvcache-ai/AgentENV/releases/latest/download/install-cli.sh | bash`"
)


def preflight_aenv(env_type: str) -> None:
    """Raise unless a new enough `aenv` is on this host's PATH, for the agentenv environment type only."""
    if env_type != "agentenv":
        return
    path = shutil.which("aenv")
    if path is None:
        raise RuntimeError(f"aenv is not on PATH. {_INSTALL_HINT}")
    try:
        completed = subprocess.run(
            [path, "--version"], capture_output=True, text=True, timeout=_VERSION_TIMEOUT_S, check=True
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise RuntimeError(f"could not run `{path} --version` ({e}). {_INSTALL_HINT}") from e
    output = (completed.stdout + completed.stderr).strip()
    match = re.search(r"\baenv (\d+)\.(\d+)\.(\d+)", output)
    if match is None:
        raise RuntimeError(f"cannot read a version from `{path} --version` ({output!r}). {_INSTALL_HINT}")
    if tuple(int(part) for part in match.groups()) < _MIN_VERSION:
        raise RuntimeError(f"{path} is {match.group(0)}. {_INSTALL_HINT}")
