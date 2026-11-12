"""The `aenv` CLI that Harbor's agentenv environment builds task Dockerfiles with.

Harbor looks it up on the PATH of the process that runs ``Trial.run()``, so the
launcher checks its own host and every rollout worker checks itself.
"""

import errno
import re
import shutil
import subprocess

from miles.utils.retry_utils import NonRetryableError

# older releases panic on HTTPS endpoints (kvcache-ai/AgentENV#317)
_MIN_VERSION = (0, 2, 3)
# estimate: --version only prints the version compiled into the binary
_VERSION_TIMEOUT_S = 30
_INSTALL_HINT = (
    f"agentenv needs aenv {'.'.join(map(str, _MIN_VERSION))} or later on the PATH of every rollout worker; "
    "install it with `curl -fsSL https://github.com/kvcache-ai/AgentENV/releases/latest/download/install-cli.sh | bash`"
)


def preflight_aenv(env_type: str) -> None:
    """Raise unless a new enough `aenv` is on PATH, for the agentenv environment type only.

    A verdict that holds for every trial on this machine (no `aenv`, one that cannot run,
    or one that answers with no version or an old one) is a `NonRetryableError`, which
    stops a rollout; a `--version` call that fails for a passing reason is a `RuntimeError`,
    which costs only this trial.
    """
    if env_type != "agentenv":
        return
    path = shutil.which("aenv")
    if path is None:
        raise NonRetryableError(f"aenv is not on PATH. {_INSTALL_HINT}")
    try:
        completed = subprocess.run(
            [path, "--version"], capture_output=True, text=True, timeout=_VERSION_TIMEOUT_S, check=True
        )
    except (OSError, subprocess.SubprocessError) as e:
        error_type = RuntimeError if _may_pass_next_time(e) else NonRetryableError
        raise error_type(f"could not run `{path} --version` ({e}). {_INSTALL_HINT}") from e
    output = (completed.stdout + completed.stderr).strip()
    match = re.search(r"\baenv (\d+)\.(\d+)\.(\d+)", output)
    if match is None:
        raise NonRetryableError(f"cannot read a version from `{path} --version` ({output!r}). {_INSTALL_HINT}")
    if tuple(int(part) for part in match.groups()) < _MIN_VERSION:
        raise NonRetryableError(f"{path} is {match.group(0)}. {_INSTALL_HINT}")


def _may_pass_next_time(error: OSError | subprocess.SubprocessError) -> bool:
    if isinstance(error, subprocess.TimeoutExpired):
        return True
    if isinstance(error, subprocess.CalledProcessError):
        return error.returncode < 0  # killed by a signal
    return isinstance(error, OSError) and error.errno in (errno.EAGAIN, errno.ENOMEM)  # no room to start a process
