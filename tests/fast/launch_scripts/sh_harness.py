import difflib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT_PLACEHOLDER = "<REPO_ROOT>"
SANDBOX_PLACEHOLDER = "<SANDBOX>"

_ARG_SEPARATOR = "\x1f"
_RECORD_SEPARATOR = "\x1e"

_SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

_FROZEN_ENV = {
    "HOME": "/root",
    "LANG": "C",
    "LC_ALL": "C",
    "TERM": "dumb",
    "MASTER_ADDR": "127.0.0.1",
    "NODE_RANK": "0",
    "WANDB_KEY": "frozen-wandb-key",
    "WANDB_API_KEY": "frozen-wandb-api-key",
}

_SHIMMED_COMMANDS = (
    "apt",
    "apt-get",
    "awk",
    "curl",
    "date",
    "docker",
    "git",
    "hf",
    "ip",
    "mkdir",
    "nc",
    "nvidia-smi",
    "pip",
    "pip3",
    "pkill",
    "ps",
    "python",
    "python3",
    "ray",
    "rm",
    "rsync",
    "scp",
    "sleep",
    "ssh",
    "torchrun",
    "wget",
)

_GPU_COUNT_LARGER_THAN_ANY_WAIT_LOOP_EXPECTS = "1000000"

_FROZEN_RAY_DASHBOARD_PROCESS = "root 1 0.0 0.0 ray dashboard --node-ip-address=10.0.0.1 --dashboard-port=8265"

_SHIM_STDOUT = {
    "date": "20260101_000000",
    "ps": _FROZEN_RAY_DASHBOARD_PROCESS,
}

_PYTHON_SHIM_BODY = """case "${1:-}" in
*/model_args_utils.py) "$MILES_SH_HARNESS_REAL_PYTHON" "$@" ;;
-c)
    case "$2" in
    *cluster_resources*) printf '%s\\n' 'REPLACE_GPU_COUNT' ;;
    *import*) ;;
    *) "$MILES_SH_HARNESS_REAL_PYTHON" "$@" ;;
    esac
    ;;
esac
""".replace(
    "REPLACE_GPU_COUNT", _GPU_COUNT_LARGER_THAN_ANY_WAIT_LOOP_EXPECTS
)

_SHIM_BODY = {
    "python": _PYTHON_SHIM_BODY,
    "python3": _PYTHON_SHIM_BODY,
}

_SHIM_TEMPLATE = """#!/bin/bash
record="$$${{MILES_SH_HARNESS_ARG_SEP}}{name}"
for arg in "$@"; do
    record="$record$MILES_SH_HARNESS_ARG_SEP$arg"
done
printf '%s%s' "$record" "$MILES_SH_HARNESS_RECORD_SEP" >>"$MILES_SH_HARNESS_CAPTURE"
{stdout_statement}exit 0
"""


def iter_launch_scripts() -> list[Path]:
    roots = [REPO_ROOT / "scripts", REPO_ROOT / "examples"]
    return sorted(
        path for root in roots for path in root.rglob("*.sh") if "ray job submit" in path.read_text(errors="replace")
    )


@dataclass(frozen=True)
class LaunchScriptRun:
    invocations: list[list[str]]
    stdout: str
    stderr: str
    returncode: int

    def invocations_of(self, command: str) -> list[list[str]]:
        return [argv for argv in self.invocations if argv[0] == command]

    def ray_job_submit_argv(self) -> list[str]:
        matches = [argv for argv in self.invocations_of("ray") if argv[1:3] == ["job", "submit"]]
        assert len(matches) == 1, f"expected exactly one `ray job submit`, got {len(matches)}"
        return matches[0]


def run_launch_script(
    script: Path,
    sandbox: Path,
    args: tuple[str, ...] = (),
    extra_env: dict[str, str] | None = None,
    timeout: float = 120.0,
) -> LaunchScriptRun:
    sandbox.mkdir(parents=True, exist_ok=True)
    fake_bin = sandbox / "fake_bin"
    capture = sandbox / "capture"
    workdir = sandbox / "workdir"
    _write_shims(fake_bin)
    capture.write_bytes(b"")
    workdir.mkdir(exist_ok=True)

    frozen = {
        **_FROZEN_ENV,
        "PATH": f"{fake_bin}:{_SYSTEM_PATH}",
        "MILES_SH_HARNESS_CAPTURE": str(capture),
        "MILES_SH_HARNESS_ARG_SEP": _ARG_SEPARATOR,
        "MILES_SH_HARNESS_RECORD_SEP": _RECORD_SEPARATOR,
        "MILES_SH_HARNESS_REAL_PYTHON": sys.executable,
    }
    _reject_unfreezing(extra_env or {}, frozen=frozen)

    deadline = time.monotonic() + timeout
    process = subprocess.Popen(
        ["bash", str(script), *args],
        cwd=workdir,
        env={**frozen, **(extra_env or {})},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise
    _wait_until_the_script_leaves_nothing_running(process.pid, deadline=deadline)

    invocations = _parse_capture(capture.read_text(), sandbox=sandbox)

    return LaunchScriptRun(
        invocations=invocations,
        stdout=sanitize(stdout, sandbox=sandbox),
        stderr=sanitize(stderr, sandbox=sandbox),
        returncode=process.returncode,
    )


SNAPSHOT_UPDATE_ENV_VAR = "MILES_UPDATE_LAUNCH_SCRIPT_SNAPSHOTS"


def assert_matches_snapshot(snapshot: Path, actual: str, subject: str) -> None:
    if os.environ.get(SNAPSHOT_UPDATE_ENV_VAR):
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(actual)
        return

    assert snapshot.exists(), f"missing snapshot for {subject}; regenerate with {SNAPSHOT_UPDATE_ENV_VAR}=1"
    expected = snapshot.read_text()
    if actual != expected:
        # a bare equality on a file-sized string reports nothing a reader can act on, and the
        # machine that regenerated the snapshot is rarely the one that fails on it
        diff = difflib.unified_diff(
            expected.splitlines(), actual.splitlines(), fromfile=f"{snapshot}", tofile="actual", lineterm=""
        )
        raise AssertionError(f"{subject} does not match its snapshot:\n" + "\n".join(diff))


def format_invocations(invocations: list[list[str]]) -> str:
    lines = []
    for index, argv in enumerate(invocations):
        lines.append(f"### {index}")
        lines.extend(json.dumps(arg) for arg in argv)
        lines.append("")
    return "\n".join(lines)


def _reject_unfreezing(extra_env: dict[str, str], frozen: dict[str, str]) -> None:
    """Silently shadowing PATH or a frozen value would unshim the run or unfreeze the snapshot."""
    collisions = sorted(set(extra_env) & set(frozen))
    assert not collisions, f"extra_env may not override the harness-controlled {collisions}"


def _wait_until_the_script_leaves_nothing_running(pgid: int, deadline: float) -> None:
    """A script backgrounding a shimmed command outlives bash, and would append after we read."""
    while True:
        alive = _live_pids_of_group(pgid)
        if not alive:
            return
        assert time.monotonic() < deadline, f"process group {pgid} still running after the timeout: {sorted(alive)}"
        time.sleep(0.005)


def _live_pids_of_group(pgid: int) -> set[int]:
    """A zombie can no longer append to the capture, but killpg still reports its group as alive.

    A command backgrounded inside a command substitution is orphaned when the substitution's
    subshell exits, and nothing reaps it when PID 1 is not an init that does so.
    """
    live = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rpartition(")")[2].split()
        except OSError:
            continue
        state, process_group = fields[0], int(fields[2])
        if process_group == pgid and state != "Z":
            live.add(int(entry.name))
    return live


def _write_shims(fake_bin: Path) -> None:
    fake_bin.mkdir(exist_ok=True)
    for name in _SHIMMED_COMMANDS:
        stdout = _SHIM_STDOUT.get(name)
        stdout_statement = "" if stdout is None else f"printf '%s\\n' {stdout!a}\n"
        shim = fake_bin / name
        shim.write_text(_SHIM_TEMPLATE.format(name=name, stdout_statement=stdout_statement + _SHIM_BODY.get(name, "")))
        shim.chmod(0o755)


def _parse_capture(raw: str, sandbox: Path) -> list[list[str]]:
    """Order by pid, not by append order: a `&` child appends whenever it gets scheduled."""
    records = [record.split(_ARG_SEPARATOR) for record in raw.split(_RECORD_SEPARATOR) if record != ""]
    records.sort(key=lambda record: int(record[0]))
    return [[sanitize(arg, sandbox=sandbox) for arg in record[1:]] for record in records]


def sanitize(text: str, sandbox: Path) -> str:
    return text.replace(str(sandbox), SANDBOX_PLACEHOLDER).replace(str(REPO_ROOT), REPO_ROOT_PLACEHOLDER)
