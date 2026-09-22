import os
import re
from pathlib import Path
from typing import Literal

import typer
from pydantic import Field

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.env_vars import POD_UID_ENV_VAR


class ProcessIdentity(FrozenStrictBaseModel):
    pid: int = Field(gt=1)
    start_ticks: int


class ProcessTarget(FrozenStrictBaseModel):
    pod_uid: str
    boot_id: str
    pid_namespace: str
    init_start_ticks: int
    pattern: str
    processes: list[ProcessIdentity] = Field(min_length=1)


class ProcessSignalReceipt(FrozenStrictBaseModel):
    kind: Literal["process_signal"] = "process_signal"
    request_id: str = Field(min_length=1)
    target: ProcessTarget
    operation: Literal["kill", "stop"]
    signalled_pids: list[int] = Field(min_length=1)

    def validate_for(self, *, request_id: str, target: ProcessTarget, operation: Literal["kill", "stop"]) -> None:
        raise NotImplementedError


def observe_processes(*, pod_uid: str, pattern: str) -> ProcessTarget:
    assert os.environ[POD_UID_ENV_VAR] == pod_uid, "Pod identity changed"
    matcher = re.compile(pattern)
    processes = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit() or int(path.name) in {1, os.getpid(), os.getppid()}:
            continue
        try:
            start_ticks = _start_ticks(int(path.name))
            command = (path / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if matcher.search(command) and _start_ticks(int(path.name)) == start_ticks:
                processes.append(ProcessIdentity(pid=int(path.name), start_ticks=start_ticks))
        except FileNotFoundError:
            continue
    return ProcessTarget(
        pod_uid=pod_uid,
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        pid_namespace=os.readlink("/proc/self/ns/pid"),
        init_start_ticks=_start_ticks(1),
        pattern=pattern,
        processes=sorted(processes, key=lambda process: process.pid),
    )


def _start_ticks(pid: int) -> int:
    return int((Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19])


app = typer.Typer()


@app.command()
def observe(pod_uid: str, pattern: str) -> None:
    print(observe_processes(pod_uid=pod_uid, pattern=pattern).model_dump_json(), flush=True)


if __name__ == "__main__":
    app()
