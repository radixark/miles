import shlex

from miles.utils.external_utils.command_utils.common import run_process


def exec_command(cmd: str, capture_output: bool = False) -> str | None:
    completed = run_process(shlex.split(cmd), capture_output=capture_output, check=True)
    return completed.stdout if capture_output else None
