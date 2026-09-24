import json
import shlex

import miles.utils.external_utils.command_utils as command_utils


def record_commands(monkeypatch) -> list[str]:
    """Replace every command-executing helper with a recorder and return the list it appends to."""
    commands: list[str] = []

    def fake_exec_command(cmd: str, capture_output: bool = False) -> str | None:
        commands.append(cmd)
        return "0" if capture_output else None

    def fake_exec_command_multi_node(
        cmd: str, capture_output: bool = False, num_nodes: int | None = None
    ) -> list[str | None]:
        commands.append(f"[multi_node num_nodes={num_nodes}] {cmd}")
        return ["0"]

    def fake_run_ray_job(*, address, entrypoint, runtime_env):
        commands.append(
            f"[launcher lifetime] ray job submit --address={shlex.quote(address)} "
            f"--runtime-env-json={shlex.quote(json.dumps(runtime_env))} -- {entrypoint}"
        )

    monkeypatch.setattr(command_utils, "exec_command_cpu", fake_exec_command)
    monkeypatch.setattr(command_utils, "exec_command_gpu", fake_exec_command)
    monkeypatch.setattr(command_utils, "exec_command_multi_node", fake_exec_command_multi_node)
    monkeypatch.setattr(command_utils, "run_ray_job", fake_run_ray_job)

    return commands
