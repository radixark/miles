import json
import shlex

import miles.utils.external_utils.ray_job as ray_job
from miles.utils.external_utils.command_utils.base_backend import BaseCommandBackend


def patch_helper(monkeypatch, name: str, replacement, *, backend_class: type = BaseCommandBackend) -> None:
    assert hasattr(backend_class, name), f"no method of {backend_class.__name__} is named {name}"
    monkeypatch.setattr(backend_class, name, replacement, raising=False)


def record_commands(monkeypatch) -> list[str]:
    """Replace every exec_command backend method with a recorder and return the list it appends to.

    Patching BaseCommandBackend covers every backend, including one added after this was written: the
    public forms are defined there and no subclass is allowed to hide them behind an override.
    """
    commands: list[str] = []

    def fake_exec_command(self, cmd: str, capture_output: bool = False, **kwargs) -> str | None:
        commands.append(cmd)
        return "0" if capture_output else None

    def fake_exec_command_multi_node(
        self,
        cmd: str,
        capture_output: bool = False,
        num_nodes: int | None = None,
        num_gpus_per_node: int | None = None,
    ) -> list[str | None]:
        commands.append(f"[multi_node num_nodes={num_nodes}] {cmd}")
        return ["0"]

    def fake_run_launcher_owned_job(*, address, entrypoint, runtime_env):
        commands.append(
            f"[launcher lifetime] ray job submit --address={shlex.quote(address)} "
            f"--runtime-env-json={shlex.quote(json.dumps(runtime_env))} -- {entrypoint}"
        )

    monkeypatch.setattr(ray_job, "exec_command_cpu", lambda cmd, **kwargs: fake_exec_command(None, cmd, **kwargs))
    monkeypatch.setattr(ray_job, "_run_launcher_owned_job", fake_run_launcher_owned_job)
    patch_helper(monkeypatch, "exec_command_cpu", fake_exec_command)
    patch_helper(monkeypatch, "exec_command_gpu", fake_exec_command)
    patch_helper(monkeypatch, "exec_command_multi_node", fake_exec_command_multi_node)

    return commands
