import miles.utils.external_utils.command_utils as command_utils
from miles.utils.external_utils import exec_command
from miles.utils.external_utils.command_utils import common, ray_backend

_MODULES_DEFINING_HELPERS = (exec_command, common, ray_backend, command_utils)


def patch_helper(monkeypatch, name: str, replacement) -> None:
    patched = [module for module in _MODULES_DEFINING_HELPERS if hasattr(module, name)]
    assert patched, f"no command_utils module defines {name}"
    for module in patched:
        monkeypatch.setattr(module, name, replacement)


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

    patch_helper(monkeypatch, "exec_command_cpu", fake_exec_command)
    patch_helper(monkeypatch, "exec_command_gpu", fake_exec_command)
    patch_helper(monkeypatch, "exec_command_multi_node", fake_exec_command_multi_node)

    return commands
