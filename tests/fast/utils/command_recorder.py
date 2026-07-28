import miles.utils.external_utils.command_utils as command_utils
import miles.utils.misc as misc


def record_commands(monkeypatch) -> list[str]:
    """Replace every command-executing helper with a recorder and return the list it appends to."""
    commands: list[str] = []

    def fake_exec_command(cmd: str, capture_output: bool = False) -> str | None:
        commands.append(cmd)
        return "0" if capture_output else None

    def fake_exec_command_all_ray_node(
        cmd: str, capture_output: bool = False, num_nodes: int | None = None
    ) -> list[str | None]:
        commands.append(f"[all_ray_node num_nodes={num_nodes}] {cmd}")
        return ["0"]

    for module in (command_utils, misc):
        monkeypatch.setattr(module, "exec_command", fake_exec_command, raising=False)
        monkeypatch.setattr(module, "exec_command_all_ray_node", fake_exec_command_all_ray_node, raising=False)

    return commands
