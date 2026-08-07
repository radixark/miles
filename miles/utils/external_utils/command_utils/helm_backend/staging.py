from __future__ import annotations

import shlex
from pathlib import Path

from miles.utils.external_utils.command_utils.common import rsync_command

_LOCK_DIR_NAME = ".miles-staging-locks"


def parse_pairs(pairs: tuple[str, ...]) -> list[tuple[str, str]]:
    parsed = []
    for pair in pairs:
        source, separator, destination = pair.partition(":")
        assert separator and source and destination, f"expected 'source:destination', got {pair!r}"
        parsed.append((source, destination))
    return parsed


def staging_command(pairs: tuple[str, ...], node_local_root: str) -> str | None:
    parsed = parse_pairs(pairs)
    if not parsed:
        return None

    return " && ".join(
        rsync_command(
            path_src=source,
            path_dst=destination,
            lock_path=lock_path_for(destination, node_local_root=node_local_root),
        )
        for source, destination in parsed
    )


def lock_path_for(destination: str, node_local_root: str) -> str:
    assert node_local_root, (
        "staging needs a node-local volume: set nodeLocalStorage.hostPath, or the copy and its lock "
        "land in each pod's own filesystem where neither is shared"
    )
    assert Path(destination).is_relative_to(node_local_root), (
        f"staging destination {destination} is not under the node-local root {node_local_root}; a "
        f"destination elsewhere is either pod-private or cluster-wide, and this lock guards neither"
    )
    return str(Path(node_local_root) / _LOCK_DIR_NAME / f"{Path(destination).name}.lock")


def with_staging(command: list[str], pairs: tuple[str, ...], node_local_root: str = "") -> list[str]:
    staging = staging_command(pairs, node_local_root=node_local_root)
    if staging is None:
        return command

    return ["bash", "-c", f"{staging} && exec {shlex.join(command)}"]
