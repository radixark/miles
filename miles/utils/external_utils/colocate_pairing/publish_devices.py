"""Publish a colocated trainer pod's assigned physical GPU UUIDs, then exec the trainer command."""

from __future__ import annotations

import argparse
import os
import sys

from miles.utils.external_utils.colocate_pairing.devices import (
    TrainerDevices,
    discover_assigned_gpu_uuids,
    publish_trainer_devices,
)
from miles.utils.external_utils.colocate_pairing.pods import PodCoordinate
from miles.utils.workers.env_vars import CELL_INDEX_ENV_VAR, NODE_NAME_ENV_VAR, POD_INDEX_ENV_VAR, POD_UID_ENV_VAR


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    assert value, f"Expected environment variable {name} to be set by the Kubernetes downward API"
    return value


def publish_from_env(*, pairing_dir: str, pool_id: str, expected_gpus: int) -> TrainerDevices:
    coord = PodCoordinate(
        pool_id=pool_id,
        cell_index=int(_require_env(CELL_INDEX_ENV_VAR)),
        pod_in_cell_index=int(_require_env(POD_INDEX_ENV_VAR)),
    )
    devices = TrainerDevices(
        pod_uid=_require_env(POD_UID_ENV_VAR),
        node_name=_require_env(NODE_NAME_ENV_VAR),
        gpu_uuids=discover_assigned_gpu_uuids(expected_count=expected_gpus),
    )
    publish_trainer_devices(pairing_dir=pairing_dir, coord=coord, devices=devices)
    return devices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairing-dir", required=True)
    parser.add_argument("--pool-id", required=True)
    parser.add_argument("--expected-gpus", required=True, type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    assert command, "Missing trainer command after '--'"

    publish_from_env(pairing_dir=args.pairing_dir, pool_id=args.pool_id, expected_gpus=args.expected_gpus)
    os.execvp(command[0], command)
    return 0


if __name__ == "__main__":
    sys.exit(main())
