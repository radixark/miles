from __future__ import annotations

import os
import subprocess
from pathlib import Path

from pydantic import Field

from miles.utils.external_utils.colocate_pairing.pods import PodCoordinate
from miles.utils.file_utils import atomic_write_text
from miles.utils.pydantic_utils import FrozenStrictBaseModel

_NVIDIA_SMI_QUERY_COMMAND = ("nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader")
_VISIBLE_DEVICES_UNFILTERED = frozenset({"", "all"})
_VISIBLE_DEVICES_NONE = frozenset({"none", "void", "-1"})


class TrainerDevices(FrozenStrictBaseModel):
    pod_uid: str = Field(min_length=1)
    node_name: str = Field(min_length=1)
    gpu_uuids: tuple[str, ...] = Field(min_length=1)


def pairing_file_path(pairing_dir: str | Path, coord: PodCoordinate) -> Path:
    return Path(pairing_dir) / f"{coord.pool_id}__{coord.cell_index}__{coord.pod_in_cell_index}.json"


def discover_assigned_gpu_uuids(*, expected_count: int | None = None) -> tuple[str, ...]:
    """Return physical GPU UUIDs assigned to this container in container-local CUDA order."""
    visible_raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_raw.lower() in _VISIBLE_DEVICES_NONE:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible_raw!r} exposes no GPUs to the colocated trainer")

    result = subprocess.run(
        _NVIDIA_SMI_QUERY_COMMAND,
        check=True,
        capture_output=True,
        text=True,
    )
    container_uuids = tuple(line.strip() for line in result.stdout.splitlines() if line.strip())
    if not container_uuids:
        raise RuntimeError("nvidia-smi reported no GPUs inside the colocated trainer container")

    selected_uuids = _apply_cuda_visible_devices(container_uuids, visible_raw=visible_raw)
    if len(set(selected_uuids)) != len(selected_uuids):
        raise ValueError(f"Duplicate physical GPU UUIDs discovered for colocated trainer: {selected_uuids}")
    if expected_count is not None and len(selected_uuids) != expected_count:
        raise RuntimeError(
            f"Colocated trainer expected {expected_count} assigned GPU(s), but discovered {len(selected_uuids)}: "
            f"{list(selected_uuids)}"
        )
    return selected_uuids


def _apply_cuda_visible_devices(container_uuids: tuple[str, ...], *, visible_raw: str) -> tuple[str, ...]:
    if visible_raw.lower() in _VISIBLE_DEVICES_UNFILTERED:
        return container_uuids

    tokens = [token.strip() for token in visible_raw.split(",") if token.strip()]
    if all(token.isdigit() for token in tokens):
        indices = [int(token) for token in tokens]
        out_of_range = [index for index in indices if index >= len(container_uuids)]
        if out_of_range:
            raise RuntimeError(
                f"CUDA_VISIBLE_DEVICES={visible_raw!r} references index(es) {out_of_range}, "
                f"but container sees only {len(container_uuids)} GPU(s)"
            )
        return tuple(container_uuids[index] for index in indices)

    unknown = [token for token in tokens if token not in container_uuids]
    if unknown:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={visible_raw!r} names GPU(s) {unknown} not present in nvidia-smi output "
            f"{list(container_uuids)}"
        )
    return tuple(tokens)


def publish_trainer_devices(
    *,
    pairing_dir: str | Path,
    coord: PodCoordinate,
    devices: TrainerDevices,
) -> Path:
    path = pairing_file_path(pairing_dir, coord)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, devices.model_dump_json() + "\n")
    return path


def read_trainer_devices(*, pairing_dir: str | Path, coord: PodCoordinate) -> TrainerDevices | None:
    path = pairing_file_path(pairing_dir, coord)
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return None
    return TrainerDevices.model_validate_json(raw)
