import copy
import enum
import logging
from argparse import Namespace
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NamedTuple

from miles.utils.simple_checkpointer import load_simple_checkpoint, save_simple_checkpoint
from miles.utils.types import Sample

logger = logging.getLogger(__name__)
_MAX_RETAINED_OUTPUTS = 3


class _RolloutExecutorOutputSnapshotter:
    def __init__(self, *, args: Namespace) -> None:
        self._args = args
        self._snapshots: dict[_OutputSnapshotKey, _OutputSnapshotEntry] = {}

    def capture(
        self, *, trainer_model_id: str | None, rollout_id: int, data: list[Sample], metadata: dict[str, Any]
    ) -> None:
        key = _OutputSnapshotKey(trainer_model_id=trainer_model_id, rollout_id=rollout_id)
        assert key not in self._snapshots, (
            f"rollout {rollout_id} of trainer model {trainer_model_id} was captured before; "
            "a replayed batch must not be captured again"
        )

        self._snapshots[key] = copy.deepcopy(
            _OutputSnapshotEntry(data=data, metadata=metadata, phase=_OutputSnapshotPhase.CAPTURED)
        )
        captured = [one for one, entry in self._snapshots.items() if entry.phase is _OutputSnapshotPhase.CAPTURED]
        for stale in captured[: len(captured) - _MAX_RETAINED_OUTPUTS]:
            del self._snapshots[stale]

    def get(self, *, trainer_model_id: str | None, rollout_id: int) -> "_OutputSnapshotEntry | None":
        key = _OutputSnapshotKey(trainer_model_id=trainer_model_id, rollout_id=rollout_id)
        if (entry := self._snapshots.get(key)) is None:
            return None
        assert entry.phase is not _OutputSnapshotPhase.CAPTURED, (
            f"rollout {rollout_id} of trainer model {trainer_model_id} was captured by this process "
            "and must not be replayed"
        )
        assert (
            entry.phase is not _OutputSnapshotPhase.REPLAYED
        ), f"rollout {rollout_id} of trainer model {trainer_model_id} was already replayed"

        self._snapshots[key] = replace(entry, phase=_OutputSnapshotPhase.REPLAYED)
        return copy.deepcopy(entry)

    def save(self, directory: Path) -> None:
        save_simple_checkpoint(
            directory=directory,
            data={key: (entry.data, entry.metadata) for key, entry in self._snapshots.items()},
        )

    def load(self, directory: Path) -> None:
        assert not any(
            entry.phase is _OutputSnapshotPhase.CAPTURED for entry in self._snapshots.values()
        ), "the executor restores its untrained outputs before it generates any"
        outputs = load_simple_checkpoint(directory=directory)

        self._snapshots.update(
            {
                key: _OutputSnapshotEntry(data=data, metadata=metadata, phase=_OutputSnapshotPhase.LOADED)
                for key, (data, metadata) in outputs.items()
            }
        )
        logger.info(f"Loaded {len(outputs)} untrained rollout batches")


class _OutputSnapshotKey(NamedTuple):
    trainer_model_id: str | None
    rollout_id: int


class _OutputSnapshotPhase(enum.Enum):
    CAPTURED = "captured"
    LOADED = "loaded"
    REPLAYED = "replayed"


@dataclass(frozen=True)
class _OutputSnapshotEntry:
    data: list[Sample]
    metadata: dict[str, Any]
    phase: _OutputSnapshotPhase
