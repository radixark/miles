import copy
import logging
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from miles.utils.simple_checkpointer import SimpleCheckpointer
from miles.utils.types import Sample

logger = logging.getLogger(__name__)
_CHECKPOINTER = SimpleCheckpointer(path_template="rollout/executor_state_{rollout_id}.pt", require_exists=True)


class _RolloutExecutorOutputSnapshotter:
    def __init__(self, *, args: Namespace) -> None:
        self._args = args
        self._outputs: dict[_OutputSnapshotKey, _OutputSnapshot] = {}

    def capture(
        self, *, trainer_model_id: str | None, rollout_id: int, data: list[Sample], metadata: dict[str, Any]
    ) -> None:
        if self._args.save is not None:
            key = _OutputSnapshotKey(trainer_model_id=trainer_model_id, rollout_id=rollout_id)
            self._outputs[key] = copy.deepcopy(_OutputSnapshot(data=data, metadata=metadata))

    def has(self, *, trainer_model_id: str | None, rollout_id: int) -> bool:
        return _OutputSnapshotKey(trainer_model_id=trainer_model_id, rollout_id=rollout_id) in self._outputs

    def get(self, *, trainer_model_id: str | None, rollout_id: int) -> "_OutputSnapshot":
        key = _OutputSnapshotKey(trainer_model_id=trainer_model_id, rollout_id=rollout_id)
        return copy.deepcopy(self._outputs[key])

    def save(self, rollout_id: int) -> None:
        self._outputs = {key: output for key, output in self._outputs.items() if key.rollout_id > rollout_id}
        _CHECKPOINTER.save(args=self._args, rollout_id=rollout_id, data=self._outputs)

    def load(self, rollout_id: int | None) -> None:
        if (outputs := _CHECKPOINTER.load(args=self._args, rollout_id=rollout_id)) is None:
            return
        self._outputs = outputs
        if self._args.ci_inject_missing_prefetched_batch_bug:
            assert self._args.ci_test and self._outputs
            self._outputs.clear()
        logger.info(f"Loaded {len(self._outputs)} untrained rollout batches")


class _OutputSnapshotKey(NamedTuple):
    trainer_model_id: str | None
    rollout_id: int


@dataclass(frozen=True)
class _OutputSnapshot:
    data: list[Sample]
    metadata: dict[str, Any]


def compute_executor_state_path(directory: str | Path, *, rollout_id: int | None) -> Path:
    return _CHECKPOINTER.path(directory=directory, rollout_id=rollout_id)
