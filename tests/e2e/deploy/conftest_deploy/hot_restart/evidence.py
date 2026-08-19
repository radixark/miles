import logging
from dataclasses import dataclass
from pathlib import Path

from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_observer import ClusterSnapshot

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.test_utils.comparisons.metrics import read_metric_events

logger = logging.getLogger(__name__)

CHECKPOINT_TRACKER_FILENAME: str = "latest_checkpointed_iteration.txt"
TRAIN_STEP_METRIC_KEY: str = "train/grad_norm"


# =========================== how far a run has come ===========================


@dataclass(frozen=True)
class RunProgress:
    last_saved_iteration: int | None
    last_finished_rollout_id: int | None


def read_run_progress(*, checkpoint_dir: Path, events_dir: Path) -> RunProgress:
    return RunProgress(
        last_saved_iteration=read_last_saved_iteration(checkpoint_dir),
        last_finished_rollout_id=read_last_finished_rollout_id(events_dir),
    )


def read_last_saved_iteration(checkpoint_dir: Path) -> int | None:
    if not checkpoint_dir.is_dir():
        return None

    trackers = sorted(checkpoint_dir.glob(f"**/{CHECKPOINT_TRACKER_FILENAME}"))
    assert len(trackers) <= 1, (
        f"{checkpoint_dir} holds the trackers {[str(one) for one in trackers]}: this run trains several policies, "
        f"and one restart window cannot be read off trainers that each save at their own pace"
    )
    if not trackers:
        return None

    text = _read_text_or_empty(trackers[0])
    return int(text) if text.isdigit() else None


def read_last_finished_rollout_id(events_dir: Path) -> int | None:
    return max(read_finished_rollout_ids(events_dir), default=None)


def read_finished_rollout_ids(events_dir: Path) -> list[int]:
    if not events_dir.is_dir():
        return []

    return sorted(
        {
            event.rollout_id
            for event in read_metric_events(events_dir)
            if event.rollout_id is not None and TRAIN_STEP_METRIC_KEY in event.metrics
        }
    )


def _read_text_or_empty(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        logger.info(f"Failed to read {path}; treating it as not written yet", exc_info=True)
        return ""


# ========================== what the take-overs left ==========================


class HotRestartRecord(FrozenStrictBaseModel):
    index: int
    saved_iteration_at_trigger: int | None
    frozen_rollout_id: int


class HotRestartEvidence(FrozenStrictBaseModel):
    records: tuple[HotRestartRecord, ...]
    snapshots: tuple[ClusterSnapshot, ...]
    release: str
    observation_attempts: int = 0
    observation_failures: int = 0

    def write(self, *, dump_dir: str) -> None:
        path = evidence_path(dump_dir=dump_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        logger.info(f"Wrote what {len(self.records)} hot restart(s) left behind to {path}")

    @classmethod
    def load(cls, *, dump_dir: str) -> "HotRestartEvidence":
        path = evidence_path(dump_dir=dump_dir)
        assert path.is_file(), (
            f"{path} does not exist, so nothing recorded which steps this run redid and which pods outlived its "
            f"orchestration script; run the target side before comparing"
        )
        return cls.model_validate_json(path.read_text())


def evidence_path(*, dump_dir: str) -> Path:
    return Path(dump_dir) / "hot_restart" / "evidence.json"
