from datetime import datetime, timedelta, timezone
from typing import Any

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.utils.audit_utils.event_logger.models import (
    EnvReport,
    EnvReportArgsDump,
    EnvReportEvent,
    EnvReportProcessFacts,
    InferenceEngineWeightChecksumEvent,
    TrainGroupStepEndEvent,
    WeightUpdateResultEvent,
)
from miles.utils.audit_utils.process_identity import (
    SimpleProcessIdentity,
    TrainerControllerProcessIdentity,
    TrainProcessIdentity,
)

START = datetime(2026, 9, 26, tzinfo=timezone.utc)

MOVEMENT_ARGS: dict[str, Any] = dict(lora_rank=0, lora_adapter_path=None, update_weights_interval=1)


def at(seconds: float) -> datetime:
    return START + timedelta(seconds=seconds)


def make_result(
    *,
    second: float,
    update_id: str,
    published_version: int | None,
    cell_hashes: dict[str, str],
    updated: list[str],
    failed: list[str] | None = None,
    load_state_timestamp: float = 0.0,
) -> WeightUpdateResultEvent:
    return WeightUpdateResultEvent(
        timestamp=at(second),
        source=TrainerControllerProcessIdentity(trainer_id="actor"),
        debug_weight_update_id=update_id,
        debug_trainer_load_state_timestamp=load_state_timestamp,
        rollout_id=0,
        candidate_version=published_version,
        published_version=published_version,
        snapshot_cell_id_to_hashes=cell_hashes,
        updated_cell_ids=updated,
        failed_cell_ids=failed or [],
    )


def make_checksum(
    *,
    second: float,
    update_id: str,
    weight_version: int,
    snapshots: dict[str, tuple[str, dict[str, str]]],
    model_id: str | None = None,
    load_state_timestamp: float = 0.0,
) -> InferenceEngineWeightChecksumEvent:
    return InferenceEngineWeightChecksumEvent(
        timestamp=at(second),
        source=SimpleProcessIdentity(component="main"),
        rollout_id=weight_version - 1,
        trainer_model_id=model_id,
        weight_version=weight_version,
        debug_trainer_load_state_timestamp=load_state_timestamp,
        debug_weight_update_id=update_id,
        engine_snapshots=[
            dict(cell_id=cell_id, workers_hash=workers_hash, tensor_checksums=checksums)
            for cell_id, (workers_hash, checksums) in snapshots.items()
        ],
    )


def make_step_end(
    *, second: float, rollout_id: int = 0, cell_outcomes: dict[int, list[TrainStepOutcome]] | None = None
) -> TrainGroupStepEndEvent:
    return TrainGroupStepEndEvent(
        timestamp=at(second),
        source=TrainerControllerProcessIdentity(trainer_id="actor"),
        rollout_id=rollout_id,
        attempt=0,
        role="actor",
        cell_outcomes={0: [TrainStepOutcome.NORMAL]} if cell_outcomes is None else cell_outcomes,
    )


def make_trainer_args(*, rank: int = 0, **overrides: Any) -> EnvReportEvent:
    return EnvReportEvent(
        timestamp=START,
        source=TrainProcessIdentity(component="actor", cell_index=0, rank_within_cell=rank),
        report=EnvReport(
            process=EnvReportProcessFacts(
                hostname="node-0",
                argv=[],
                args=EnvReportArgsDump(values={**MOVEMENT_ARGS, **overrides}, skipped_names=[]),
                env_vars={},
                launcher_env_report=None,
            ),
            key_versions={},
            editable_packages=[],
            git_repos=[],
            full_pip_list=[],
            packages_probed=False,
        ),
    )
