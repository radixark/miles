import uuid

from pydantic import Field, JsonValue, NonNegativeInt

from miles.utils.audit_utils.process_identity import ProcessIdentity
from miles.utils.pydantic_utils import FrozenStrictBaseModel


# =============================== Raw snapshots ================================


class ConfigSnapshotContext(FrozenStrictBaseModel):
    """Identify a capture session, its source process, and its run UUID."""

    name: str
    deploy_component: str
    deploy_instance_id: str
    source: ProcessIdentity
    run_uuid: str = Field(min_length=1)
    capture_id: str = Field(default_factory=lambda: uuid.uuid4().hex)


class ConfigSnapshotPoint(FrozenStrictBaseModel):
    """Identify one occurrence of a stage within a capture session."""

    stage: str  # Capture point, such as process_config, checkpoint_load, or train_first_step.
    index: NonNegativeInt  # Zero-based occurrence of this stage within the same capture session.

    def to_key(self) -> str:
        return f"{self.stage}-{self.index:04d}"


class ConfigSnapshotRecord(FrozenStrictBaseModel):
    """Store one raw configuration capture with its context and sampling point."""

    context: ConfigSnapshotContext
    point: ConfigSnapshotPoint
    config: JsonValue


# ============================ Converted snapshots =============================


class ConfigSnapshotProcess(FrozenStrictBaseModel):
    """Represent equivalent ranks using one base sample and per-stage diffs."""

    ranks: list[int]
    base: JsonValue
    diffs: dict[str, str]


class ConfigSnapshotCase(FrozenStrictBaseModel):
    """Collect all logical process snapshots for one test case."""

    processes: dict[str, ConfigSnapshotProcess]
