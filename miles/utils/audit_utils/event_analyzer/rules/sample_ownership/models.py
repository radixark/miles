from pydantic import Field

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class IssuedSampleIdentityIssue(FrozenStrictBaseModel):
    description: str
    sample_index: int
    identities: list[str]


class CurrentTrainerWitnessIssue(FrozenStrictBaseModel):
    description: str
    replicas: list[str]


class SampleResolutionIssue(FrozenStrictBaseModel):
    description: str
    group_index: int
    slot: int
    sample_index: int
    replica_id: str | None
    trained_rows: list[str]
    skipped_rows: list[str] = Field(default_factory=list)
    drop_count: int


SampleOwnershipIssue = IssuedSampleIdentityIssue | CurrentTrainerWitnessIssue | SampleResolutionIssue
