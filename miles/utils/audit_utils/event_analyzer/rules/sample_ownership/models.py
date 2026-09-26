from pydantic import Field

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class IssuedSampleIdentityIssue(FrozenStrictBaseModel):
    """Produced when one source sample index was issued for more than one GRPO slot."""

    description: str
    sample_index: int
    identities: list[str]


class SampleResolutionIssue(FrozenStrictBaseModel):
    """Produced when an eligible source sample does not end in exactly one outcome on a current cell."""

    description: str
    group_index: int | None
    slot: int | None
    sample_index: int
    cell_index: int | None
    trained_consumptions: list[str]
    skipped_consumptions: list[str] = Field(default_factory=list)
    drop_count: int


SampleOwnershipIssue = IssuedSampleIdentityIssue | SampleResolutionIssue
