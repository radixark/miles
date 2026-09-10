from typing import Self

from pydantic import Field, model_validator

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class SoakCellPolicy(FrozenStrictBaseModel):
    expected_cells: int | None = Field(default=None, ge=1)
    min_survivors: int = Field(default=1, ge=0)
    allow_during_recovery: bool = True
    require_ready_target: bool = False

    @model_validator(mode="after")
    def _validate_topology(self) -> Self:
        if not self.allow_during_recovery and self.expected_cells is None:
            raise ValueError("Recovery gating requires an explicit expected cell count")
        if self.expected_cells is not None and self.min_survivors >= self.expected_cells:
            raise ValueError("The minimum survivor count must leave a cell available for faults")
        return self


class SoakPolicy(FrozenStrictBaseModel):
    max_concurrent_actions: int = Field(default=1, ge=1)
    cell_policies: dict[str, SoakCellPolicy] = Field(default_factory=dict)


def create_policy(
    *,
    expected_cells: dict[str, int],
    allow_during_recovery: bool = True,
    min_survivors: int = 1,
    max_concurrent_actions: int = 1,
) -> SoakPolicy:
    return SoakPolicy(
        max_concurrent_actions=max_concurrent_actions,
        cell_policies={
            kind: SoakCellPolicy(
                expected_cells=count,
                min_survivors=min_survivors,
                allow_during_recovery=allow_during_recovery,
            )
            for kind, count in expected_cells.items()
        },
    )
