from typing import Annotated, Literal

from pydantic import Discriminator, NonNegativeInt

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class _ProcessIdentityBase(FrozenStrictBaseModel):
    component: str

    def to_name(self) -> str:
        return self.component


class MainProcessIdentity(_ProcessIdentityBase):
    component: Literal["main"] = "main"


class RolloutExecutorProcessIdentity(_ProcessIdentityBase):
    component: Literal["rollout_executor"] = "rollout_executor"


class TrainProcessIdentity(_ProcessIdentityBase):
    component: Literal["actor", "critic"]
    cell_id: str
    rank_within_cell: NonNegativeInt

    def to_name(self) -> str:
        return f"{self.component}_{self.cell_id}_rank{self.rank_within_cell}"


ProcessIdentity = Annotated[
    MainProcessIdentity | RolloutExecutorProcessIdentity | TrainProcessIdentity,
    Discriminator("component"),
]
