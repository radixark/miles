from typing import Annotated, Literal

from pydantic import Discriminator

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class _ProcessIdentityBase(FrozenStrictBaseModel):
    component: str

    def to_name(self) -> str:
        return self.component


class MainProcessIdentity(_ProcessIdentityBase):
    component: Literal["main"] = "main"


class RolloutManagerProcessIdentity(_ProcessIdentityBase):
    component: Literal["rollout_manager"] = "rollout_manager"


class TrainProcessIdentity(_ProcessIdentityBase):
    component: Literal["actor", "critic"]
    cell_index: int
    rank_within_cell: int

    def to_name(self) -> str:
        return f"{self.component}_cell{self.cell_index}_rank{self.rank_within_cell}"


ProcessIdentity = Annotated[
    MainProcessIdentity | RolloutManagerProcessIdentity | TrainProcessIdentity,
    Discriminator("component"),
]
