from typing import Annotated, Literal

from pydantic import Discriminator, NonNegativeInt

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.naming import format_name_index


class _ProcessIdentityBase(FrozenStrictBaseModel):
    component: str

    def to_name(self) -> str:
        return self.component


class SimpleProcessIdentity(_ProcessIdentityBase):
    component: Literal[
        "main",
        "rollout_executor",
        "inference_controller",
        "multi_lora_controller",
        "worker_manager",
        "registration_reporter",
    ]


class TrainerControllerProcessIdentity(_ProcessIdentityBase):
    component: Literal["trainer_controller"] = "trainer_controller"
    trainer_id: str
    model_id: str | None = None

    def to_name(self) -> str:
        return f"{self.component}_{self.trainer_id}"


class TrainProcessIdentity(_ProcessIdentityBase):
    component: Literal["actor", "critic"]
    model_id: str | None = None
    cell_index: NonNegativeInt
    rank_within_cell: NonNegativeInt

    def to_name(self) -> str:
        return (
            f"{f'{x}_' if (x := self.model_id) else ''}{self.component}"
            f"_cell{format_name_index(self.cell_index)}_rank{format_name_index(self.rank_within_cell)}"
        )


ProcessIdentity = Annotated[
    SimpleProcessIdentity | TrainerControllerProcessIdentity | TrainProcessIdentity,
    Discriminator("component"),
]
