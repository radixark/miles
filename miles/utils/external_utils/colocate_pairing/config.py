from __future__ import annotations

from pydantic import Field, model_validator

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class PairingLayout(FrozenStrictBaseModel):
    num_inference_cells: int = Field(ge=1)
    num_trainer_cells: int = Field(ge=1)
    num_pods_per_inference_cell: int = Field(ge=1)
    num_pods_per_trainer_cell: int = Field(ge=1)
    num_gpus_per_node: int = Field(ge=1)
    gpu_offset: int = Field(ge=0)

    @property
    def pod_offset(self) -> int:
        return self.gpu_offset // self.num_gpus_per_node

    @model_validator(mode="after")
    def _assert_inference_pods_pair(self) -> PairingLayout:
        assert self.num_pods_per_inference_cell <= self.num_pods_per_trainer_cell, (
            f"An inference cell of {self.num_pods_per_inference_cell} pods cannot fit in a trainer cell of "
            f"{self.num_pods_per_trainer_cell}; colocate needs every inference worker to sit on a trainer's node"
        )
        assert self.num_pods_per_trainer_cell % self.num_pods_per_inference_cell == 0, (
            f"{self.num_pods_per_trainer_cell} trainer pods per cell is not a whole number of "
            f"{self.num_pods_per_inference_cell}-pod inference cells, so an inference would straddle two trainer cells"
        )
        assert self.gpu_offset % self.num_gpus_per_node == 0, (
            f"An inference pool starting at gpu {self.gpu_offset} of the trainer's {self.num_gpus_per_node}-gpu nodes "
            f"starts inside a node, so each of its pods would want part of two trainer pods' gpus"
        )
        assert self.pod_offset % self.num_pods_per_inference_cell == 0, (
            f"An inference pool starting at trainer pod {self.pod_offset} is not a whole number of "
            f"{self.num_pods_per_inference_cell}-pod inference cells in, so its cells would straddle trainer cells"
        )
        assert (
            self.pod_offset + self.num_inference_cells * self.num_pods_per_inference_cell
            <= self.num_trainer_cells * self.num_pods_per_trainer_cell
        ), (
            f"{self.num_inference_cells} inference cells starting at trainer pod {self.pod_offset} do not fit in "
            f"{self.num_trainer_cells} trainer cells of {self.num_pods_per_trainer_cell} pods; an inference worker "
            f"whose trainer pod does not exist would wait for a node forever"
        )
        return self


class InferencePool(FrozenStrictBaseModel):
    pool_id: str
    layout: PairingLayout


class PairingConfig(FrozenStrictBaseModel):
    namespace: str
    release: str
    trainer_pool_id: str
    inference_pools: list[InferencePool] = Field(min_length=1)
