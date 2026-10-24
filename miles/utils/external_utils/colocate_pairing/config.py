from __future__ import annotations

from pydantic import Field, model_validator

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class PairingLayout(FrozenStrictBaseModel):
    num_inference_cells: int = Field(ge=1)
    num_trainer_cells: int = Field(ge=1)
    num_pods_per_inference_cell: int = Field(ge=1)
    num_pods_per_trainer_cell: int = Field(ge=1)
    num_gpus_per_node: int = Field(ge=1)
    num_gpus_per_inference_pod: int = Field(ge=1)
    gpu_offset: int = Field(ge=0)

    @property
    def gpus_per_inference_cell(self) -> int:
        return self.num_pods_per_inference_cell * self.num_gpus_per_inference_pod

    @property
    def total_inference_gpus(self) -> int:
        return self.num_inference_cells * self.num_pods_per_inference_cell * self.num_gpus_per_inference_pod

    @property
    def total_trainer_gpus(self) -> int:
        return self.num_trainer_cells * self.num_pods_per_trainer_cell * self.num_gpus_per_node

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
        assert self.gpu_offset % self.gpus_per_inference_cell == 0, (
            f"An inference pool starting at gpu {self.gpu_offset} is not a whole number of its own "
            f"{self.gpus_per_inference_cell}-gpu cells in, so its cells would straddle trainer cells"
        )
        assert self.num_gpus_per_node % self.num_gpus_per_inference_pod == 0, (
            f"An inference pod of {self.num_gpus_per_inference_pod} gpus does not divide a "
            f"{self.num_gpus_per_node}-gpu node, so one of them would straddle two trainer pods"
        )
        assert self.gpu_offset + self.total_inference_gpus <= self.total_trainer_gpus, (
            f"{self.num_inference_cells} inference cells of {self.num_gpus_per_inference_pod} gpus starting at gpu "
            f"{self.gpu_offset} do not fit in the trainer's {self.total_trainer_gpus} gpus; an inference worker "
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

    @model_validator(mode="after")
    def _assert_pools_claim_distinct_gpus(self) -> PairingConfig:
        owner_of_gpu: dict[int, str] = {}
        for pool in self.inference_pools:
            span = range(pool.layout.gpu_offset, pool.layout.gpu_offset + pool.layout.total_inference_gpus)
            for gpu in span:
                assert gpu not in owner_of_gpu, (
                    f"'{pool.pool_id}' and '{owner_of_gpu[gpu]}' both claim the trainer's gpu {gpu}; one of them "
                    f"would be pinned to a node whose gpus the other already holds"
                )
                owner_of_gpu[gpu] = pool.pool_id
        return self
