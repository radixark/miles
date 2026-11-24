from __future__ import annotations

from pydantic import Field, model_validator

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class PairingLayout(FrozenStrictBaseModel):
    num_inference_cells: int = Field(ge=1)
    num_trainer_cells: int = Field(ge=1)
    num_pods_per_inference_cell: int = Field(ge=1)
    num_pods_per_trainer_cell: int = Field(ge=1)
    num_gpus_per_node: int = Field(ge=1)
    num_gpus_per_trainer_pod: int | None = Field(default=None, ge=1)
    num_gpus_per_inference_pod: int = Field(ge=1)
    gpu_offset: int = Field(ge=0)

    @property
    def gpus_per_trainer_pod(self) -> int:
        return self.num_gpus_per_node if self.num_gpus_per_trainer_pod is None else self.num_gpus_per_trainer_pod

    @property
    def gpus_per_inference_cell(self) -> int:
        return self.num_pods_per_inference_cell * self.num_gpus_per_inference_pod

    @property
    def total_inference_gpus(self) -> int:
        return self.num_inference_cells * self.num_pods_per_inference_cell * self.num_gpus_per_inference_pod

    @property
    def total_trainer_gpus(self) -> int:
        return self.num_trainer_cells * self.num_pods_per_trainer_cell * self.gpus_per_trainer_pod

    @model_validator(mode="after")
    def _assert_inference_pods_pair(self) -> PairingLayout:
        assert self.gpus_per_trainer_pod <= self.num_gpus_per_node, (
            f"A trainer pod of {self.gpus_per_trainer_pod} gpus exceeds a {self.num_gpus_per_node}-gpu node"
        )
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
        assert self.gpus_per_trainer_pod % self.num_gpus_per_inference_pod == 0, (
            f"An inference pod of {self.num_gpus_per_inference_pod} gpus does not divide a "
            f"{self.gpus_per_trainer_pod}-gpu trainer pod, so one of them would straddle two trainer pods"
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
    pairing_dir: str | None = None

    @model_validator(mode="after")
    def _assert_pools_claim_distinct_gpus(self) -> PairingConfig:
        if self.pairing_dir is not None:
            assert self.pairing_dir.startswith("/"), (
                f"pairing_dir {self.pairing_dir!r} must be an absolute path shared by trainer pods and the pairing "
                f"controller"
            )
        trainer_pod_widths = {pool.layout.gpus_per_trainer_pod for pool in self.inference_pools}
        assert len(trainer_pod_widths) == 1, (
            f"Inference pools disagree on the trainer pod width: {sorted(trainer_pod_widths)}"
        )
        if any(pool.layout.gpus_per_trainer_pod < pool.layout.num_gpus_per_node for pool in self.inference_pools):
            assert self.pairing_dir is not None, (
                "Sub-node trainer pods require pairing_dir so the controller can pair each inference pod with the "
                "physical GPU UUIDs assigned to its trainer pod"
            )
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
