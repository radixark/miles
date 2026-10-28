import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.backends.sglang_utils.sglang_scaling_config import ServerGroupScalingConfig, SglangScalingConfig
from miles.ray.specs.inference import InferenceEngineSpec
from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.workers.connection_config import (
    WORKER_METADATA_ANNOTATION,
    WorkerPodMetadata,
    build_worker_annotations,
)
from miles.utils.workers.ray_worker_manager import RayWorkerManager, _CellManager


class TestInferenceScalingMetadata:
    @pytest.mark.parametrize("current_gpus", [16, 64])
    def test_retained_spec_exports_current_gpu_offsets_to_ray_and_kubernetes(self, current_gpus: int) -> None:
        """Both providers publish the current layout after a retained engine pool grows or shrinks."""
        args = make_args(rollout_num_gpus=32, rollout_num_gpus_per_engine=16)
        (spec,) = InferenceEngineSpec.create(args)
        manager = RayWorkerManager()
        manager.scaling = ScalingConfig(sglang_scaling=args.sglang_scaling)
        cell = _CellManager(manager=manager, cell_index=0, spec=spec, actors=None)
        original = cell.get_info().meta
        manager.scaling = ScalingConfig(
            sglang_scaling=SglangScalingConfig(
                groups={"default": [ServerGroupScalingConfig(num_gpus=current_gpus, gpu_offset=48, engine_offset=7)]}
            )
        )

        metadata = WorkerPodMetadata.model_validate_json(
            build_worker_annotations(spec=spec, scaling=manager.scaling)[WORKER_METADATA_ANNOTATION]
        )

        assert original["gpu_offset"] == 0
        assert cell.get_info().meta["gpu_offset"] == 48
        assert metadata.static_meta.resolve(cell_index=0)["gpu_offset"] == 48
        assert metadata.static_meta.resolve(cell_index=1)["gpu_offset"] == 64
        assert metadata.static_meta.resolve(cell_index=0) == cell.get_info().meta
        assert (metadata.workers_per_pod, metadata.pods_per_cell, metadata.gpu_slots_per_worker) == (1, 2, 8)
        assert spec.scheduling(manager.scaling).num_cells == current_gpus // 16
        assert original["gpu_offset"] == 0
