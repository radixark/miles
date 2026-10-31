from typing import Any

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.backends.sglang_utils.sglang_scaling_config import ServerGroupScalingConfig, SglangScalingConfig
from miles.ray.specs import inference
from miles.ray.specs.inference import InferenceEngineSpec
from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.workers.worker_spec import HostAndPort, LaunchCommandContext


class TestRuntimeInferenceScaling:
    @pytest.mark.parametrize(
        ("width", "cells", "workers", "gpu_offset", "seed"),
        [(1, 64, 1, 17, 114), (16, 4, 2, 32, 116)],
    )
    def test_runtime_layout_drives_schedule_metadata_and_launch(
        self,
        monkeypatch: pytest.MonkeyPatch,
        width: int,
        cells: int,
        workers: int,
        gpu_offset: int,
        seed: int,
    ) -> None:
        """A retained engine spec uses the new layout for placement, metadata and launch seeds."""
        args = make_args(rollout_num_gpus=32, rollout_num_gpus_per_engine=width, seed=100)
        (spec,) = InferenceEngineSpec.create(args)
        original = ScalingConfig(sglang_scaling=args.sglang_scaling)
        current = original.model_copy(
            update=dict(
                sglang_scaling=SglangScalingConfig(
                    groups={"default": [ServerGroupScalingConfig(num_gpus=64, gpu_offset=16, engine_offset=13)]}
                )
            )
        )
        captured: dict[str, Any] = {}

        def record_launch(**kwargs: Any) -> str:
            captured.update(kwargs)
            return "engine-command"

        monkeypatch.setattr(inference, "compute_engine_launch_cmd", record_launch)
        scheduling = spec.scheduling(current)
        ctx = LaunchCommandContext(
            args=args,
            scaling=current,
            cell_index=1,
            worker_in_cell_index=workers - 1,
            num_workers_per_cell=workers,
            gpu_ids=[0],
            local_gpu_ids=[0],
            self_addrs={
                name: HostAndPort(host="127.0.0.1", port=port)
                for name, port in dict(
                    primary=30000, dist_init=9000, nccl=10000, engine_info_bootstrap=12000, gate=13000
                ).items()
            },
            pool_addrs={},
        )

        assert (scheduling.num_cells, scheduling.num_workers_per_cell, scheduling.pg_slot_offset) == (
            cells,
            workers,
            16,
        )
        assert spec.compute_static_meta(current).resolve(cell_index=1)["gpu_offset"] == gpu_offset
        assert spec.launch_command(ctx) == "engine-command"
        assert captured["random_seed"] == seed
        assert captured["node_rank"] == workers - 1
        assert spec.scheduling(original).num_cells == 32 // width
        assert spec.compute_static_meta(original).resolve(cell_index=1)["gpu_offset"] == width
