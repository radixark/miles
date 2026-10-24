import json

import pytest
from tests.fast.utils.workers.fake_specs import FakeCommandSpec, FakeServeSpec

from miles.backends.sglang_utils.sglang_scaling_config import SglangScalingConfig
from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.workers.connection_config import (
    WORKER_METADATA_ANNOTATION,
    WorkerPodMetadata,
    build_static_conn_config,
    build_worker_annotations,
)
from miles.utils.workers.worker_spec import PortInfo, SchedulingSpec, StaticMeta


class TestStaticConnectionConfig:
    def test_only_static_pools_are_embedded_in_the_launcher_payload(self) -> None:
        """GPU-slot pools must be discovered dynamically even when they request fractional GPUs."""
        scaling = ScalingConfig(sglang_scaling=SglangScalingConfig(groups={}))
        fixed = FakeServeSpec(name="fixed", worker_class="worker.Fixed", fixed_scheduling=SchedulingSpec.single(0))
        dynamic = FakeServeSpec(
            name="dynamic",
            worker_class="worker.Dynamic",
            fixed_scheduling=SchedulingSpec(
                num_cells=3,
                num_workers_per_cell=2,
                num_gpus_per_worker=0.25,
                num_gpu_slots_per_worker=1,
                num_gpus_per_node=8,
            ),
        )

        result = build_static_conn_config(specs=[fixed, dynamic], scaling=scaling)

        assert set(result.static_conn_infos) == {"fixed"}
        assert result.static_conn_infos["fixed"].worker_class == "worker.Fixed"
        assert result.static_conn_infos["fixed"].num_cells == 1

    def test_command_pools_have_no_served_worker_class(self) -> None:
        """A static command pool must not advertise a callable RPC class."""
        spec = FakeCommandSpec(
            name="command",
            port_infos=[PortInfo(name="http", static_port=9000)],
            fixed_scheduling=SchedulingSpec.single(0),
            command=str,
        )
        result = build_static_conn_config(
            specs=[spec], scaling=ScalingConfig(sglang_scaling=SglangScalingConfig(groups={}))
        )

        assert result.static_conn_infos["command"].worker_class is None
        assert result.static_conn_infos["command"].port_infos[0].static_port == 9000


class TestWorkerAnnotations:
    def test_multinode_metadata_describes_each_pod_instead_of_the_whole_cell(self) -> None:
        """A two-node cell advertises each pod's worker count while preserving cell topology."""
        spec = FakeServeSpec(
            name="trainer",
            category="training",
            worker_class="worker.Trainer",
            fixed_scheduling=SchedulingSpec(
                num_cells=3,
                num_workers_per_cell=16,
                num_gpus_per_worker=1,
                num_gpu_slots_per_worker=1,
                num_gpus_per_node=8,
            ),
            partial_static_meta=StaticMeta(include_cell_index=True, gpu_offset_base=4, gpu_offset_stride_per_cell=16),
        )
        annotations = build_worker_annotations(
            spec=spec, scaling=ScalingConfig(sglang_scaling=SglangScalingConfig(groups={}))
        )
        metadata = WorkerPodMetadata.model_validate_json(annotations[WORKER_METADATA_ANNOTATION])

        assert metadata.workers_per_pod == 8
        assert metadata.pods_per_cell == 2
        assert metadata.gpu_slots_per_worker == 1
        assert metadata.dynamic_pool
        assert metadata.category == "training"
        assert metadata.worker_class == "worker.Trainer"
        assert metadata.static_meta.resolve(cell_index=2) == {"cell_index": 2, "gpu_offset": 36}
        assert json.loads(annotations[WORKER_METADATA_ANNOTATION])["port_infos"][0]["name"] == "rpc"

    @pytest.mark.parametrize(
        "field,value", [("workers_per_pod", 0), ("pods_per_cell", 0), ("gpu_slots_per_worker", -1)]
    )
    def test_invalid_pod_topology_is_rejected(self, field: str, value: int) -> None:
        """Malformed metadata cannot create empty pods or negative GPU reservations."""
        payload = dict(
            workers_per_pod=1,
            pods_per_cell=1,
            gpu_slots_per_worker=0,
            dynamic_pool=False,
            worker_class=None,
            port_infos=[],
            static_meta={},
        )
        with pytest.raises(ValueError):
            WorkerPodMetadata.model_validate(payload | {field: value})
