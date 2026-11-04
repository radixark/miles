import pytest
from tests.fast.utils.external_utils.command_utils.helm_backend.launcher.values.utils import (
    LAYOUT,
    build_values_as_launched,
)
from tests.fast.utils.workers.scaling_fixtures import _ScalingCommandSpec

from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.workers.connection_config import (
    WORKER_METADATA_ANNOTATION,
    WorkerPodMetadata,
    build_static_conn_config,
    build_worker_annotations,
)
from miles.utils.workers.ray_worker_manager import RayWorkerManager, _PoolManager
from miles.utils.workers.worker_spec import SchedulingSpec


@pytest.mark.parametrize("cells,workers", [(2, 3), (5, 7)])
def test_connection_metadata_helm_and_ray_use_the_supplied_scaling(cells: int, workers: int) -> None:
    """Every scheduling consumer forwards the supplied layout instead of substituting a default or cached schedule."""
    scaling = ScalingConfig.model_construct(actor_num_nodes=cells, actor_num_gpus_per_node=workers)
    spec = _ScalingCommandSpec(
        name="dynamic-layout", port_infos=[], fixed_scheduling=SchedulingSpec.single(0), command=lambda _ctx: "worker"
    )

    connections = build_static_conn_config(specs=[spec], scaling=scaling)
    annotations = build_worker_annotations(spec=spec, scaling=scaling)
    metadata = WorkerPodMetadata.model_validate_json(annotations[WORKER_METADATA_ANNOTATION])
    values = build_values_as_launched([spec], LAYOUT, scaling=scaling).as_values()
    manager = object.__new__(RayWorkerManager)
    manager.scaling = scaling
    pool = _PoolManager.initial(spec=spec, manager=manager)

    assert connections.static_conn_infos[spec.name].num_cells == cells
    assert connections.static_conn_infos[spec.name].num_workers_per_cell == workers
    assert metadata.workers_per_pod == workers
    assert values["run"]["staticWorkers"][0]["replicas"] == cells
    assert [cell.cell_index for cell in pool.cells] == list(range(cells))
    assert all(cell.actors is None for cell in pool.cells)
