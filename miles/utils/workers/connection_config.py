import json
from pydantic import Field

from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.worker_spec import BaseServeSpec, BaseSpec, PortInfo, StaticMeta

WORKER_METADATA_ANNOTATION = "miles.radixark.io/worker-metadata"


class StaticPoolConnInfo(FrozenStrictBaseModel):
    name: str
    port_infos: list[PortInfo]
    worker_class: str | None
    num_cells: int
    num_workers_per_cell: int
    pods_per_cell: int


class StaticConnConfig(FrozenStrictBaseModel):
    static_conn_infos: dict[str, StaticPoolConnInfo] = Field(default_factory=dict)


class WorkerPodMetadata(FrozenStrictBaseModel):
    category: str | None = None
    workers_per_pod: int = Field(gt=0)
    pods_per_cell: int = Field(gt=0)
    gpu_slots_per_worker: int = Field(ge=0)
    dynamic_pool: bool
    worker_class: str | None
    port_infos: list[PortInfo]
    static_meta: StaticMeta


def build_static_conn_config(*, specs: list[BaseSpec], scaling: ScalingConfig) -> StaticConnConfig:
    return StaticConnConfig(
        static_conn_infos={
            spec.name: StaticPoolConnInfo(
                name=spec.name,
                port_infos=spec.port_infos,
                worker_class=spec.worker_class if isinstance(spec, BaseServeSpec) else None,
                num_cells=scheduling.num_cells,
                num_workers_per_cell=scheduling.num_workers_per_cell,
                pods_per_cell=scheduling.pods_per_cell(),
            )
            for spec in specs
            if not (scheduling := spec.scheduling(scaling)).declares_dynamic_pool()
        }
    )


def build_worker_annotations(*, spec: BaseSpec, scaling: ScalingConfig) -> dict[str, str]:
    scheduling = spec.scheduling(scaling)
    metadata = WorkerPodMetadata(
        category=spec.category,
        workers_per_pod=scheduling.workers_per_pod(),
        pods_per_cell=scheduling.pods_per_cell(),
        gpu_slots_per_worker=scheduling.num_gpu_slots_per_worker,
        dynamic_pool=scheduling.declares_dynamic_pool(),
        worker_class=spec.worker_class if isinstance(spec, BaseServeSpec) else None,
        port_infos=spec.port_infos,
        static_meta=spec.static_meta,
    )
    return {
        WORKER_METADATA_ANNOTATION: json.dumps(metadata.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    }
