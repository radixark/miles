from __future__ import annotations

import shlex
import sys
from typing import Any

from miles.utils.external_utils.command_utils.helm_backend import naming, staging
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers import colocate_matching
from miles.utils.workers.naming import compute_cell_id
from miles.utils.workers.worker_provider.kubernetes.helm.labels import GPU_IDS_META
from miles.utils.workers.worker_spec import (
    BaseWorkerSpec,
    CommandWorkerSpec,
    HostAndPort,
    LaunchCommandContext,
    NamedHostAndPorts,
    ServeWorkerSpec,
)

WORKER_INDEX_PLACEHOLDER = "$(LWS_WORKER_INDEX)"
LEADER_ADDRESS_PLACEHOLDER = "$(LWS_LEADER_ADDRESS)"

_BIND_HOST = "0.0.0.0"

_SERVE_MODULE = "miles.utils.workers.serving.serve"
_SUPERVISOR_MODULE = "miles.utils.workers.process_supervisor"
_CTOR_KWARGS_FN = "miles.ray.specs.bootstrap.compute_ctor_kwargs"

_WORKER_INDEX_SENTINEL = 987654321


class RunLayout(FrozenStrictBaseModel):
    run_id: str
    release: str
    orchestrator_command: list[str]
    worker_argv: list[str]
    num_gpus_per_node: int
    env: dict[str, str] = {}
    colocate: bool = False
    uses_mooncake: bool = False
    mooncake_port: int = 0
    stage_to_local: tuple[str, ...] = ()
    node_local_root: str = ""


def build_values(specs: list[BaseWorkerSpec], layout: RunLayout) -> dict[str, Any]:
    specs = _deployed_specs(specs)
    addresses = _predict_addresses(specs, layout.release)

    run: dict[str, Any] = {
        "id": layout.run_id,
        "orchestrator": {"command": layout.orchestrator_command},
        "staticWorkers": [],
        "inferenceEngines": [],
        "trainers": [],
    }
    if layout.env:
        run["env"] = dict(layout.env)
    if layout.uses_mooncake:
        run["mooncake"] = _mooncake_section(layout)
    for spec in specs:
        run[section_of(spec)].append(_build_entry(spec, layout=layout, addresses=addresses))
    if layout.colocate:
        run["colocate"] = colocate_section(specs, layout=layout)
    return {"run": run}


def _deployed_specs(specs: list[BaseWorkerSpec]) -> list[BaseWorkerSpec]:
    return [spec for spec in specs if spec.scheduling.num_cells > 0]


def _mooncake_section(layout: RunLayout) -> dict[str, Any]:
    section: dict[str, Any] = {"enabled": True}
    if layout.mooncake_port:
        section["rpcPort"] = layout.mooncake_port
    return section


def colocate_section(specs: list[BaseWorkerSpec], *, layout: RunLayout) -> dict[str, Any]:
    engines = [spec for spec in specs if section_of(spec) == "inferenceEngines"]
    trainers = [spec for spec in specs if section_of(spec) == "trainers"]
    assert len(trainers) == 1, (
        f"colocate pins engines onto one trainer pool_id's nodes, but this run has {len(trainers)} trainer "
        f"pool_ids; which one an engine belongs beside would be undefined"
    )

    trainer = trainers[0]
    engine = _colocated_engine(engines)
    colocate_matching.assert_colocate_supported(
        layout=pairing_layout(engine=engine, trainer=trainer, num_gpus_per_node=layout.num_gpus_per_node),
        gpus_per_engine_pod=min(gpus_per_cell_of(engine), layout.num_gpus_per_node),
        gpus_per_trainer_pod=min(gpus_per_cell_of(trainer), layout.num_gpus_per_node),
        gpus_per_node=layout.num_gpus_per_node,
    )
    return {"enabled": True, "enginePool": engine.name, "trainerPool": trainer.name}


def _colocated_engine(engines: list[BaseWorkerSpec]) -> BaseWorkerSpec:
    declared = [engine for engine in engines if engine.scheduling.colocate_with_trainer]
    assert len(declared) == 1, (
        f"colocate pins one engine pool_id onto the trainer pool_id's nodes, but "
        f"{[engine.name for engine in declared]} of {[engine.name for engine in engines]} say they are that "
        f"pool_id; name it with --colocate-engine-pool, "
        f"because with prefill/decode disaggregation several pool_ids share the trainer's gpus and only the "
        f"named one is paired cell for cell"
    )
    return declared[0]


def pairing_layout(
    *, engine: BaseWorkerSpec, trainer: BaseWorkerSpec, num_gpus_per_node: int
) -> colocate_matching.PairingLayout:
    return colocate_matching.PairingLayout(
        engine_cells=engine.scheduling.num_cells,
        trainer_cells=trainer.scheduling.num_cells,
        pods_per_engine_cell=pods_per_cell_of(engine, num_gpus_per_node),
        pods_per_trainer_cell=pods_per_cell_of(trainer, num_gpus_per_node),
    )


def section_of(spec: BaseWorkerSpec) -> str:
    if gpus_per_cell_of(spec) == 0:
        return "staticWorkers"
    return "trainers" if isinstance(spec, ServeWorkerSpec) else "inferenceEngines"


def gpus_per_cell_of(spec: BaseWorkerSpec) -> int:
    return spec.scheduling.num_workers_per_cell * spec.scheduling.num_gpu_slots_per_worker


def pods_per_cell_of(spec: BaseWorkerSpec, num_gpus_per_node: int) -> int:
    gpus_per_cell = gpus_per_cell_of(spec)
    if gpus_per_cell <= num_gpus_per_node:
        return 1
    assert gpus_per_cell % num_gpus_per_node == 0, (
        f"spec '{spec.name}' wants {gpus_per_cell} gpus per cell, which is not a whole number of "
        f"{num_gpus_per_node}-gpu nodes"
    )
    return gpus_per_cell // num_gpus_per_node


def _predict_addresses(specs: list[BaseWorkerSpec], release: str) -> dict[str, list[NamedHostAndPorts]]:
    return {
        spec.name: [
            {
                port.name: HostAndPort(host=_cell_host(spec, release, cell_index), port=port.static_port)
                for port in spec.port_infos
            }
            for cell_index in range(spec.scheduling.num_cells)
        ]
        for spec in specs
    }


def _cell_host(spec: BaseWorkerSpec, release: str, cell_index: int) -> str:
    if section_of(spec) == "staticWorkers":
        return naming.static_worker_host(release, spec.name, cell_index)
    return naming.cell_leader_host(release, spec.name, cell_index)


def _build_entry(
    spec: BaseWorkerSpec, layout: RunLayout, addresses: dict[str, list[NamedHostAndPorts]]
) -> dict[str, Any]:
    assert spec.scheduling.num_cells > 0, (
        f"spec '{spec.name}' asks for {spec.scheduling.num_cells} cells; a spec a run has turned off is dropped "
        f"before conversion, because a values entry always renders at least one pod"
    )
    context = _launch_context(spec, layout=layout, addresses=addresses)
    pods_per_cell = pods_per_cell_of(spec, layout.num_gpus_per_node)

    entry: dict[str, Any] = {
        "name": spec.name,
        "poolId": spec.name,
        "command": staging.with_staging(
            _command_of(spec, context, layout=layout, pods_per_cell=pods_per_cell),
            layout.stage_to_local if _stages_inputs(spec) else (),
            node_local_root=layout.node_local_root,
        ),
        "ports": [{"name": _port_name(port.name), "port": port.static_port} for port in spec.port_infos],
    }
    if env := dict(spec.env_var(context)):
        entry["env"] = env
    if meta := _meta_of(spec, layout=layout):
        entry["meta"] = meta
    entry["replicas"] = spec.scheduling.num_cells
    if pods_per_cell > 1:
        entry["size"] = pods_per_cell
    if gpus := min(gpus_per_cell_of(spec), layout.num_gpus_per_node):
        entry["resources"] = {"limits": {"nvidia.com/gpu": gpus}}
    return entry


def _stages_inputs(spec: BaseWorkerSpec) -> bool:
    return section_of(spec) == "trainers"


def _meta_of(spec: BaseWorkerSpec, *, layout: RunLayout) -> dict[str, str]:
    gpus_per_pod = min(gpus_per_cell_of(spec), layout.num_gpus_per_node)
    if not gpus_per_pod:
        return {}
    return {GPU_IDS_META: ",".join(str(gpu_id) for gpu_id in range(gpus_per_pod))}


def _launch_context(
    spec: BaseWorkerSpec, layout: RunLayout, addresses: dict[str, list[NamedHostAndPorts]]
) -> LaunchCommandContext:
    self_addrs = {
        port.name: HostAndPort(
            host=LEADER_ADDRESS_PLACEHOLDER if port.mode == "master" else _BIND_HOST,
            port=port.static_port,
        )
        for port in spec.port_infos
    }
    return LaunchCommandContext(
        cell_id=compute_cell_id(pool_id=spec.name, cell_index=0),
        cell_ordinal=0,
        worker_in_cell_index=_WORKER_INDEX_SENTINEL,
        gpu_ids=list(range(max(1, min(gpus_per_cell_of(spec), layout.num_gpus_per_node)))),
        self_addrs=self_addrs,
        spec_addrs=addresses,
    )


def _command_of(
    spec: BaseWorkerSpec, context: LaunchCommandContext, layout: RunLayout, pods_per_cell: int
) -> list[str]:
    if isinstance(spec, CommandWorkerSpec):
        return _with_worker_index(shlex.split(spec.launch_command(context)), spec)

    assert isinstance(spec, ServeWorkerSpec), spec
    ranks_per_pod = min(spec.scheduling.num_workers_per_cell, layout.num_gpus_per_node)
    serve = [
        sys.executable,
        "-m",
        _SERVE_MODULE,
        "--worker",
        spec.worker_class,
        "--pool-id",
        spec.name,
        "--ctor-kwargs-fn",
        _CTOR_KWARGS_FN,
        "--ranks-per-pod",
        str(ranks_per_pod),
        "--gpu-slots-per-rank",
        str(spec.scheduling.num_gpu_slots_per_worker),
        "--",
    ] + layout.worker_argv
    if ranks_per_pod == 1:
        return serve
    return [sys.executable, "-m", _SUPERVISOR_MODULE, "--num-subprocesses", str(ranks_per_pod), "--"] + serve


def _with_worker_index(argv: list[str], spec: BaseWorkerSpec) -> list[str]:
    sentinel = str(_WORKER_INDEX_SENTINEL)
    embedded = [argument for argument in argv if sentinel in argument and argument != sentinel]
    assert not embedded, (
        f"spec '{spec.name}' builds {embedded} out of its node rank; kubelet substitutes a whole "
        f"argument, so the rank has to reach the command unchanged"
    )
    return [WORKER_INDEX_PLACEHOLDER if argument == sentinel else argument for argument in argv]


def _port_name(name: str) -> str:
    return name.replace("_", "-")[:15]
