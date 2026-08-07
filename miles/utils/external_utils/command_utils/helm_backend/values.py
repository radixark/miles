from __future__ import annotations

import shlex
import sys
from typing import Any

from miles.utils.external_utils.command_utils.helm_backend import mooncake, naming, staging
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers import colocate_matching
from miles.utils.workers.naming import compute_cell_id
from miles.utils.workers.worker_provider.kubernetes.helm.labels import DEFAULT_LABEL_KEYS
from miles.utils.workers.worker_provider.kubernetes.helm.naming import static_cell_addrs
from miles.utils.workers.worker_spec import (
    BaseWorkerSpec,
    CommandWorkerSpec,
    HostAndPort,
    LaunchCommandContext,
    NamedHostAndPorts,
    ServeWorkerSpec,
    assert_rank_ports_fit,
)

WORKER_INDEX_PLACEHOLDER = "$(LWS_WORKER_INDEX)"
LEADER_ADDRESS_PLACEHOLDER = "$(LWS_LEADER_ADDRESS)"

ORCHESTRATOR_COMPONENT = "orchestrator"
COLOCATE_PAIRING_COMPONENT = "colocate-pairing"

_BIND_HOST = "0.0.0.0"

_SERVE_MODULE = "miles.utils.workers.serving.serve"
_SUPERVISOR_MODULE = "miles.utils.workers.process_supervisor"
_SPECS_FN = "miles.ray.specs.entrypoint.compute_specs_from_argv"

# A values entry describes a whole pool, so a command is rendered as if for its first cell. A
# per-cell fact has to reach the pod some other way; MILES_CELL_INDEX carries the real one.
_RENDERED_CELL_INDEX = 0
_WORKER_INDEX_SENTINEL = 987654321
# One command is rendered for every cell of a pool, so no real cell id exists yet; a spec that needs its
# own is served, so its own process reads it off the pod instead of off the command.


class RunLayout(FrozenStrictBaseModel):
    run_id: str
    release: str
    orchestrator_command: list[str]
    worker_argv: list[str]
    env: dict[str, str] = {}
    colocate: bool = False
    uses_mooncake: bool = False
    mooncake_port: int = 0
    stage_to_local: tuple[str, ...] = ()
    node_local_root: str = ""


def build_values(specs: list[BaseWorkerSpec], layout: RunLayout) -> dict[str, Any]:
    specs = _deployed_specs(specs)
    for spec in specs:
        if isinstance(spec, ServeWorkerSpec):
            assert_rank_ports_fit(spec)
    addresses = _decide_addresses(specs, layout.release)

    run: dict[str, Any] = {
        "id": layout.run_id,
        "objectNames": object_names(layout.release),
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
        run["colocate"] = colocate_section(specs)
    return {"run": run}


def object_names(release: str) -> dict[str, str]:
    return {
        "orchestrator": naming.component_name(release, ORCHESTRATOR_COMPONENT),
        "mooncakeMaster": mooncake.master_object_name(release),
        "colocatePairing": naming.component_name(release, COLOCATE_PAIRING_COMPONENT),
    }


def _deployed_specs(specs: list[BaseWorkerSpec]) -> list[BaseWorkerSpec]:
    return [spec for spec in specs if spec.scheduling.num_cells > 0]


def _mooncake_section(layout: RunLayout) -> dict[str, Any]:
    section: dict[str, Any] = {"enabled": True}
    if layout.mooncake_port:
        section["rpcPort"] = layout.mooncake_port
    return section


def colocate_section(specs: list[BaseWorkerSpec]) -> dict[str, Any]:
    engines = [spec for spec in specs if section_of(spec) == "inferenceEngines"]
    trainers = [spec for spec in specs if section_of(spec) == "trainers"]
    assert len(trainers) == 1, (
        f"colocate pins engines onto one trainer pool_id's nodes, but this run has {len(trainers)} trainer "
        f"pool_ids; which one an engine belongs beside would be undefined"
    )

    trainer = trainers[0]
    engine = _colocated_engine(engines)
    colocate_matching.assert_colocate_supported(
        layout=pairing_layout(engine=engine, trainer=trainer),
        gpus_per_engine_pod=engine.scheduling.gpus_per_pod(),
        gpus_per_trainer_pod=trainer.scheduling.gpus_per_pod(),
        gpus_per_node=trainer.scheduling.num_gpus_per_node,
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


def pairing_layout(*, engine: BaseWorkerSpec, trainer: BaseWorkerSpec) -> colocate_matching.PairingLayout:
    return colocate_matching.PairingLayout(
        engine_cells=engine.scheduling.num_cells,
        trainer_cells=trainer.scheduling.num_cells,
        pods_per_engine_cell=engine.scheduling.pods_per_cell(),
        pods_per_trainer_cell=trainer.scheduling.pods_per_cell(),
    )


def section_of(spec: BaseWorkerSpec) -> str:
    if spec.scheduling.gpus_per_cell() == 0:
        return "staticWorkers"
    return "trainers" if isinstance(spec, ServeWorkerSpec) else "inferenceEngines"


def _decide_addresses(specs: list[BaseWorkerSpec], release: str) -> dict[str, dict[str, NamedHostAndPorts]]:
    return {
        spec.name: {
            compute_cell_id(pool_id=spec.name, cell_index=cell_index): static_cell_addrs(
                spec=spec, release=release, cell_index=cell_index
            )
            for cell_index in range(spec.scheduling.num_cells)
        }
        for spec in specs
        if section_of(spec) == "staticWorkers"
    }


def _build_entry(
    spec: BaseWorkerSpec, layout: RunLayout, addresses: dict[str, dict[str, NamedHostAndPorts]]
) -> dict[str, Any]:
    assert spec.scheduling.num_cells > 0, (
        f"spec '{spec.name}' asks for {spec.scheduling.num_cells} cells; a spec a run has turned off is dropped "
        f"before conversion, because a values entry always renders at least one pod"
    )
    context = _launch_context(spec, addresses=addresses)
    pods_per_cell = spec.scheduling.pods_per_cell()

    entry: dict[str, Any] = {
        "name": spec.name,
        "objectName": naming.component_name(layout.release, spec.name),
        "poolId": spec.name,
        "command": staging.with_staging(
            _command_of(spec, context, layout=layout),
            layout.stage_to_local if _stages_inputs(spec) else (),
            node_local_root=layout.node_local_root,
        ),
        "ports": [{"name": _port_name(port.name), "port": port.static_port} for port in spec.port_infos],
    }
    if env := _command_env_of(spec, addresses=addresses):
        entry["env"] = env
    if meta := _meta_of(spec):
        entry["meta"] = meta
    entry["replicas"] = spec.scheduling.num_cells
    if pods_per_cell > 1:
        entry["size"] = pods_per_cell
    if gpus := spec.scheduling.gpus_per_pod():
        entry["resources"] = {"limits": {"nvidia.com/gpu": gpus}}
    return entry


def _command_env_of(spec: BaseWorkerSpec, *, addresses: dict[str, dict[str, NamedHostAndPorts]]) -> dict[str, str]:
    if isinstance(spec, ServeWorkerSpec):
        return {}

    first = dict(spec.env_var(_launch_context(spec, addresses=addresses)))
    second = dict(spec.env_var(_launch_context(spec, addresses=addresses, cell_index=1, worker_in_cell_index=1)))
    assert first == second, (
        f"spec '{spec.name}' builds its environment out of the cell and rank it is given, but a values entry "
        f"describes a whole pool and is rendered before any of them exists; serve the spec so its pod can "
        f"compute the environment itself, or drop the dependency"
    )
    return first


def _stages_inputs(spec: BaseWorkerSpec) -> bool:
    return section_of(spec) == "trainers"


def _meta_of(spec: BaseWorkerSpec) -> dict[str, str]:
    gpus_per_pod = spec.scheduling.gpus_per_pod()
    if not gpus_per_pod:
        return {}
    return {DEFAULT_LABEL_KEYS.gpu_ids_meta: ",".join(str(gpu_id) for gpu_id in range(gpus_per_pod))}


def _launch_context(
    spec: BaseWorkerSpec,
    addresses: dict[str, dict[str, NamedHostAndPorts]],
    *,
    cell_index: int = _RENDERED_CELL_INDEX,
    worker_in_cell_index: int = _WORKER_INDEX_SENTINEL,
) -> LaunchCommandContext:
    self_addrs = {
        port.name: HostAndPort(
            host=LEADER_ADDRESS_PLACEHOLDER if port.mode == "master" else _BIND_HOST,
            port=port.static_port,
        )
        for port in spec.port_infos
    }
    return LaunchCommandContext(
        cell_index=cell_index,
        worker_in_cell_index=worker_in_cell_index,
        gpu_ids=list(range(max(1, spec.scheduling.gpus_per_pod()))),
        self_addrs=self_addrs,
        spec_addrs={pool_id: list(cells.values()) for pool_id, cells in addresses.items()},
    )


def _command_of(spec: BaseWorkerSpec, context: LaunchCommandContext, layout: RunLayout) -> list[str]:
    if isinstance(spec, CommandWorkerSpec):
        return _with_worker_index(shlex.split(spec.launch_command(context)), spec)

    assert isinstance(spec, ServeWorkerSpec), spec
    ranks_per_pod = spec.scheduling.ranks_per_pod()
    serve = [
        sys.executable,
        "-m",
        _SERVE_MODULE,
        "--specs",
        _SPECS_FN,
        "--pool-id",
        spec.name,
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
