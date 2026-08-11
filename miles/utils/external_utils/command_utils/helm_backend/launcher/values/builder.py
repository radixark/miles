from __future__ import annotations

from miles.utils.external_utils.command_utils.helm_backend import naming
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.helm_values_types import (
    MilesRunChartValues,
    ObjectNames,
    OrchestratorSection,
    PoolEntry,
    RunValues,
)
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import (
    INFERENCE_ENGINES_SECTION,
    SECTION_OF_CATEGORY,
    STATIC_WORKERS_SECTION,
    TRAINER_ENGINES_SECTION,
    LaunchPlan,
    MooncakeInfo,
)
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.pool_entry import build_entry
from miles.utils.workers.naming import compute_cell_id
from miles.utils.workers.worker_spec import RPC_PORT_NAME, BaseWorkerSpec, NamedHostAndPorts, ServeWorkerSpec

def build_values(specs: list[BaseWorkerSpec], plan: LaunchPlan) -> MilesRunChartValues:
    return MilesRunChartValues(run=_build_run_values(specs, plan))


def _build_run_values(specs: list[BaseWorkerSpec], plan: LaunchPlan) -> RunValues:
    assert plan.orchestrator_command, (
        "Nothing but the orchestrator command starts the training run, so a run rendered without one would "
        "install every pod of the run and then sit there forever"
    )
    specs = _deployed_specs(specs)
    for spec in specs:
        if isinstance(spec, ServeWorkerSpec):
            _assert_worker_ports_fit(spec)
    addresses = _compute_addresses(specs, plan.release)

    entries: dict[str, list[PoolEntry]] = {
        STATIC_WORKERS_SECTION: [],
        INFERENCE_ENGINES_SECTION: [],
        TRAINER_ENGINES_SECTION: [],
    }
    for spec in specs:
        entries[SECTION_OF_CATEGORY[spec.category]].append(build_entry(spec, plan=plan, addresses=addresses))

    return RunValues(
        id=plan.run_id,
        state_file=plan.state_file,
        object_names=_object_names(plan.release),
        orchestrator=OrchestratorSection(command=plan.orchestrator_command),
        static_workers=entries[STATIC_WORKERS_SECTION],
        inference_engines=entries[INFERENCE_ENGINES_SECTION],
        trainer_engines=entries[TRAINER_ENGINES_SECTION],
        env=dict(plan.env) or None,
    )


def _object_names(release: str) -> ObjectNames:
    return ObjectNames(
        orchestrator=naming.component_name(release, naming.ORCHESTRATOR_COMPONENT),
        mooncake_master=MooncakeInfo.master_object_name(release),
    )


def _deployed_specs(specs: list[BaseWorkerSpec]) -> list[BaseWorkerSpec]:
    return [spec for spec in specs if spec.scheduling.num_cells > 0]


def _compute_addresses(specs: list[BaseWorkerSpec], release: str) -> dict[str, dict[str, NamedHostAndPorts]]:
    return {
        spec.name: {
            compute_cell_id(pool_id=spec.name, cell_index=cell_index): naming.static_cell_addrs(
                spec=spec, release=release, cell_index=cell_index
            )
            for cell_index in range(spec.scheduling.num_cells)
        }
        for spec in specs
        if SECTION_OF_CATEGORY[spec.category] == STATIC_WORKERS_SECTION
    }


def _assert_worker_ports_fit(spec: ServeWorkerSpec) -> None:
    workers_per_pod = spec.scheduling.workers_per_pod()
    rpc_port = next(port.static_port for port in spec.port_infos if port.name == RPC_PORT_NAME)
    for port in spec.port_infos:
        if port.name == RPC_PORT_NAME:
            continue
        assert rpc_port + workers_per_pod <= port.static_port or port.static_port + port.num_consecutive <= rpc_port, (
            f"Spec '{spec.name}' serves {workers_per_pod} workers per pod from {RPC_PORT_NAME} port {rpc_port} "
            f"upwards, which reaches into the {port.num_consecutive} port(s) '{port.name}' claims from "
            f"{port.static_port}"
        )
