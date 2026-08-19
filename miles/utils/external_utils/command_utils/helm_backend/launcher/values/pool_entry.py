from __future__ import annotations

import shlex

from miles.utils.external_utils.colocate_pairing.config import PairingLayout
from miles.utils.external_utils.command_utils.base_backend import TRAINER_ROLE
from miles.utils.external_utils.command_utils.helm_backend import naming
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.helm_values_types import (
    PoolEntry,
    PortEntry,
)
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import (
    SECTION_OF_CATEGORY,
    TRAINER_ENGINES_SECTION,
    LaunchPlan,
)
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.placeholders import (
    LEADER_ADDRESS_PLACEHOLDER,
    RENDERED_CELL_INDEX,
    WORKER_INDEX_SENTINEL,
    real_or_sentinel_gpu_ids,
    sentinels_to_placeholders,
)
from miles.utils.workers.argv_utils import python_argv_prefix
from miles.utils.workers.worker_provider.kubernetes.helm import env
from miles.utils.workers.worker_spec import (
    BaseWorkerSpec,
    CommandWorkerSpec,
    HostAndPort,
    LaunchCommandContext,
    NamedHostAndPorts,
    ServeWorkerSpec,
)

_BIND_HOST = "0.0.0.0"

_SERVE_MODULE = "miles.utils.workers.serving.serve"
_SUPERVISOR_MODULE = "miles.utils.workers.process_supervisor"
_SPECS_FN = "miles.ray.specs.entrypoint.compute_specs_from_argv"


def build_entry(
    spec: BaseWorkerSpec,
    plan: LaunchPlan,
    addresses: dict[str, dict[str, NamedHostAndPorts]],
    pairing_layout: PairingLayout | None = None,
) -> PoolEntry:
    assert spec.scheduling.num_cells > 0, (
        f"Spec '{spec.name}' asks for {spec.scheduling.num_cells} cells; a spec a run has turned off is dropped "
        f"before conversion, because a values entry always renders at least one pod"
    )
    is_sub_node = _is_sub_node(pairing_layout)
    context = _launch_context(
        spec,
        addresses=addresses,
        cell_index=RENDERED_CELL_INDEX,
        worker_in_cell_index=WORKER_INDEX_SENTINEL,
        is_sub_node=is_sub_node,
    )
    pods_per_cell = spec.scheduling.pods_per_cell()
    gpus_per_pod = spec.scheduling.gpus_per_pod()

    return PoolEntry(
        name=spec.name,
        object_name=naming.component_name(plan.release, spec.name),
        pool_id=spec.name,
        command=_with_prepare_cmd(_command_of_spec(spec, context, plan=plan), spec, plan=plan),
        ports=[PortEntry(name=_port_name(port.name), port=port.static_port) for port in spec.port_infos],
        env=_command_env_of_spec(spec, context, addresses=addresses, is_sub_node=is_sub_node) or None,
        meta=_meta_of_spec(spec) or None,
        service_account_name=(
            naming.component_name(plan.release, naming.ORCHESTRATOR_COMPONENT)
            if spec.needs_platform_read_permission
            else None
        ),
        replicas=spec.scheduling.num_cells,
        size=pods_per_cell if pods_per_cell > 1 else None,
        resources={"limits": {"nvidia.com/gpu": gpus_per_pod}} if gpus_per_pod else None,
        restart_at=plan.rendered_restart_at(spec.name),
    )


def _command_env_of_spec(
    spec: BaseWorkerSpec,
    context: LaunchCommandContext,
    *,
    addresses: dict[str, dict[str, NamedHostAndPorts]],
    is_sub_node: bool,
) -> dict[str, str]:
    if isinstance(spec, ServeWorkerSpec):
        return {}

    first = dict(spec.env_var(context))
    second = dict(
        spec.env_var(
            _launch_context(
                spec,
                addresses=addresses,
                cell_index=1,
                worker_in_cell_index=1,
                is_sub_node=is_sub_node,
            )
        )
    )
    assert first == second, (
        f"Spec '{spec.name}' builds its environment out of the cell and worker it is given, but a values entry "
        f"describes a whole pool and is rendered before any of them exists; serve the spec so its pod can "
        f"compute the environment itself, or drop the dependency"
    )
    return first


def _with_prepare_cmd(command: list[str], spec: BaseWorkerSpec, plan: LaunchPlan) -> list[str]:
    if SECTION_OF_CATEGORY[spec.category] != TRAINER_ENGINES_SECTION:
        return command
    prepare = plan.prepare_cmd.get(TRAINER_ROLE)
    if not prepare:
        return command

    assert spec.scheduling.gpus_per_pod() >= spec.scheduling.num_gpus_per_node, (
        f"A prepare command runs once per pod of '{spec.name}', but that pool takes "
        f"{spec.scheduling.gpus_per_pod()} of a node's {spec.scheduling.num_gpus_per_node} gpus, so two of its "
        f"pods can land on one node and run the command against the same node-local path at the same time; "
        f"give the pool whole nodes, or serialize the command yourself with flock"
    )
    return ["bash", "-c", f"{prepare} && exec {shlex.join(command)}"]


def _meta_of_spec(spec: BaseWorkerSpec) -> dict[str, str]:
    gpus_per_pod = spec.scheduling.gpus_per_pod()
    if not gpus_per_pod:
        return {}
    return {env.DEFAULT_LABEL_KEYS.gpu_ids_meta: ",".join(str(gpu_id) for gpu_id in range(gpus_per_pod))}


def _launch_context(
    spec: BaseWorkerSpec,
    addresses: dict[str, dict[str, NamedHostAndPorts]],
    *,
    cell_index: int,
    worker_in_cell_index: int,
    is_sub_node: bool = False,
) -> LaunchCommandContext:
    self_addrs = {
        port.name: HostAndPort(
            host=LEADER_ADDRESS_PLACEHOLDER if port.mode == "master" else _BIND_HOST,
            port=port.static_port,
        )
        for port in spec.port_infos
    }
    pod_gpu_ids = real_or_sentinel_gpu_ids(spec, is_sub_node=is_sub_node)
    return LaunchCommandContext(
        cell_index=cell_index,
        worker_in_cell_index=worker_in_cell_index,
        gpu_ids=pod_gpu_ids,
        local_gpu_ids=pod_gpu_ids,
        self_addrs=self_addrs,
        pool_addrs={pool_id: list(cells.values()) for pool_id, cells in addresses.items()},
    )


def _command_of_spec(spec: BaseWorkerSpec, context: LaunchCommandContext, plan: LaunchPlan) -> list[str]:
    match spec:
        case CommandWorkerSpec():
            return sentinels_to_placeholders(shlex.split(spec.launch_command(context)), spec)
        case ServeWorkerSpec():
            return _serve_command(spec, plan)
        case _:
            raise AssertionError(f"{spec.name} is neither launched by a command nor served over rpc: {spec}")


def _serve_command(spec: ServeWorkerSpec, plan: LaunchPlan) -> list[str]:
    interpreter_prefix = python_argv_prefix()
    workers_per_pod = spec.scheduling.workers_per_pod()
    serve = [
        *interpreter_prefix,
        "-m",
        _SERVE_MODULE,
        "--specs",
        _SPECS_FN,
        "--pool-id",
        spec.name,
        "--",
    ] + plan.worker_argv
    if workers_per_pod == 1:
        return serve
    return [
        *interpreter_prefix,
        "-m",
        _SUPERVISOR_MODULE,
        "--num-subprocesses",
        str(workers_per_pod),
        "--",
    ] + serve


def _is_sub_node(pairing_layout: PairingLayout | None) -> bool:
    if pairing_layout is None:
        return False
    return pairing_layout.num_gpus_per_inference_pod < pairing_layout.num_gpus_per_node


def _port_name(name: str) -> str:
    return name.replace("_", "-")[:15]
