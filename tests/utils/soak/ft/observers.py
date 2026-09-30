import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
from tests.utils.soak.core.events import SoakObservationEvent
from tests.utils.soak.core.types import SoakObserver
from tests.utils.soak.core.utils import recording_error
from tests.utils.soak.ft.actions.base import CellFaultForms
from tests.utils.soak.ft.cells import cell_is_alive, cell_is_ready, cell_type_of
from tests.utils.soak.ft.types import CellTarget
from tests.utils.soak.k8s_utils.pod_manipulation import SoakPodTarget
from tests.utils.soak.k8s_utils.pod_processes import ProcessTarget

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.common import run_process
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.ft_utils.api_server.models import Cell, CellList
from miles.utils.test_utils.fault_injector.models import ObservedFaultHookTarget
from miles.utils.test_utils.kubectl_reads import KUBECTL_TIMEOUT_SECONDS, compute_release_selector
from miles.utils.workers.k8s_types import Pod, PodList
from miles.utils.workers.types import ClusterBackend, DeployComponent
from miles.utils.workers.worker_provider.kubernetes.core.pod_view import parse_pod
from miles.utils.workers.worker_provider.kubernetes.helm.env import DEFAULT_LABEL_KEYS


@dataclass(frozen=True, kw_only=True)
class CellObserver(SoakObserver):
    base_url: str
    cell_types: set[str]
    namespace: str | None = None
    release: str | None = None
    fault_target_cell_types: frozenset[str] = frozenset()
    process_patterns_of_type: dict[str, dict[str, str]] = field(default_factory=dict)

    async def observe(self) -> SoakObservationEvent:
        observed_at = datetime.now(timezone.utc)
        errors: dict[str, str] = {}

        observed_cells, pods = await asyncio.gather(self._observe_cells(errors=errors), self._read_pods(errors=errors))
        cells, fault_targets = observed_cells
        pods_of_cell = self._create_pod_targets(cells=cells, pods=pods, errors=errors)
        await self._observe_processes(cells=cells, pods_of_cell=pods_of_cell, errors=errors)

        return SoakObservationEvent(
            timestamp=observed_at,
            targets=(
                None
                if cells is None
                else [
                    _create_cell_target(
                        cell,
                        pods=pods_of_cell.get(cell.metadata.name, []),
                        fault_target=fault_targets.get(cell.metadata.name),
                    )
                    for cell in cells
                ]
            ),
            errors=errors,
        )

    async def _observe_cells(
        self, *, errors: dict[str, str]
    ) -> tuple[list[Cell] | None, dict[str, ObservedFaultHookTarget]]:
        cells: list[Cell] | None = None
        fault_targets: dict[str, ObservedFaultHookTarget] = {}
        with recording_error(errors, "cells"):
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/v1/cells")
                response.raise_for_status()
                cells = [
                    cell
                    for cell in CellList.model_validate(response.json()).items
                    if cell_type_of(cell) in self.cell_types
                ]
                fault_targets = await self._observe_fault_targets(client=client, cells=cells, errors=errors)
        return cells, fault_targets

    async def _observe_fault_targets(
        self, *, client: httpx.AsyncClient, cells: list[Cell], errors: dict[str, str]
    ) -> dict[str, ObservedFaultHookTarget]:
        async def read_target(cell: Cell) -> tuple[str, ObservedFaultHookTarget | None]:
            name = cell.metadata.name
            target: ObservedFaultHookTarget | None = None
            with recording_error(errors, f"fault_target:{name}"):
                response = await client.get(f"{self.base_url}/api/v1/cells/{name}/fault-target", params={"rank": 0})
                response.raise_for_status()
                observed = ObservedFaultHookTarget.model_validate(response.json())
                assert observed.cell_id == name and observed.rank == 0
                assert observed.workers_hash == cell.status.workers_hash, f"Cell {name} changed during observation"
                target = observed
            return name, target

        observations = await asyncio.gather(
            *(read_target(cell) for cell in cells if cell_type_of(cell) in self.fault_target_cell_types)
        )
        return {name: target for name, target in observations if target is not None}

    async def _read_pods(self, *, errors: dict[str, str]) -> list[Pod] | None:
        if self.release is None:
            return None
        assert self.namespace, "A release observation needs a namespace"

        pods: list[Pod] | None = None
        with recording_error(errors, "pods"):
            pods = await self._read_release_pods()
        return pods

    def _create_pod_targets(
        self, *, cells: list[Cell] | None, pods: list[Pod] | None, errors: dict[str, str]
    ) -> dict[str, list[SoakPodTarget]]:
        if pods is None:
            return {}
        assert self.namespace and self.release, "A release observation needs a namespace"

        with recording_error(errors, "pods"):
            pods_of_cell: dict[str, list[SoakPodTarget]] = {cell.metadata.name: [] for cell in cells or []}
            for pod in pods:
                parsed = parse_pod(pod, DEFAULT_LABEL_KEYS)
                if parsed is None or parsed.cell_id not in pods_of_cell:
                    continue
                pods_of_cell[parsed.cell_id].append(
                    SoakPodTarget(
                        namespace=self.namespace,
                        release=self.release,
                        name=pod.metadata.name,
                        uid=pod.metadata.uid,
                    )
                )
            return pods_of_cell
        return {}

    async def _read_release_pods(self) -> list[Pod]:
        result = await asyncio.to_thread(
            run_process,
            [
                "kubectl",
                "get",
                "pods",
                "--namespace",
                self.namespace,
                "--selector",
                compute_release_selector(release=self.release),
                "--output",
                "json",
            ],
            capture_output=True,
            check=True,
            timeout=KUBECTL_TIMEOUT_SECONDS,
        )
        return PodList.model_validate_json(result.stdout).items

    async def _observe_processes(
        self, *, cells: list[Cell] | None, pods_of_cell: dict[str, list[SoakPodTarget]], errors: dict[str, str]
    ) -> None:
        await asyncio.gather(
            *(
                self._observe_process(pod=pod, container=container, pattern=pattern, errors=errors)
                for cell in cells or []
                for pod in pods_of_cell.get(cell.metadata.name, [])
                for container, pattern in self.process_patterns_of_type.get(cell_type_of(cell), {}).items()
            )
        )

    async def _observe_process(
        self, *, pod: SoakPodTarget, container: str, pattern: str, errors: dict[str, str]
    ) -> None:
        with recording_error(errors, f"processes:{pod.name}:{container}"):
            process_result = await asyncio.to_thread(
                run_process,
                [
                    "kubectl",
                    "exec",
                    "--namespace",
                    pod.namespace,
                    pod.name,
                    "--container",
                    container,
                    "--",
                    "python3",
                    "-m",
                    "tests.utils.soak.k8s_utils.pod_processes_cli",
                    "observe",
                    pod.uid,
                    pattern,
                ],
                capture_output=True,
                check=True,
                timeout=KUBECTL_TIMEOUT_SECONDS,
            )
            target = ProcessTarget.model_validate_json(process_result.stdout)
            assert target.pod_uid == pod.uid and target.pattern == pattern
            pod.process_targets[container] = target


def create_cell_observer(
    *,
    base_url: str,
    cell_types: set[str],
    forms: CellFaultForms,
    config: ExecuteTrainConfig,
) -> CellObserver:
    fault_target_types = {
        fault_target_type
        for kind in cell_types
        for form in forms[kind]
        for fault_target_type in form.fault_target_cell_types(kind)
    }
    process_patterns = {
        kind: {container: pattern for form in forms[kind] for container, pattern in form.process_patterns.items()}
        for kind in cell_types
    }
    use_kubernetes = config.cluster_backend is ClusterBackend.KUBERNETES

    return CellObserver(
        base_url=base_url,
        cell_types=cell_types | fault_target_types,
        namespace=config.namespace if use_kubernetes else None,
        release=(
            ReleaseName(
                run_id=config.run_id, deploy_component=DeployComponent.ALL, deploy_instance_id=None
            ).serialize()
            if use_kubernetes
            else None
        ),
        fault_target_cell_types=frozenset(fault_target_types),
        process_patterns_of_type=process_patterns,
    )


def _create_cell_target(
    cell: Cell, *, pods: list[SoakPodTarget], fault_target: ObservedFaultHookTarget | None = None
) -> CellTarget:
    return CellTarget(
        kind=cell_type_of(cell),
        identity=cell.metadata.name,
        incarnation=cell.status.workers_hash,
        alive=cell_is_alive(cell),
        ready=cell_is_ready(cell),
        pods=pods,
        fault_target=fault_target,
    )
