import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
from tests.utils.soak.core.events import SoakObservationEvent
from tests.utils.soak.core.types import SoakObserver
from tests.utils.soak.core.utils import recording_error
from tests.utils.soak.ft.actions.base import CellFaultForms
from tests.utils.soak.ft.types import ROLLOUT_CELL_TYPE, CellTarget
from tests.utils.soak.k8s_utils.pod_manipulation import SoakPodTarget
from tests.utils.soak.k8s_utils.process_target import ProcessTarget

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.common import run_process
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.test_utils.kubectl_reads import KUBECTL_TIMEOUT_SECONDS, compute_release_selector
from miles.utils.workers.cell_operations.base import FaultTarget
from miles.utils.workers.naming import parse_cell_id
from miles.utils.workers.types import ClusterBackend, DeployComponent
from miles.utils.workers.worker_provider.kubernetes.helm.env import DEFAULT_LABEL_KEYS


def cell_type_of(cell: dict) -> str:
    return cell["metadata"]["labels"]["miles.io/cell-type"]


def cell_is_alive(cell: dict) -> bool:
    return any(cond["type"] == "Healthy" and cond["status"] == "True" for cond in cell["status"]["conditions"])


def cell_is_ready(cell: dict) -> bool:
    if cell["status"]["phase"] != "Running" or not cell_is_alive(cell):
        return False
    return cell_type_of(cell) != ROLLOUT_CELL_TYPE or any(
        condition["type"] == "Serving" and condition["status"] == "True" for condition in cell["status"]["conditions"]
    )


def create_cell_target(
    cell: dict, *, pods: list[SoakPodTarget], fault_target: FaultTarget | None = None
) -> CellTarget:
    return CellTarget(
        kind=cell_type_of(cell),
        identity=cell["metadata"]["name"],
        incarnation=cell["status"].get("workers_hash") or "",
        alive=cell_is_alive(cell),
        ready=cell_is_ready(cell),
        pods=pods,
        fault_target=fault_target,
    )


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
                    create_cell_target(
                        cell,
                        pods=pods_of_cell.get(cell["metadata"]["name"], []),
                        fault_target=fault_targets.get(cell["metadata"]["name"]),
                    )
                    for cell in cells
                ]
            ),
            errors=errors,
        )

    async def _observe_cells(self, *, errors: dict[str, str]) -> tuple[list[dict] | None, dict[str, FaultTarget]]:
        cells: list[dict] | None = None
        fault_targets: dict[str, FaultTarget] = {}
        with recording_error(errors, "cells"):
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/v1/cells")
                response.raise_for_status()
                cells = [cell for cell in response.json()["items"] if cell_type_of(cell) in self.cell_types]
                fault_targets = await self._observe_fault_targets(client=client, cells=cells, errors=errors)
        return cells, fault_targets

    async def _observe_fault_targets(
        self, *, client: httpx.AsyncClient, cells: list[dict], errors: dict[str, str]
    ) -> dict[str, FaultTarget]:
        async def read_target(cell: dict) -> tuple[str, FaultTarget | None]:
            name = cell["metadata"]["name"]
            target: FaultTarget | None = None
            with recording_error(errors, f"fault_target:{name}"):
                response = await client.get(
                    f"{self.base_url}/api/v1/cells/{name}/fault-target", params={"sub_index": 0}
                )
                response.raise_for_status()
                observed = FaultTarget.model_validate(response.json())
                assert observed.cell_id == name and observed.sub_index == 0
                assert (
                    observed.workers_hash == cell["status"]["workers_hash"]
                ), f"Cell {name} changed during observation"
                target = observed
            return name, target

        observations = await asyncio.gather(
            *(read_target(cell) for cell in cells if cell_type_of(cell) in self.fault_target_cell_types)
        )
        return {name: target for name, target in observations if target is not None}

    async def _read_pods(self, *, errors: dict[str, str]) -> list[dict] | None:
        if self.release is None:
            return None
        assert self.namespace, "A release observation needs a namespace"

        pods: list[dict] | None = None
        with recording_error(errors, "pods"):
            pods = await self._read_release_pods()
        return pods

    def _create_pod_targets(
        self, *, cells: list[dict] | None, pods: list[dict] | None, errors: dict[str, str]
    ) -> dict[str, list[SoakPodTarget]]:
        if pods is None:
            return {}
        assert self.namespace and self.release, "A release observation needs a namespace"

        pods_of_cell: dict[str, list[SoakPodTarget]] = {}
        with recording_error(errors, "pods"):
            for cell in cells or []:
                name = cell["metadata"]["name"]
                parsed = parse_cell_id(name)
                pods_of_cell[name] = [
                    SoakPodTarget(
                        namespace=self.namespace,
                        release=self.release,
                        name=pod["metadata"]["name"],
                        uid=pod["metadata"]["uid"],
                    )
                    for pod in pods
                    if pod["metadata"].get("labels", {}).get(DEFAULT_LABEL_KEYS.pool_id) == parsed.pool_id
                    and pod["metadata"].get("labels", {}).get(DEFAULT_LABEL_KEYS.cell_index) == str(parsed.cell_index)
                ]
            return pods_of_cell
        return {}

    async def _read_release_pods(self) -> list[dict]:
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
        return json.loads(result.stdout)["items"]

    async def _observe_processes(
        self, *, cells: list[dict] | None, pods_of_cell: dict[str, list[SoakPodTarget]], errors: dict[str, str]
    ) -> None:
        await asyncio.gather(
            *(
                self._observe_process(pod=pod, container=container, pattern=pattern, errors=errors)
                for cell in cells or []
                for pod in pods_of_cell.get(cell["metadata"]["name"], [])
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
                    "tests.utils.soak.k8s_utils.process_target",
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
    fault_target_types = {kind for kind in cell_types for form in forms[kind] if form.needs_fault_target}
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
