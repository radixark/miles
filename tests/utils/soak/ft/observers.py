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

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.common import run_process
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.ft_utils.api_server.models import Cell, CellList
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

        cells, pods = await asyncio.gather(self._observe_cells(errors=errors), self._read_pods(errors=errors))
        pods_of_cell = self._create_pod_targets(cells=cells, pods=pods, errors=errors)

        return SoakObservationEvent(
            timestamp=observed_at,
            targets=(
                None
                if cells is None
                else [_create_cell_target(cell, pods=pods_of_cell.get(cell.metadata.name, [])) for cell in cells]
            ),
            errors=errors,
        )

    async def _observe_cells(self, *, errors: dict[str, str]) -> list[Cell] | None:
        cells: list[Cell] | None = None
        with recording_error(errors, "cells"):
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/v1/cells")
                response.raise_for_status()
                cells = [
                    cell
                    for cell in CellList.model_validate(response.json()).items
                    if cell_type_of(cell) in self.cell_types
                ]
        return cells

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


def create_cell_observer(
    *,
    base_url: str,
    cell_types: set[str],
    forms: CellFaultForms,
    config: ExecuteTrainConfig,
) -> CellObserver:
    use_kubernetes = config.cluster_backend is ClusterBackend.KUBERNETES

    return CellObserver(
        base_url=base_url,
        cell_types=cell_types,
        namespace=config.namespace if use_kubernetes else None,
        release=(
            ReleaseName(
                run_id=config.run_id, deploy_component=DeployComponent.ALL, deploy_instance_id=None
            ).serialize()
            if use_kubernetes
            else None
        ),
    )


def _create_cell_target(cell: Cell, *, pods: list[SoakPodTarget]) -> CellTarget:
    return CellTarget(
        kind=cell_type_of(cell),
        identity=cell.metadata.name,
        incarnation=cell.status.workers_hash,
        alive=cell_is_alive(cell),
        ready=cell_is_ready(cell),
        pods=pods,
    )
