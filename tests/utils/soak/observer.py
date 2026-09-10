import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
from tests.utils.soak.action import run_command
from tests.utils.soak.process_target import ProcessTarget
from tests.utils.soak.state import SoakObservation, SoakPodTarget, cell_type_of

from miles.utils.test_utils.kubectl_reads import KUBECTL_TIMEOUT_SECONDS, compute_release_selector
from miles.utils.workers.cell_operations.base import FaultTarget
from miles.utils.workers.naming import parse_cell_id
from miles.utils.workers.worker_provider.kubernetes.helm.env import DEFAULT_LABEL_KEYS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SoakObserver:
    base_url: str
    cell_types: set[str]
    namespace: str | None = None
    release: str | None = None
    fault_target_cell_types: frozenset[str] = frozenset()
    process_patterns_of_type: dict[str, dict[str, str]] = field(default_factory=dict)

    async def observe(self) -> SoakObservation:
        observed_at = datetime.now(timezone.utc)
        cells = None
        pods_of_cell: dict[str, list[SoakPodTarget]] = {}
        errors: dict[str, str] = {}
        fault_targets: dict[str, FaultTarget] = {}
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/v1/cells")
                response.raise_for_status()
                cells = [cell for cell in response.json()["items"] if cell_type_of(cell) in self.cell_types]
                fault_targets = await self._observe_fault_targets(client=client, cells=cells, errors=errors)
        except Exception as error:
            logger.info("Failed to observe cells", exc_info=True)
            errors["cells"] = repr(error)

        if self.release is not None:
            assert self.namespace, "A release observation needs a namespace"
            try:
                result = await run_command(
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
                    timeout_seconds=KUBECTL_TIMEOUT_SECONDS,
                )
                pods = json.loads(result.stdout)["items"]
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
                        and pod["metadata"].get("labels", {}).get(DEFAULT_LABEL_KEYS.cell_index)
                        == str(parsed.cell_index)
                    ]
                    for pod in pods_of_cell[name]:
                        for container, pattern in self.process_patterns_of_type.get(cell_type_of(cell), {}).items():
                            try:
                                process_result = await run_command(
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
                                        "tests.utils.soak.process_target",
                                        "observe",
                                        pod.uid,
                                        pattern,
                                    ],
                                    timeout_seconds=KUBECTL_TIMEOUT_SECONDS,
                                )
                                target = ProcessTarget.model_validate_json(process_result.stdout)
                                assert target.pod_uid == pod.uid and target.pattern == pattern
                                pod.process_targets[container] = target
                            except Exception as error:
                                logger.info("Failed to observe processes in %s/%s", pod.name, container, exc_info=True)
                                errors[f"processes:{pod.name}:{container}"] = repr(error)
            except Exception as error:
                logger.info("Failed to observe pods", exc_info=True)
                errors["pods"] = repr(error)
                pods_of_cell = {}
        return SoakObservation(
            timestamp=observed_at, cells=cells, pods_of_cell=pods_of_cell, errors=errors, fault_targets=fault_targets
        )

    async def _observe_fault_targets(
        self, *, client: httpx.AsyncClient, cells: list[dict], errors: dict[str, str]
    ) -> dict[str, FaultTarget]:
        async def read_target(cell: dict) -> tuple[str, FaultTarget | None]:
            name = cell["metadata"]["name"]
            try:
                response = await client.get(
                    f"{self.base_url}/api/v1/cells/{name}/fault-target", params={"sub_index": 0}
                )
                response.raise_for_status()
                target = FaultTarget.model_validate(response.json())
                assert target.cell_id == name and target.sub_index == 0
                assert target.workers_hash == cell["status"]["workers_hash"], f"Cell {name} changed during observation"
                return name, target
            except Exception as error:
                logger.info("Failed to observe fault target %s", name, exc_info=True)
                errors[f"fault_target:{name}"] = repr(error)
                return name, None

        observations = await asyncio.gather(
            *(read_target(cell) for cell in cells if cell_type_of(cell) in self.fault_target_cell_types)
        )
        return {name: target for name, target in observations if target is not None}
