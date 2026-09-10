import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx
from tests.utils.soak.action import run_command
from tests.utils.soak.state import SoakObservation, SoakPodTarget, cell_type_of

from miles.utils.test_utils.kubectl_reads import KUBECTL_TIMEOUT_SECONDS, compute_release_selector
from miles.utils.workers.naming import parse_cell_id
from miles.utils.workers.worker_provider.kubernetes.helm.env import DEFAULT_LABEL_KEYS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SoakObserver:
    base_url: str
    cell_types: set[str]
    namespace: str | None = None
    release: str | None = None

    async def observe(self) -> SoakObservation:
        observed_at = datetime.now(timezone.utc)
        cells = None
        pods_of_cell: dict[str, list[SoakPodTarget]] = {}
        errors: dict[str, str] = {}
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/v1/cells")
                response.raise_for_status()
                cells = [cell for cell in response.json()["items"] if cell_type_of(cell) in self.cell_types]
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
            except Exception as error:
                logger.info("Failed to observe pods", exc_info=True)
                errors["pods"] = repr(error)
                pods_of_cell = {}
        return SoakObservation(timestamp=observed_at, cells=cells, pods_of_cell=pods_of_cell, errors=errors)
