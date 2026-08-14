# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import logging
import random

from miles.utils.external_utils.command_utils.common import run_process
from miles.utils.workers.naming import parse_cell_id
from miles.utils.workers.worker_provider.kubernetes.helm.env import DEFAULT_LABEL_KEYS, INSTANCE_LABEL

logger = logging.getLogger(__name__)

KUBECTL_TIMEOUT_SECONDS: float = 60.0


def delete_one_pod_of_cell(*, namespace: str, release: str, cell_id: str, rng: random.Random) -> str:
    pod_names = _list_pod_names_of_cell(namespace=namespace, release=release, cell_id=cell_id)
    assert pod_names, f"Release {release} has no pod of cell {cell_id} in {namespace}, so there is nothing to delete"

    pod_name = rng.choice(pod_names)
    run_process(
        ["kubectl", "delete", "pod", "--namespace", namespace, "--wait=false", pod_name],
        capture_output=True,
        check=True,
        timeout=KUBECTL_TIMEOUT_SECONDS,
    )
    logger.info(f"Deleted pod {pod_name} of cell {cell_id}, one of {pod_names}")
    return pod_name


def _list_pod_names_of_cell(*, namespace: str, release: str, cell_id: str) -> list[str]:
    result = run_process(
        [
            "kubectl",
            "get",
            "pods",
            "--namespace",
            namespace,
            "--selector",
            _compute_cell_pod_selector(release=release, cell_id=cell_id),
            "--output",
            "jsonpath={.items[*].metadata.name}",
        ],
        capture_output=True,
        check=True,
        timeout=KUBECTL_TIMEOUT_SECONDS,
    )
    return result.stdout.split()


def _compute_cell_pod_selector(*, release: str, cell_id: str) -> str:
    parsed = parse_cell_id(cell_id)
    return ",".join(
        [
            f"{INSTANCE_LABEL}={release}",
            f"{DEFAULT_LABEL_KEYS.pool_id}={parsed.pool_id}",
            f"{DEFAULT_LABEL_KEYS.cell_index}={parsed.cell_index}",
        ]
    )
