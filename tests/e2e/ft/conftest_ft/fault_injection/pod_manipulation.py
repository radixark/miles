# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import logging
import random

from miles.utils.external_utils.command_utils.common import run_process
from miles.utils.test_utils.kubectl_reads import KUBECTL_TIMEOUT_SECONDS, read_objects_of_release
from miles.utils.workers.naming import parse_cell_id
from miles.utils.workers.worker_provider.kubernetes.helm.env import DEFAULT_LABEL_KEYS

logger = logging.getLogger(__name__)


def delete_one_pod_of_cell(*, namespace: str, release: str, cell_id: str, rng: random.Random) -> str:
    pod_names = list_pod_names_of_cell(namespace=namespace, release=release, cell_id=cell_id)
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


def list_pod_names_of_cell(*, namespace: str, release: str, cell_id: str) -> list[str]:
    return read_objects_of_release(
        kind="pods",
        release=release,
        namespace=namespace,
        output="jsonpath={.items[*].metadata.name}",
        extra_labels=_compute_cell_labels(cell_id),
    ).split()


def _compute_cell_labels(cell_id: str) -> list[str]:
    parsed = parse_cell_id(cell_id)
    return [f"{DEFAULT_LABEL_KEYS.pool_id}={parsed.pool_id}", f"{DEFAULT_LABEL_KEYS.cell_index}={parsed.cell_index}"]


def sigkill_process_patterns_in_pod(*, namespace: str, pod_name: str, container: str, process_pattern: str) -> None:
    result = run_process(
        [
            "kubectl",
            "exec",
            "--namespace",
            namespace,
            pod_name,
            "--container",
            container,
            "--",
            "pkill",
            "-9",
            "-f",
            process_pattern,
        ],
        capture_output=True,
        check=False,
        timeout=KUBECTL_TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, (
        f"No process matching {process_pattern!r} was killed inside {pod_name} (exit "
        f"{result.returncode}): {result.stderr.strip() or result.stdout.strip()}. A crash nobody caused would "
        f"otherwise be counted as one that happened"
    )

    logger.info(f"Sigkilled a {process_pattern} process inside pod {pod_name}")
