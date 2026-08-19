from collections.abc import Sequence

from miles.utils.external_utils.command_utils.common import run_process
from miles.utils.workers.worker_provider.kubernetes.helm.env import INSTANCE_LABEL

KUBECTL_TIMEOUT_SECONDS: float = 60.0


def read_objects_of_release(
    *, kind: str, release: str, namespace: str, output: str, extra_labels: Sequence[str] = ()
) -> str:
    result = run_process(
        [
            "kubectl",
            "get",
            kind,
            "--namespace",
            namespace,
            "--selector",
            compute_release_selector(release=release, extra_labels=extra_labels),
            "--output",
            output,
        ],
        capture_output=True,
        check=True,
        timeout=KUBECTL_TIMEOUT_SECONDS,
    )
    return result.stdout


def compute_release_selector(*, release: str, extra_labels: Sequence[str] = ()) -> str:
    return ",".join([f"{INSTANCE_LABEL}={release}", *extra_labels])
