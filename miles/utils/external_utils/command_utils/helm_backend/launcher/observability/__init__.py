from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager

from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Kubectl
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability.cluster_info import with_cluster_info
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability.log_follower import (
    with_log_following,
)

_MAX_LOG_REQUESTS = 100

_OBSERVABILITY_BOUNDARY = (
    "this launcher follows the pods of this run while it is watching, and prints their phases and warning "
    "events; anything beyond that -- metrics, history, logs of pods that already went away -- belongs to your "
    "cluster's own observability stack, which miles deliberately does not replace"
)


@contextmanager
def with_observability(*, namespace: str, selector: str) -> Iterator[None]:
    with ExitStack() as stack:
        stack.enter_context(with_cluster_info(namespace=namespace, selector=selector))
        stack.enter_context(with_log_following(namespace=namespace, selector=selector))
        yield


def farewell(*, namespace: str, release: str, workload: str) -> str:
    return "\n".join(
        [
            "the run keeps going after this launcher exits",
            f"  orchestrator log: {_orchestrator_log_command(namespace=namespace, workload=workload)}",
            f"  every pod of the run: {_release_log_command(namespace=namespace, release=release)}",
            "  this release uninstalls itself about two minutes after the run's verdict; the logs and the "
            "diagnosis of the run stay in its run directory",
            f"  tear down earlier: helm uninstall -n {namespace} {release}",
            f"  {_OBSERVABILITY_BOUNDARY}",
        ]
    )


def _orchestrator_log_command(*, namespace: str, workload: str) -> str:
    return f"kubectl logs --follow --namespace {namespace} statefulset/{workload} -c orchestrator"


def _release_log_command(*, namespace: str, release: str) -> str:
    return (
        f"kubectl logs --follow --namespace {namespace} --selector {Kubectl.release_selector(release)} "
        f"--all-containers --prefix --max-log-requests {_MAX_LOG_REQUESTS}"
    )
