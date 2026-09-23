import time

from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Helm, Kubectl
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability.pod_facts import selected_pods

RELEASE_POLL_INTERVAL_SECONDS: float = 1.0
RELEASE_TIMEOUT_SECONDS: float = 300.0


def remove_release_and_wait(*, release: str, namespace: str) -> None:
    selector = Kubectl.release_selector(release)
    deadline = time.monotonic() + RELEASE_TIMEOUT_SECONDS

    Helm.uninstall_if_present(release=release, namespace=namespace)
    while True:
        manifest = Helm.get_manifest(release, namespace)
        pods = selected_pods(namespace, selector)
        if manifest is None and not pods:
            return
        if time.monotonic() >= deadline:
            pod_names = sorted(pod.metadata.name for pod in pods)
            raise TimeoutError(
                f"Timed out removing release {release!r} from namespace {namespace!r}; "
                f"release_exists={manifest is not None}, pods={pod_names}"
            )
        time.sleep(RELEASE_POLL_INTERVAL_SECONDS)
