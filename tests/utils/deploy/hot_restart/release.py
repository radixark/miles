import time
from subprocess import TimeoutExpired

from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Helm, Kubectl
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability.pod_facts import selected_pods

RELEASE_POLL_INTERVAL_SECONDS: float = 1.0
RELEASE_TIMEOUT_SECONDS: float = 300.0


def remove_release_and_wait(*, release: str, namespace: str) -> None:
    selector = Kubectl.release_selector(release)
    deadline = time.monotonic() + RELEASE_TIMEOUT_SECONDS
    last_observed_state = ""

    def remaining_timeout() -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out removing release {release!r} from namespace {namespace!r}{last_observed_state}"
            )
        return remaining

    try:
        Helm.uninstall_if_present(release=release, namespace=namespace, timeout=remaining_timeout())
        while True:
            manifest = Helm.get_manifest(release=release, namespace=namespace, timeout=remaining_timeout())
            pods = selected_pods(namespace=namespace, selector=selector, timeout=remaining_timeout())
            if manifest is None and not pods:
                return
            last_observed_state = (
                f"; release_exists={manifest is not None}, pods={sorted(pod.metadata.name for pod in pods)}"
            )
            time.sleep(min(RELEASE_POLL_INTERVAL_SECONDS, remaining_timeout()))
    except TimeoutExpired as error:
        raise TimeoutError(
            f"Timed out removing release {release!r} from namespace {namespace!r}{last_observed_state}"
        ) from error
