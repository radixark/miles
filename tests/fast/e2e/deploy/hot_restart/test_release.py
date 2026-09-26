import pytest
from tests.fast.e2e.deploy.hot_restart.conftest import InstallReleaseCluster
from tests.utils.deploy.hot_restart import release as release_module
from tests.utils.deploy.hot_restart.release import RELEASE_POLL_INTERVAL_SECONDS, remove_release_and_wait

from miles.utils.workers.k8s_types import Pod, PodMetadata

_RELEASE = "miles-run-run-baseline-all"
_SELECTOR = f"app.kubernetes.io/instance={_RELEASE}"


class TestRemoveReleaseAndWait:
    def test_the_pods_are_waited_for_after_helm_no_longer_lists_the_release(
        self, install_cluster: InstallReleaseCluster
    ) -> None:
        """Helm can forget a release before its terminating GPU pods stop reserving the node."""
        cluster = install_cluster(manifests=[None], pods=[[_pod("gpu-pod")], []])

        remove_release_and_wait(release=_RELEASE, namespace="ci")

        assert cluster.events == [
            f"uninstall:ci/{_RELEASE}",
            f"manifest:ci/{_RELEASE}",
            f"pods:ci/{_SELECTOR}",
            f"sleep:{RELEASE_POLL_INTERVAL_SECONDS}",
            f"manifest:ci/{_RELEASE}",
            f"pods:ci/{_SELECTOR}",
        ]

    def test_a_release_helm_still_lists_is_waited_for_even_with_no_pods_left(
        self, install_cluster: InstallReleaseCluster
    ) -> None:
        """An uninstall still in flight may yet recreate hooks, so an empty pod list alone is not removal."""
        cluster = install_cluster(manifests=[object(), None], pods=[[]])

        remove_release_and_wait(release=_RELEASE, namespace="ci")

        assert cluster.events.count(f"sleep:{RELEASE_POLL_INTERVAL_SECONDS}") == 1

    def test_a_stuck_terminating_pod_fails_after_the_bounded_wait_naming_it(
        self, install_cluster: InstallReleaseCluster, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stuck pod fails the handoff instead of polling forever or starting the next side on its GPUs."""
        monkeypatch.setattr(release_module, "RELEASE_TIMEOUT_SECONDS", 3.0)
        cluster = install_cluster(manifests=[None], pods=[[_pod("stuck-gpu-pod")]])

        with pytest.raises(TimeoutError, match=r"release_exists=False, pods=\['stuck-gpu-pod'\]"):
            remove_release_and_wait(release=_RELEASE, namespace="ci")

        assert cluster.now == 3.0


def _pod(name: str) -> Pod:
    return Pod(metadata=PodMetadata(name=name, uid=f"{name}-uid"))
