from collections.abc import Callable

import pytest
from tests.utils.deploy.hot_restart import release as release_module

from miles.utils.workers.k8s_types import Pod


class FakeReleaseCluster:
    def __init__(self, *, manifests: list[object | None], pods: list[list[Pod]]) -> None:
        self.events: list[str] = []
        self.now = 0.0
        self._manifests = manifests
        self._pods = pods

    def uninstall_if_present(self, *, release: str, namespace: str) -> None:
        self.events.append(f"uninstall:{namespace}/{release}")

    def get_manifest(self, release: str, namespace: str) -> object | None:
        self.events.append(f"manifest:{namespace}/{release}")
        return self._manifests.pop(0) if len(self._manifests) > 1 else self._manifests[0]

    def selected_pods(self, namespace: str, selector: str) -> list[Pod]:
        self.events.append(f"pods:{namespace}/{selector}")
        return self._pods.pop(0) if len(self._pods) > 1 else self._pods[0]

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.events.append(f"sleep:{seconds}")
        self.now += seconds


InstallReleaseCluster = Callable[..., FakeReleaseCluster]


@pytest.fixture
def install_cluster(monkeypatch: pytest.MonkeyPatch) -> InstallReleaseCluster:
    def install(*, manifests: list[object | None], pods: list[list[Pod]]) -> FakeReleaseCluster:
        cluster = FakeReleaseCluster(manifests=manifests, pods=pods)
        monkeypatch.setattr(release_module.Helm, "uninstall_if_present", cluster.uninstall_if_present)
        monkeypatch.setattr(release_module.Helm, "get_manifest", cluster.get_manifest)
        monkeypatch.setattr(release_module, "selected_pods", cluster.selected_pods)
        monkeypatch.setattr(release_module.time, "monotonic", cluster.monotonic)
        monkeypatch.setattr(release_module.time, "sleep", cluster.sleep)
        return cluster

    return install
