from subprocess import TimeoutExpired

import pytest
from tests.fast.e2e.deploy.hot_restart.conftest import InstallReleaseCluster
from tests.fast.e2e.deploy.hot_restart.release_timeout_fakes import _TimedOutCommand
from tests.utils.deploy.hot_restart import release as release_module

from miles.utils.workers.k8s_types import Pod, PodMetadata


class TestReleaseCommandTimeout:
    @pytest.mark.parametrize("stage", ["uninstall", "manifest", "pods"])
    def test_a_command_timeout_names_the_release_and_preserves_the_cause(
        self, install_cluster: InstallReleaseCluster, monkeypatch: pytest.MonkeyPatch, stage: str
    ) -> None:
        """Each external-command timeout must retain the owned release and original cause."""
        install_cluster(manifests=[None], pods=[[]])
        error = TimeoutExpired(cmd="cluster-command", timeout=2)
        owner, name = {
            "uninstall": (release_module.Helm, "uninstall_if_present"),
            "manifest": (release_module.Helm, "get_manifest"),
            "pods": (release_module, "selected_pods"),
        }[stage]
        monkeypatch.setattr(owner, name, _TimedOutCommand(error=error))

        with pytest.raises(TimeoutError, match="release 'owned-run' from namespace 'ci'") as raised:
            release_module.remove_release_and_wait(release="owned-run", namespace="ci")

        assert raised.value.__cause__ is error

    def test_a_later_command_timeout_keeps_the_last_observed_pods(
        self, install_cluster: InstallReleaseCluster, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A timeout after polling retains the last known resources that still block removal."""
        pod = Pod(metadata=PodMetadata(name="remaining-gpu", uid="uid"))
        cluster = install_cluster(manifests=[None], pods=[[pod]])
        error = TimeoutExpired(cmd="helm get manifest", timeout=2)

        def sleep(seconds: float) -> None:
            cluster.sleep(seconds)
            monkeypatch.setattr(release_module.Helm, "get_manifest", _TimedOutCommand(error=error))

        monkeypatch.setattr(release_module.time, "sleep", sleep)

        with pytest.raises(TimeoutError, match="release_exists=False, pods=\\['remaining-gpu'\\]") as raised:
            release_module.remove_release_and_wait(release="owned-run", namespace="ci")

        assert raised.value.__cause__ is error
