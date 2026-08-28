import json
import subprocess

import pytest

from miles.utils.external_utils.command_utils.helm_backend.launcher import command_wrapper, entrypoint
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Kubectl
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.workers.types import DeployComponent
from miles.utils.workers.worker_provider.kubernetes.helm.env import INSTANCE_LABEL

RUN_ID = "260101-000000-000"
OTHER_RUN_ID = "260101-111111-111"


def _release(component: DeployComponent, instance_id: str | None = None, run_id: str = RUN_ID) -> str:
    return ReleaseName(run_id=run_id, deploy_component=component, deploy_instance_id=instance_id).serialize()


PRIMARY = _release(DeployComponent.PRIMARY)
TRAINER = _release(DeployComponent.TRAINER)
ENGINE = _release(DeployComponent.INFERENCE, "e0")
STRANGER = _release(DeployComponent.ALL, run_id=OTHER_RUN_ID)


def _helm_listing(monkeypatch: pytest.MonkeyPatch, releases: list[str]) -> list[list[str]]:
    commands: list[list[str]] = []

    def fake_run(command: list[str], capture_output: bool) -> subprocess.CompletedProcess:
        commands.append(command)
        body = json.dumps([{"name": release} for release in releases])
        return subprocess.CompletedProcess(args=command, returncode=0, stdout=body, stderr="")

    monkeypatch.setattr(command_wrapper, "_run", fake_run)
    return commands


class TestTheReleasesADiagnosisCovers:
    def test_covers_every_deployment_the_run_installed(self, monkeypatch):
        """A split run fails in whichever deployment broke, and that is rarely the one being followed."""
        _helm_listing(monkeypatch, [PRIMARY, TRAINER, ENGINE, STRANGER])

        found = entrypoint._releases_of_run(release=PRIMARY, namespace="rl")

        assert found == sorted([PRIMARY, TRAINER, ENGINE])

    def test_leaves_another_run_in_the_namespace_alone(self, monkeypatch):
        """Namespaces hold more than one run, and describing someone else's pods reads their experiment."""
        _helm_listing(monkeypatch, [PRIMARY, STRANGER])

        assert STRANGER not in entrypoint._releases_of_run(release=PRIMARY, namespace="rl")

    def test_asks_helm_for_the_whole_namespace_because_a_run_carries_no_label(self, monkeypatch):
        """Releases are labelled per release, so the siblings can only be found by listing and parsing."""
        commands = _helm_listing(monkeypatch, [PRIMARY])

        entrypoint._releases_of_run(release=PRIMARY, namespace="rl")

        assert "--selector" not in commands[0]
        assert commands[0][commands[0].index("--namespace") + 1] == "rl"

    def test_falls_back_to_the_followed_release_when_helm_cannot_be_asked(self, monkeypatch):
        """A diagnosis of one deployment beats no diagnosis at all when the run has already failed."""

        def fake_run(command: list[str], capture_output: bool) -> subprocess.CompletedProcess:
            raise RuntimeError("helm is not reachable")

        monkeypatch.setattr(command_wrapper, "_run", fake_run)

        assert entrypoint._releases_of_run(release=PRIMARY, namespace="rl") == [PRIMARY]

    def test_keeps_the_followed_release_even_when_helm_forgot_it(self, monkeypatch):
        """The release being diagnosed is the one known to have failed, listed or not."""
        _helm_listing(monkeypatch, [])

        assert entrypoint._releases_of_run(release=PRIMARY, namespace="rl") == [PRIMARY]


class TestTheSelectorItDescribesWith:
    def test_matches_every_release_of_the_run_at_once(self):
        """One kubectl call per describe keeps the failure path short while covering all deployments."""
        selector = Kubectl.releases_selector([PRIMARY, TRAINER])

        assert selector == f"{INSTANCE_LABEL} in ({PRIMARY},{TRAINER})"

    def test_stays_an_equality_for_an_unsplit_run(self):
        """An all-in-one run has one release, and the plain selector is what its snapshots already carry."""
        assert Kubectl.releases_selector([PRIMARY]) == Kubectl.release_selector(PRIMARY)
