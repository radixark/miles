import json
import re
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

    def test_asks_helm_only_for_the_releases_of_this_run(self, monkeypatch):
        """helm truncates a busy namespace to 256 releases, which can drop the sibling that actually failed."""
        commands = _helm_listing(monkeypatch, [PRIMARY])

        entrypoint._releases_of_run(release=PRIMARY, namespace="rl")

        assert commands[0][commands[0].index("--filter") + 1] == f"^{re.escape(ReleaseName.run_prefix(run_id=RUN_ID))}"

    def test_the_filter_is_a_regex_the_run_id_cannot_break_out_of(self):
        """helm reads it as a regular expression, and a run id is not written to be one."""
        assert re.fullmatch(f"^{re.escape(ReleaseName.run_prefix(run_id=RUN_ID))}.*", PRIMARY)


class TestTheSelectorItDescribesWith:
    def test_matches_every_release_of_the_run_at_once(self):
        """One kubectl call per describe keeps the failure path short while covering all deployments."""
        selector = Kubectl.releases_selector([PRIMARY, TRAINER])

        assert selector == f"{INSTANCE_LABEL} in ({PRIMARY},{TRAINER})"

    def test_stays_an_equality_for_an_unsplit_run(self):
        """An all-in-one run has one release, and the plain selector is what its snapshots already carry."""
        assert Kubectl.releases_selector([PRIMARY]) == Kubectl.release_selector(PRIMARY)


# Legal for helm, but not for miles: the run id is one character longer than the naming rules allow.
UNPARSABLE_NEIGHBOUR = "miles-run-" + "a" * 34 + "-all"


class TestAReleaseTheMilesRulesCannotRead:
    def test_it_belongs_to_no_run(self):
        """ReleaseName.parse builds a validated model, so such a name raised out of the candidate loop."""
        assert entrypoint._belongs_to_run(UNPARSABLE_NEIGHBOUR, run_id=RUN_ID) is False

    def test_a_release_of_another_naming_scheme_belongs_to_no_run(self):
        """A namespace holds releases of other charts, and none of them is part of this run."""
        assert entrypoint._belongs_to_run("someone-elses-release", run_id=RUN_ID) is False

    def test_a_release_of_this_run_still_belongs_to_it(self):
        """Reading an unreadable name as belonging to nothing may not cost the readable ones."""
        assert entrypoint._belongs_to_run(PRIMARY, run_id=RUN_ID) is True

    def test_such_a_neighbour_does_not_replace_the_diagnosis_of_the_failed_run(self, monkeypatch):
        """That loop runs outside both try blocks, so one neighbour replaced the diagnosis with a traceback."""
        _helm_listing(monkeypatch, [PRIMARY, TRAINER, UNPARSABLE_NEIGHBOUR])

        assert entrypoint._releases_of_run(release=PRIMARY, namespace="rl") == sorted([PRIMARY, TRAINER])

    def test_such_a_neighbour_is_not_diagnosed_as_part_of_the_run(self, monkeypatch):
        """Describing someone else's pods reads an experiment this run has nothing to do with."""
        _helm_listing(monkeypatch, [PRIMARY, UNPARSABLE_NEIGHBOUR])

        assert UNPARSABLE_NEIGHBOUR not in entrypoint._releases_of_run(release=PRIMARY, namespace="rl")
