import subprocess
from dataclasses import dataclass, field

import pytest

from miles.utils.external_utils.command_utils.helm_backend.launcher import command_wrapper, entrypoint

NAMESPACE = "rl"
_OTHER_RUN_ID = "260101-000000-999"


@dataclass
class FakeCluster:
    releases: list[dict[str, str | bool]]
    commands: list[list[str]] = field(default_factory=list)

    def __call__(self, command: list[str], capture_output: bool = False) -> subprocess.CompletedProcess:
        self.commands.append(command)
        if command[1] == "list":
            return subprocess.CompletedProcess(args=command, returncode=0, stdout=self._listed(command), stderr="")
        self.releases = [release for release in self.releases if release["name"] != command[2]]
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    def uninstalled(self) -> list[str]:
        return [command[2] for command in self.commands if command[1] == "uninstall"]

    def _listed(self, command: list[str]) -> str:
        namespace = command[command.index("--namespace") + 1]
        selector = command[command.index("--selector") + 1]
        assert selector == f"{command_wrapper.CI_LABEL}=true"
        matched = [release for release in self.releases if release["namespace"] == namespace and release["ci"] is True]
        return "[" + ", ".join(f'{{"name": "{release["name"]}"}}' for release in matched) + "]"


def _cluster(monkeypatch: pytest.MonkeyPatch, releases: list[dict[str, str | bool]]) -> FakeCluster:
    cluster = FakeCluster(releases=releases)
    monkeypatch.setattr(command_wrapper, "_run", cluster)
    return cluster


def _clean_up(*, keep_run_id: str = _OTHER_RUN_ID) -> list[str]:
    return entrypoint._uninstall_leftover_ci_releases(NAMESPACE, keep_run_id=keep_run_id)


class TestUninstallLeftoverCiReleases:
    def test_removes_only_releases_matching_both_the_namespace_and_the_label(self, monkeypatch):
        """A human's release in the same namespace, or a CI release of another namespace, is not CI's to delete."""
        cluster = _cluster(
            monkeypatch,
            [
                {"name": "ci-here", "namespace": NAMESPACE, "ci": True},
                {"name": "human-here", "namespace": NAMESPACE, "ci": False},
                {"name": "ci-elsewhere", "namespace": "other", "ci": True},
            ],
        )

        _clean_up()

        assert cluster.uninstalled() == ["ci-here"]

    def test_asks_helm_for_both_filters_at_once(self, monkeypatch):
        """Filtering on only one of the two would list releases the caller must never touch."""
        cluster = _cluster(monkeypatch, [])

        _clean_up()

        listing = cluster.commands[0]
        assert listing[listing.index("--namespace") + 1] == NAMESPACE
        assert listing[listing.index("--selector") + 1] == f"{command_wrapper.CI_LABEL}=true"

    def test_uninstalls_every_release_it_listed_in_order(self, monkeypatch):
        """Stopping at the first one would leak the rest, which is how a runner fills up over a week."""
        cluster = _cluster(
            monkeypatch,
            [{"name": name, "namespace": NAMESPACE, "ci": True} for name in ("first", "second", "third")],
        )

        _clean_up()

        assert cluster.uninstalled() == ["first", "second", "third"]

    def test_uninstalls_each_release_from_the_namespace_it_was_found_in(self, monkeypatch):
        """helm uninstall without the namespace looks in the caller's default context and finds nothing."""
        cluster = _cluster(monkeypatch, [{"name": "ci-here", "namespace": NAMESPACE, "ci": True}])

        _clean_up()

        assert cluster.commands[1] == ["helm", "uninstall", "ci-here", "--namespace", NAMESPACE]

    def test_reports_the_names_it_removed(self, monkeypatch):
        """The caller logs what it cleaned, and a silent cleanup is indistinguishable from a broken one."""
        _cluster(monkeypatch, [{"name": name, "namespace": NAMESPACE, "ci": True} for name in ("a", "b")])

        assert _clean_up() == ["a", "b"]

    def test_does_nothing_when_no_ci_release_is_left(self, monkeypatch):
        """An empty listing must not turn into an uninstall of nothing, which helm treats as an error."""
        cluster = _cluster(monkeypatch, [{"name": "human-here", "namespace": NAMESPACE, "ci": False}])

        assert _clean_up() == []
        assert cluster.uninstalled() == []

    def test_keeps_the_sibling_releases_of_the_run_being_launched(self, monkeypatch):
        """A split run installs one ci release per component, so cleaning them would tear down its own halves."""
        run_id = "260101-000000-000"
        siblings = [f"miles-run-{run_id}-all", f"miles-run-{run_id}-trainer", f"miles-run-{run_id}-primary"]
        cluster = _cluster(
            monkeypatch,
            [{"name": name, "namespace": NAMESPACE, "ci": True} for name in ("ci-of-another-run", *siblings)],
        )

        removed = _clean_up(keep_run_id=run_id)

        assert removed == ["ci-of-another-run"]
        assert cluster.uninstalled() == ["ci-of-another-run"]

    def test_recognizes_every_release_this_run_installs_by_the_name_they_share(self):
        """One component release missed here is one release of this run that its own next launch uninstalls."""
        run_id = "260101-000000-000"

        assert entrypoint._releases_of_run(run_id) == {
            f"miles-run-{run_id}",
            f"miles-run-{run_id}-primary",
            f"miles-run-{run_id}-trainer",
        }

    def test_a_release_of_a_run_whose_id_starts_with_this_one_is_still_cleaned(self, monkeypatch):
        """A prefix test reads another run's releases as this run's siblings and leaves them behind forever."""
        run_id = "260101-000000-000"
        _cluster(monkeypatch, [{"name": f"miles-run-{run_id}-trainer-primary", "namespace": NAMESPACE, "ci": True}])

        assert _clean_up(keep_run_id=run_id) == [f"miles-run-{run_id}-trainer-primary"]

    def test_treats_empty_helm_output_as_nothing_to_clean(self, monkeypatch):
        """helm prints an empty body rather than [] when it has no rows, and json.loads would raise on it."""
        monkeypatch.setattr(
            command_wrapper,
            "_run",
            lambda command, capture_output=False: subprocess.CompletedProcess(
                args=command, returncode=0, stdout="", stderr=""
            ),
        )

        assert _clean_up() == []
