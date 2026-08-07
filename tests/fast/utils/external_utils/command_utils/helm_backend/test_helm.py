import json

from miles.utils.external_utils.command_utils.helm_backend import helm


class TestUpgradeCommand:
    def test_installs_a_missing_release_and_updates_an_existing_one(self):
        """Relaunching a run id must update it in place, which plain upgrade would refuse to do."""
        command = helm.upgrade_command(release="r", namespace="rl", chart="/c", values_files=[])

        assert command[:4] == ["helm", "upgrade", "--install", "r"]

    def test_keeps_the_user_values_ahead_of_the_computed_ones(self):
        """A run value must win over a cluster default, and helm lets the later file win."""
        command = helm.upgrade_command(
            release="r", namespace="rl", chart="/c", values_files=["/infra.yaml", "/run.yaml"]
        )

        assert command[command.index("/infra.yaml") - 1] == "--values"
        assert command.index("/infra.yaml") < command.index("/run.yaml")

    def test_labels_a_ci_release_so_the_next_run_can_clean_it_up(self):
        """The cleanup selects on this label, and an unlabelled CI release is one nothing will ever remove."""
        command = helm.upgrade_command(release="r", namespace="rl", chart="/c", values_files=[], ci_run=True)

        assert command[command.index("--labels") + 1] == f"{helm.CI_LABEL}=true"

    def test_leaves_a_human_release_unlabelled(self):
        """A developer's run carrying the CI label would be uninstalled by the next CI job in that namespace."""
        command = helm.upgrade_command(release="r", namespace="rl", chart="/c", values_files=[])

        assert "--labels" not in command

    def test_labels_the_release_rather_than_its_objects(self):
        """helm --labels records release metadata; a values-level label would not be selectable by helm list."""
        command = helm.upgrade_command(release="r", namespace="rl", chart="/c", values_files=[], ci_run=True)

        assert command.index("--labels") > command.index("--namespace")


class TestCiCleanup:
    def test_narrows_the_search_by_both_namespace_and_label(self):
        """Deleting another user's run would kill a live experiment, so neither filter may be dropped."""
        command = helm.list_ci_releases_command("ci-runner-3")

        assert command[command.index("--namespace") + 1] == "ci-runner-3"
        assert command[command.index("--selector") + 1] == f"{helm.CI_LABEL}=true"

    def test_reads_the_release_names_helm_reports(self):
        """The names drive uninstall, so a parse that silently returns nothing would leave releases behind."""
        output = json.dumps([{"name": "miles-run-a", "namespace": "ci"}, {"name": "miles-run-b"}])

        assert helm.parse_release_names(output) == ["miles-run-a", "miles-run-b"]

    def test_treats_no_output_as_nothing_to_clean(self):
        """helm prints nothing when no release matches, and that is not an error."""
        assert helm.parse_release_names("") == []

    def test_uninstalls_inside_the_namespace_it_was_told(self):
        """A release name exists per namespace, so a missing namespace could hit a different one."""
        assert helm.uninstall_command("miles-run-a", "ci") == [
            "helm",
            "uninstall",
            "miles-run-a",
            "--namespace",
            "ci",
        ]


class TestChartDir:
    def test_finds_the_chart_inside_the_checkout(self):
        """The launcher installs the chart of the code it runs, not one from a registry."""
        assert helm.chart_dir("/repo").as_posix() == "/repo/charts/miles-run"
