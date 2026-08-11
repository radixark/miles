from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Helm, Kubectl


class TestUpgradeCommand:
    def test_installs_a_missing_release_and_updates_an_existing_one(self):
        """Relaunching a run id must update it in place, which plain upgrade would refuse to do."""
        command = Helm.upgrade_command("r", "myns", "/c", [])

        assert command[:4] == ["helm", "upgrade", "--install", "r"]

    def test_keeps_the_user_values_ahead_of_the_computed_ones(self):
        """A run value must win over a cluster default, and helm lets the later file win."""
        command = Helm.upgrade_command("r", "myns", "/c", ["/infra.yaml", "/run.yaml"])

        assert command[command.index("/infra.yaml") - 1] == "--values"
        assert command.index("/infra.yaml") < command.index("/run.yaml")


class TestRawCommands:
    def test_a_helm_call_reports_its_failure_instead_of_raising(self, monkeypatch):
        """The callers of these wrappers all want to read a failure, not to be unwound by it."""
        recorded = {}

        def fake_run(argv, *, capture_output, check, input=None):
            recorded.update(argv=argv, check=check)
            return None

        monkeypatch.setattr(
            "miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper.run_process", fake_run
        )
        Helm.run_raw("template", "r", "/c")

        assert recorded["argv"] == ["helm", "template", "r", "/c"]
        assert recorded["check"] is False

    def test_a_kubectl_call_is_spelled_out_the_same_way(self, monkeypatch):
        """One wrapper for both binaries is what keeps command strings out of the callers."""
        recorded = {}

        def fake_run(argv, *, capture_output, check, input=None):
            recorded.update(argv=argv)
            return None

        monkeypatch.setattr(
            "miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper.run_process", fake_run
        )
        Kubectl.run_raw("get", "namespace", "--", "myns")

        assert recorded["argv"] == ["kubectl", "get", "namespace", "--", "myns"]
