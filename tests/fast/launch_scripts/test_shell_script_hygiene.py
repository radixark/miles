import re

from tests.fast.launch_scripts.sh_harness import REPO_ROOT

_HARDCODED_CHECKOUTS = ("/root/miles", "/workspace/miles")

_REMOVED_COMMAND_HELPERS = re.compile(r"(?<![\w.])exec_command\s*\(")


class TestShellScriptHygiene:
    def test_no_shell_script_hardcodes_the_checkout_location(self):
        """A script that assumes one absolute checkout only runs inside one container image."""
        offenders = [
            path.relative_to(REPO_ROOT).as_posix()
            for root in (REPO_ROOT / "scripts", REPO_ROOT / "examples")
            for path in root.rglob("*.sh")
            for text in [path.read_text(errors="replace")]
            if any(hardcoded in text for hardcoded in _HARDCODED_CHECKOUTS)
        ]

        assert offenders == []


class TestDockerPatchHygiene:
    def test_no_patch_calls_the_removed_exec_command_helper(self):
        """A patch is applied at image build time, so a stale symbol only fails on the target hardware."""
        offenders = [
            path.relative_to(REPO_ROOT).as_posix()
            for path in (REPO_ROOT / "docker").rglob("*.patch")
            if _REMOVED_COMMAND_HELPERS.search(path.read_text(errors="replace"))
        ]

        assert offenders == []
