import importlib.util
import shlex
from pathlib import Path

import pytest
from tests.fast.launch_scripts.py_harness import (
    REPO_ROOT,
    call_entrypoint,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)


class TestExplicitExampleCli:
    @pytest.mark.parametrize(
        "script,expected",
        [
            (
                "examples/experimental/hud/run_hud2048.py",
                {
                    "--hud-env-dir": "/root/v6browser",
                    "--hud-snapshot-name": "miles-hud-v6browser",
                    "--hud-max-steps": "24",
                    "--hud-shot-width": "512",
                    "--hud-max-tokens-per-turn": "512",
                    "--hud-max-sandboxes": "32",
                    "--hud-episode-timeout-s": "1800",
                },
            ),
            (
                "examples/geo3k_vlm/multi_turn/run_geo3k_vlm_multi_turn.py",
                {"--max-turns": "3", "--rollout-interaction-env-path": "examples.geo3k_vlm.multi_turn.env_geo3k"},
            ),
        ],
    )
    def test_environment_recipe_values_reach_the_submitted_command(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, script: str, expected: dict[str, str]
    ) -> None:
        """Custom configuration removal preserves the actual HUD and Geo3k environment recipes."""
        freeze_environment(monkeypatch)
        monkeypatch.setenv("DAYTONA_API_KEY_FILE", str(tmp_path / "missing-key"))
        monkeypatch.setenv("DAYTONA_API_KEY", "test-credential")
        find_spec = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *args: object() if name in {"hud", "daytona"} else find_spec(name, *args),
        )
        recording = install_command_recorder(monkeypatch)
        module = import_launch_script(REPO_ROOT / script)

        call_entrypoint(module, "execute", {}, sandbox=tmp_path)

        tokens = shlex.split(recording.commands[-1])
        assert "--custom-config-path" not in tokens
        for flag, value in expected.items():
            assert tokens[tokens.index(flag) + 1] == value
