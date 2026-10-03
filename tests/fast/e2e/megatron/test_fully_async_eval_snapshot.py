import json
import runpy
import shlex
from pathlib import Path

import pytest

from miles.utils.audit_utils.config_snapshot.generated_values import GENERATED_VALUES_ENV_VAR
from miles.utils.external_utils import command_utils
from miles.utils.test_utils.snapshot import SNAPSHOT_RECORD_DIR_ENV_VAR


@pytest.mark.parametrize("snapshot_enabled", [False, True])
def test_original_eval_modes_register_owned_directories_before_worker_launch(
    eval_launch_commands: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path, snapshot_enabled: bool
) -> None:
    """The original script transports each mode's allocated directory only for snapshot recording."""
    if snapshot_enabled:
        monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, str(tmp_path / "records"))
    else:
        monkeypatch.delenv(SNAPSHOT_RECORD_DIR_ENV_VAR, raising=False)
    script = Path(command_utils.repo_base_dir) / "tests/e2e/megatron/test_qwen3_4b_fully_async_eval.py"
    runpy.run_path(str(script), run_name="__main__")

    submissions = [command for command in eval_launch_commands if "ray job submit" in command]
    assert len(submissions) == 3
    for mode, command in zip(("shared", "fleet", "external"), submissions, strict=True):
        words = shlex.split(command)
        environment = json.loads(
            next(word.split("=", 1)[1] for word in words if word.startswith("--runtime-env-json="))
        )["env_vars"]
        if not snapshot_enabled:
            assert GENERATED_VALUES_ENV_VAR not in environment
            continue
        values = json.loads(environment[GENERATED_VALUES_ENV_VAR])
        [allocated] = [
            value
            for value in values
            if value["kind"] == "temporary_directory" and value["name"] == f"fully_async_eval_{mode}"
        ]
        assert allocated["value"].startswith(f"/dev/shm/miles_eval_{mode}_")
        if mode != "shared":
            assert words[words.index("--eval-hf-dir") + 1] == allocated["value"]
