import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.fast.launch_scripts.py_harness import (
    CLEARED_ENV,
    call_entrypoint,
    format_recording,
    freeze_environment,
    host_filesystem_frozen,
    import_launch_script,
    install_command_recorder,
    iter_py_launch_scripts,
)
from tests.fast.launch_scripts.sh_harness import REPO_ROOT, assert_matches_snapshot

_SNAPSHOT_DIR = REPO_ROOT / "tests" / "snapshots" / "launch_scripts" / "py"

_SCRIPTS_IMPORTABLE_ONLY_UNDER_THE_NPU_PATCH = {"scripts/run_qwen3_4b_npu.py"}


def _glm_checkpoint(sandbox: Path, model_name: str, num_layers: int) -> dict[str, object]:
    model_dir = sandbox / "models"
    (model_dir / model_name).mkdir(parents=True)
    (model_dir / model_name / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "architectures": ["GlmMoeDsaForCausalLM"],
                "num_hidden_layers": num_layers,
            }
        )
    )
    return {"model_dir": str(model_dir)}


def _nemotron_checkpoint(sandbox: Path) -> dict[str, object]:
    model_dir = sandbox / "models"
    checkpoint = model_dir / "NVIDIA-Nemotron-3-Nano-4B-BF16"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "model_type": "nemotron_h",
                "auto_map": {"AutoConfig": "configuration_nemotron_h.NemotronHConfig"},
            }
        )
    )
    return {"model_dir": str(model_dir)}


_SCRIPTS_WHOSE_DEFAULTS_ARE_UNSUPPORTED: dict[str, Callable[[Path], dict[str, object]]] = {
    "scripts/run_deepseek_v4.py": lambda sandbox: {"model_name": "DeepSeek-V4-Flash-FP8-4layer"},
    "scripts/run_glm5_744b_a40b.py": lambda sandbox: _glm_checkpoint(sandbox, "GLM-5", 78),
    "scripts/run_glm5_2_744b_a40b.py": lambda sandbox: _glm_checkpoint(sandbox, "GLM-5.2", 78),
    "scripts/run_inkling.py": lambda sandbox: {"model_name": "Inkling-4layer"},
    "scripts/run_nemotron_3_nano_4b_fsdp.py": _nemotron_checkpoint,
    "scripts/run_nemotron_3_ultra_550b_a55b.py": lambda sandbox: {
        "model_name": "NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16-4layer"
    },
}

_ENTRYPOINTS_DISABLED_BY_THEIR_OWN_DEFAULTS = {("scripts/run_deepseek_v4.py", "prepare_mxfp8")}

_SCRIPTS = [
    script for script in iter_py_launch_scripts() if script.rel not in _SCRIPTS_IMPORTABLE_ONLY_UNDER_THE_NPU_PATCH
]
_CASES = [(script.rel, entrypoint) for script in _SCRIPTS for entrypoint in script.entrypoints]


@pytest.fixture(params=_CASES, ids=[f"{rel}::{entrypoint}" for rel, entrypoint in _CASES])
def recorded(request, monkeypatch, tmp_path):
    rel, entrypoint = request.param
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / rel)
    call_entrypoint(
        module,
        entrypoint,
        _SCRIPTS_WHOSE_DEFAULTS_ARE_UNSUPPORTED.get(rel, lambda sandbox: {})(tmp_path),
        sandbox=tmp_path,
    )
    return rel, entrypoint, recording, tmp_path


class TestEveryLauncherEntrypoint:
    def test_commands_match_snapshot(self, recorded):
        """Every launcher entrypoint must build exactly the recorded shell commands."""
        rel, entrypoint, recording, sandbox = recorded
        snapshot = _SNAPSHOT_DIR / rel / f"{entrypoint}.txt"

        assert_matches_snapshot(snapshot, format_recording(recording, sandbox=sandbox), f"{rel}::{entrypoint}")

    def test_entrypoint_issues_commands(self, recorded):
        """An entrypoint that silently does nothing is a broken launcher, not a passing test."""
        rel, entrypoint, recording, _ = recorded
        if (rel, entrypoint) in _ENTRYPOINTS_DISABLED_BY_THEIR_OWN_DEFAULTS:
            assert not recording.commands
        else:
            assert recording.commands


class TestHostFilesystemIsFrozen:
    def test_paths_outside_the_checkout_and_the_sandbox_report_absence(self, tmp_path):
        """A launcher that can see the host's checkpoints skips work, so the snapshot would follow the machine."""
        inside = tmp_path / "checkpoint.json"
        inside.write_text("{}")

        with host_filesystem_frozen(tmp_path):
            assert inside.exists()
            assert not Path("/root/models/some-checkpoint/model.safetensors.index.json").exists()

    def test_the_checkout_stays_visible(self, tmp_path):
        """A launcher resolves its own model args script out of the checkout, so hiding it breaks every entrypoint."""
        with host_filesystem_frozen(tmp_path):
            assert (REPO_ROOT / "pyproject.toml").exists()
            assert (REPO_ROOT / "scripts" / "models").exists()

    def test_an_unreadable_parent_reports_absence_instead_of_raising(self, tmp_path):
        """python 3.11 raises PermissionError from exists(), which is how the CPU runner's /root broke this."""
        unreadable = tmp_path / "unreadable"
        unreadable.mkdir()
        unreadable.chmod(0o000)
        try:
            with host_filesystem_frozen(tmp_path / "sandbox"):
                assert not (unreadable / "model.safetensors.index.json").exists()
        finally:
            unreadable.chmod(0o700)


class TestDiscovery:
    def test_all_py_launch_scripts_are_discovered(self):
        """Guards against the discovery glob silently going empty."""
        assert len(_SCRIPTS) > 15

    def test_every_discovered_launcher_is_covered_except_the_one_this_checkout_cannot_import(self):
        """A denylist that nobody rechecks only grows; name the survivors so the count cannot drift."""
        discovered = {script.rel for script in iter_py_launch_scripts()}

        assert discovered - {script.rel for script in _SCRIPTS} == _SCRIPTS_IMPORTABLE_ONLY_UNDER_THE_NPU_PATCH

    @pytest.mark.parametrize("rel", sorted(_SCRIPTS_IMPORTABLE_ONLY_UNDER_THE_NPU_PATCH))
    def test_the_uncovered_launcher_really_is_uncoverable_here(self, rel):
        """Once the NPU patch is upstreamed this fails, forcing the exclusion out instead of letting it rot."""
        with pytest.raises(ImportError, match="execute_train_npu"):
            import_launch_script(REPO_ROOT / rel)

    def test_every_discovered_entrypoint_has_a_golden_and_vice_versa(self):
        """A removed entrypoint leaves its golden behind, and a new one would otherwise go unrecorded."""
        expected = {f"{rel}/{entrypoint}.txt" for rel, entrypoint in _CASES}
        recorded = {path.relative_to(_SNAPSHOT_DIR).as_posix() for path in _SNAPSHOT_DIR.rglob("*.txt")}

        assert expected == recorded

    def test_the_snapshot_tree_holds_nothing_outside_the_three_recorded_families(self):
        """A whole orphan directory would sit under tests/snapshots/launch_scripts with nobody checking it."""
        root = _SNAPSHOT_DIR.parent
        families = {root / name for name in ("py", "sh", "self_executing")}

        assert set(root.iterdir()) == families

    def test_every_environment_knob_a_model_script_reads_is_frozen(self):
        """The snapshots now pin expanded model args, so a developer's exported override would fail them."""
        knobs = set()
        for script in sorted((REPO_ROOT / "scripts" / "models").iterdir()):
            if not script.is_file():
                continue
            text = script.read_text()
            knobs |= set(re.findall(r"\$\{([A-Z][A-Z0-9_]*):-", text))
            knobs |= set(re.findall(r"environ\.get\(\s*\"([A-Z][A-Z0-9_]*)\"", text))

        assert knobs
        assert knobs <= set(CLEARED_ENV)

    def test_execute_train_config_defaults_are_not_taken_from_a_slurm_allocation(self, monkeypatch):
        """SLURM_JOB_NUM_NODES is read at import time, so a stale allocation would skew every snapshot."""
        import miles.utils.external_utils.command_utils as command_utils

        assert command_utils.ExecuteTrainConfig().num_nodes == 1
