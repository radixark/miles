from pathlib import Path

import pytest
from tests.fast.launch_scripts.sh_harness import REPO_ROOT, run_launch_script


class TestExplicitMisShell:
    @pytest.mark.parametrize(
        "script,hook",
        [
            ("run-qwen3-4b-mis.sh", "compute_mis_weights_with_cp"),
            ("run-qwen3-4b-fsdp-mis.sh", "compute_mis_weights_fsdp"),
        ],
    )
    def test_both_shell_recipes_preserve_mis_values_and_backend_specific_hooks(
        self, tmp_path: Path, script: str, hook: str
    ) -> None:
        """The shell recipes retain TIS and RS when the custom YAML disappears."""
        result = run_launch_script(
            script=REPO_ROOT / "examples" / "infra_features" / "train_infer_mismatch_helper" / script,
            sandbox=tmp_path,
        )
        assert result.returncode == 0, result.stderr
        tokens = result.ray_job_submit_argv()
        assert "--custom-config-path" not in tokens
        assert "--use-tis" in tokens
        assert "--use-rs" in tokens
        assert "--tis-batch-normalize" in tokens
        for flag, value in {
            "--tis-level": "token",
            "--rs-level": "token",
            "--tis-mode": "truncate",
            "--tis-lower-bound": "0.5",
            "--tis-upper-bound": "2.0",
            "--rs-veto-threshold": "1.0e-4",
            "--custom-tis-function-path": f"examples.infra_features.train_infer_mismatch_helper.mis.{hook}",
        }.items():
            assert tokens[tokens.index(flag) + 1] == value
