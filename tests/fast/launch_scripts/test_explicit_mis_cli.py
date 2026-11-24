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


_MIS_FLAGS = frozenset(
    {
        "--use-tis",
        "--use-rs",
        "--tis-level",
        "--rs-level",
        "--tis-mode",
        "--tis-lower-bound",
        "--tis-upper-bound",
        "--rs-veto-threshold",
        "--tis-batch-normalize",
        "--custom-tis-function-path",
    }
)


class TestExplicitMisCli:
    @pytest.mark.parametrize(
        "script,entrypoint,hardware",
        [
            ("scripts/run_qwen3_4b.py", "train", "GB300"),
            ("scripts/run_qwen3_30b_a3b.py", "execute", "GB300"),
            ("scripts/amd/run_qwen3_30b_a3b.py", "execute", "MI355X"),
            ("scripts/run_glm45_355b_a32b.py", "train", "GB300"),
            ("scripts/run_joy_ai_llm_flash.py", "execute", "GB300"),
            ("scripts/run_deepseek_v32.py", "train", "GB300"),
        ],
    )
    @pytest.mark.parametrize("enabled,use_rs", [(False, True), (True, False), (True, True)])
    def test_the_submitted_recipe_keeps_mis_values_and_the_conditional_rs_switch(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        script: str,
        entrypoint: str,
        hardware: str,
        enabled: bool,
        use_rs: bool,
    ) -> None:
        """MIS CLI migration preserves the effective recipe and the independent rejection sampling switch."""
        freeze_environment(monkeypatch, hardware=hardware)
        recording = install_command_recorder(monkeypatch)
        module = import_launch_script(REPO_ROOT / script)

        call_entrypoint(
            module,
            entrypoint,
            {"hardware": hardware, "enable_mis": enabled, "tis_use_rs": use_rs},
            sandbox=tmp_path,
        )

        tokens = shlex.split(recording.commands[-1])
        assert "--custom-config-path" not in tokens
        base_tis = script == "scripts/run_glm45_355b_a32b.py"
        assert ("--use-tis" in tokens) is (enabled or base_tis)
        assert ("--use-rs" in tokens) is (enabled and use_rs)
        if enabled:
            for flag, value in {
                "--tis-level": "token",
                "--rs-level": "token",
                "--tis-mode": "truncate",
                "--tis-lower-bound": "0.5",
                "--tis-upper-bound": "2.0",
                "--rs-veto-threshold": "1.0e-4",
                "--custom-tis-function-path": "examples.infra_features.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp",
            }.items():
                assert tokens[tokens.index(flag) + 1] == value
            assert "--tis-batch-normalize" in tokens
        else:
            assert _MIS_FLAGS.intersection(tokens) == ({"--use-tis"} if base_tis else set())

    @pytest.mark.parametrize(
        "script,hardware", [("scripts/run_deepseek_v4.py", "H200"), ("scripts/amd/run_deepseek_v4.py", "MI355X")]
    )
    @pytest.mark.parametrize("enabled", [False, True])
    def test_fixed_rs_recipes_emit_all_mis_options_only_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, script: str, hardware: str, enabled: bool
    ) -> None:
        """The DSV4 recipes preserve the complete fixed rejection sampling policy."""
        freeze_environment(monkeypatch, hardware=hardware)
        recording = install_command_recorder(monkeypatch)
        module = import_launch_script(REPO_ROOT / script)
        call_entrypoint(
            module,
            "train",
            {
                "enable_mis": enabled,
                "model_name": "DeepSeek-V4-Flash-FP8-4layer",
                "num_nodes": 1,
                "num_gpus_per_node": 4,
            }
            | ({"hardware": hardware} if script == "scripts/run_deepseek_v4.py" else {}),
            sandbox=tmp_path,
        )
        tokens = shlex.split(recording.commands[-1])

        assert "--custom-config-path" not in tokens
        assert ("--use-tis" in tokens) is enabled
        assert ("--use-rs" in tokens) is enabled
        if enabled:
            for flag, value in {
                "--tis-level": "token",
                "--rs-level": "token",
                "--tis-mode": "truncate",
                "--tis-lower-bound": "0.5",
                "--tis-upper-bound": "2.0",
                "--rs-veto-threshold": "1.0e-4",
                "--custom-tis-function-path": "examples.infra_features.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp",
            }.items():
                assert tokens[tokens.index(flag) + 1] == value
            assert "--tis-batch-normalize" in tokens
        else:
            assert _MIS_FLAGS.isdisjoint(tokens)
