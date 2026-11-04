import argparse

import pytest

from miles.utils.args.custom_function import (
    CustomFunctionConfig,
    add_user_provided_function_arguments,
    resolve_custom_function_configs,
)
from miles.utils.args.schema import BaseConfig


class _RunConfig(BaseConfig):
    first: CustomFunctionConfig


class TestExampleHookRegistration:
    @pytest.mark.parametrize(
        "path,options,field,expected",
        [
            (
                "examples.experimental.hud.rollout.generate",
                ["--hud-env-dir", "/hud", "--hud-snapshot-name", "snapshot", "--hud-max-steps", "13"],
                "hud_max_steps",
                13,
            ),
            (
                "examples.experimental.hud.rollout.reward_func",
                ["--hud-env-dir", "/hud", "--hud-snapshot-name", "snapshot", "--hud-max-sandboxes", "5"],
                "hud_max_sandboxes",
                5,
            ),
            (
                "examples.geo3k_vlm.multi_turn.rollout.generate",
                ["--max-turns", "7", "--rollout-interaction-env-path", "custom.env"],
                "rollout_interaction_env_path",
                "custom.env",
            ),
            (
                "examples.infra_features.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp",
                ["--tis-level", "sequence", "--use-rs"],
                "tis_level",
                "sequence",
            ),
            (
                "examples.infra_features.train_infer_mismatch_helper.mis.compute_mis_weights_fsdp",
                ["--tis-mode", "clip", "--tis-upper-bound", "3.5"],
                "tis_upper_bound",
                3.5,
            ),
        ],
    )
    def test_example_hook_discovers_its_real_schema_and_transports_cli_values(
        self, monkeypatch: pytest.MonkeyPatch, path: str, options: list[str], field: str, expected: object
    ) -> None:
        """Every migrated example hook owns its custom CLI fields through JSON transport."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--first", required=True)
        monkeypatch.setattr("sys.argv", ["test", "--first", path, *options])
        add_user_provided_function_arguments(parser, config_class=_RunConfig)
        args = parser.parse_args()

        resolve_custom_function_configs(args, config_class=_RunConfig)
        restored = CustomFunctionConfig.model_validate_json(args.first.model_dump_json())

        assert restored.path == path
        assert dict(restored.config)[field] == expected
        assert restored.config == args.first.config
        assert vars(args).keys() == {"first"}

    def test_mis_cli_rejects_invalid_tis_choices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The migrated MIS hook retains its finite CLI choice contract."""
        parser = argparse.ArgumentParser()
        path = "examples.infra_features.train_infer_mismatch_helper.mis.compute_mis_weights_fsdp"
        parser.add_argument("--first", default=path)
        monkeypatch.setattr("sys.argv", ["test"])
        add_user_provided_function_arguments(parser, config_class=_RunConfig)

        with pytest.raises(SystemExit):
            parser.parse_args(["--tis-mode", "unsupported"])
