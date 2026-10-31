import argparse

import pytest
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config

from miles.utils.args.runtime import AllConfig
from miles.utils.arguments import _validate_argument_ownership


class TestArgumentOwnership:
    def test_declared_backend_and_sdk_options_have_explicit_owners(self) -> None:
        """Backend and declared SGLang options may coexist with typed Miles and wrapper options."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--sglang-test-option")
        parser.add_argument("--eval-sglang-test-option")
        args = argparse.Namespace(
            train_backend="megatron",
            backend_test_option=7,
            sglang_test_option="train",
            eval_sglang_test_option="eval",
            custom_config_path="custom.yaml",
            megatron_config="megatron.yaml",
        )
        before = vars(args).copy()

        _validate_argument_ownership(args, parser=parser, training_backend_arg_names={"backend_test_option"})

        assert vars(args) == before

    @pytest.mark.parametrize("name", ["unowned", "sglang_unregistered", "eval_sglang_unregistered"])
    def test_unowned_fields_are_not_accepted_merely_for_looking_like_sdk_options(self, name: str) -> None:
        """Only registered SDK destinations bypass typed field ownership validation."""
        args = argparse.Namespace(**{name: 7})

        with pytest.raises(ValueError, match=f"no configuration owner.*{name}"):
            _validate_argument_ownership(args, parser=argparse.ArgumentParser(), training_backend_arg_names=set())

        assert vars(args) == {name: 7}

    def test_tinker_cli_options_survive_the_strict_all_config_transport(self) -> None:
        """Tinker gateway options retain their explicit values after typed parser transport."""
        config = parse_megatron_test_config(
            "--tinker-server-host",
            "127.0.0.8",
            "--tinker-server-port",
            "11882",
            "--tinker-base-model",
            "advertised",
            "--tinker-checkpoint-root",
            "/checkpoints/tinker",
        )
        restored = AllConfig.model_validate_json(config.model_dump_json())

        assert restored.tinker_server_host == "127.0.0.8"
        assert restored.tinker_server_port == 11882
        assert restored.tinker_base_model == "advertised"
        assert restored.tinker_checkpoint_root == "/checkpoints/tinker"


class TestTinkerServeParsing:
    @pytest.mark.parametrize("independent", [False, True])
    def test_tinker_preprocessing_participates_in_normal_validation(
        self, monkeypatch: pytest.MonkeyPatch, independent: bool
    ) -> None:
        """The serve entry retains gateway fields and rejects independent DP with Tinker batching."""
        from miles.tinker.arguments import configure_tinker_args
        from miles.utils.arguments import parse_args

        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("MILES_SCRIPT_ENV_REPORT", "")
        argv = [
            "test",
            "--train-backend",
            "megatron",
            "--rollout-batch-size",
            "2",
            "--num-rollout",
            "1",
            "--actor-num-gpus-per-node",
            "1",
            "--micro-batch-size",
            "1",
            "--num-layers",
            "1",
            "--hidden-size",
            "128",
            "--num-attention-heads",
            "2",
            "--use-dynamic-batch-size",
            "--max-tokens-per-gpu",
            "128",
            "--tinker-server-port",
            "11883",
        ]
        if independent:
            argv.append("--indep-dp")
        monkeypatch.setattr("sys.argv", argv)

        if independent:
            with pytest.raises(AssertionError, match="indep-dp does not support --use-dynamic-global-batch-size"):
                parse_args(entry="serve", preprocess_args=configure_tinker_args)
        else:
            config = parse_args(entry="serve", preprocess_args=configure_tinker_args)
            assert config.entry == "serve"
            assert config.tinker_server_port == 11883
            assert config.use_dynamic_global_batch_size
            assert config.delay_split_train_data_by_dp
            assert config.tinker_lora_groups == ["attn", "mlp", "unembed"]


class TestRouterSdkOwnership:
    def test_real_router_sdk_fields_are_recognized_as_owned(self) -> None:
        """The external router parser remains accepted without copying its options into AllConfig."""
        from sglang_router.launch_router import RouterArgs

        parser = argparse.ArgumentParser(add_help=False)
        RouterArgs.add_cli_args(parser, use_router_prefix=True, exclude_host_port=True)
        args = parser.parse_args([])
        before = vars(args).copy()
        assert vars(args).keys() - AllConfig.model_fields.keys()

        _validate_argument_ownership(args, parser=parser, training_backend_arg_names=set())

        assert vars(args) == before
