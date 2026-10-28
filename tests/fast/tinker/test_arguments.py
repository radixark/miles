from argparse import Namespace

from miles.tinker.arguments import configure_tinker_args


class TestTinkerBatchOwnership:
    def test_preprocessing_enables_command_sized_batches_before_config_slicing(self) -> None:
        """Tinker preprocessing enables dynamic command batches before immutable component configs are built."""
        args = Namespace(
            train_backend="megatron",
            exclude_modules=None,
            target_modules=["attn"],
            use_dynamic_global_batch_size=False,
            delay_split_train_data_by_dp=False,
        )

        configure_tinker_args(args)

        assert args.use_dynamic_global_batch_size is True
        assert args.delay_split_train_data_by_dp is True
        assert args.tinker_lora_groups == ["attn"]
        assert args.target_modules == ["attn"]
