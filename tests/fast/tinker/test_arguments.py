from argparse import Namespace

import pytest

from miles.tinker.arguments import configure_tinker_args, validate_tinker_args


def _tinker_args(**overrides) -> Namespace:
    args = Namespace(
        train_backend="megatron",
        exclude_modules=None,
        target_modules=["attn"],
        use_dynamic_global_batch_size=False,
        delay_split_train_data_by_dp=False,
        save=None,
        save_hf=None,
        tinker_checkpoint_root="/checkpoints/tinker",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestTinkerBatchOwnership:
    def test_preprocessing_enables_command_sized_batches_before_config_slicing(self) -> None:
        """Tinker preprocessing enables dynamic command batches before immutable component configs are built."""
        args = _tinker_args()

        configure_tinker_args(args)

        assert args.use_dynamic_global_batch_size is True
        assert args.delay_split_train_data_by_dp is True
        assert args.tinker_lora_groups == ["attn"]
        assert args.target_modules == ["attn"]


class TestTinkerCheckpointOwnership:
    @pytest.mark.parametrize(
        "field,value,flag",
        [("save", "/ckpt", "--save"), ("save_hf", "/hf", "--save-hf")],
    )
    def test_trainer_driven_saves_are_rejected_instead_of_ignored(self, field: str, value, flag: str) -> None:
        """Trainer save flags are inert under the gateway, so accepting them would mislead."""
        with pytest.raises(AssertionError, match=flag):
            validate_tinker_args(_tinker_args(**{field: value}))

    def test_checkpoint_root_is_required(self) -> None:
        with pytest.raises(AssertionError, match="--tinker-checkpoint-root"):
            validate_tinker_args(_tinker_args(tinker_checkpoint_root=None))
