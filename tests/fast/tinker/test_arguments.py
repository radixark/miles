from argparse import ArgumentParser, Namespace
from types import SimpleNamespace

import pytest

from miles.ray.placement_group import _get_placement_group_layout
from miles.ray.specs.train import ACTOR_ROLE
from miles.tinker.arguments import configure_tinker_args, configure_tinker_mode, validate_tinker_args
from miles.utils.args.configs.tinker import TinkerConfig
from miles.utils.arguments import _apply_custom_config


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
        tinker_mode=None,
        debug_train_only=False,
        debug_rollout_only=False,
        load_debug_rollout_data=None,
        eval_num_gpus=0,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestTinkerModes:
    @pytest.mark.parametrize("mode,total,offset", [("training", 16, 16), ("inference", 8, 0), ("combined", 24, 16)])
    def test_public_mode_reserves_only_selected_component_gpus(self, mode: str, total: int, offset: int) -> None:
        parser = ArgumentParser()
        TinkerConfig.add_arguments(parser)
        parsed = parser.parse_args(["--tinker-mode", mode])
        args = _tinker_args(
            tinker_mode=parsed.tinker_mode,
            deploy_component="all",
            actor_num_nodes=2,
            actor_num_gpus_per_node=8,
            rollout_num_gpus=8,
            rollout_external=False,
            colocate=False,
            raw_megatron=SimpleNamespace(trainers=[SimpleNamespace(role=ACTOR_ROLE)]),
        )

        configure_tinker_mode(args)

        assert _get_placement_group_layout(args) == (total, offset)

    @pytest.mark.parametrize(
        "overrides,expected",
        [({}, "combined"), ({"debug_train_only": True}, "training"), ({"debug_rollout_only": True}, "inference")],
    )
    def test_omitted_mode_preserves_existing_component_selection(self, overrides: dict, expected: str) -> None:
        args = _tinker_args(**overrides)

        configure_tinker_mode(args)

        assert args.tinker_mode == expected

    @pytest.mark.parametrize(
        "overrides",
        [
            {"tinker_mode": "training", "debug_rollout_only": True},
            {"tinker_mode": "inference", "debug_train_only": True},
            {"tinker_mode": "combined", "load_debug_rollout_data": "/data/rollout.pt"},
            {"debug_train_only": True, "debug_rollout_only": True},
        ],
    )
    def test_conflicting_component_selection_fails_before_allocation(self, overrides: dict) -> None:
        with pytest.raises(ValueError, match="conflicts"):
            configure_tinker_mode(_tinker_args(**overrides))

    def test_training_mode_rejects_an_inference_eval_allocation(self) -> None:
        with pytest.raises(ValueError, match="--eval-num-gpus 0"):
            configure_tinker_mode(_tinker_args(tinker_mode="training", eval_num_gpus=8))

    def test_custom_config_mode_replaces_cli_selection_before_resolution(self) -> None:
        args = _tinker_args(tinker_mode="combined")
        _apply_custom_config(args, {"tinker_mode": "training"})

        configure_tinker_mode(args)

        assert args.debug_train_only
        assert not args.debug_rollout_only


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
