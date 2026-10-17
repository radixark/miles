import sys
from argparse import Namespace

import pytest
from tools.convert_hf_to_torch_dist import _configure_pipeline_parallel, _ConversionConfig, get_args


class TestConversionPipelineParallel:
    @pytest.mark.parametrize(
        "world_size,num_layers,pp_size,ep_size,keep_pp1,expected_pp,expected_last",
        [
            (4, 8, 1, 1, False, 4, 2),
            (4, 8, 2, 1, False, 2, None),
            (4, 8, 1, 1, True, 1, None),
            (8, 8, 1, 2, False, 4, 2),
            (8, 5, 1, 1, False, 2, 2),
            (4, 8, 1, 3, False, 1, None),
        ],
    )
    def test_pipeline_layout_respects_rank_groups_and_layer_capacity(
        self,
        monkeypatch: pytest.MonkeyPatch,
        world_size: int,
        num_layers: int,
        pp_size: int,
        ep_size: int,
        keep_pp1: bool,
        expected_pp: int,
        expected_last: int | None,
    ) -> None:
        """Automatic stages fit layers and whole expert groups without overriding explicit layouts."""
        monkeypatch.setenv("WORLD_SIZE", str(world_size))
        monkeypatch.delenv("CONVERT_KEEP_PP1", raising=False)
        if keep_pp1:
            monkeypatch.setenv("CONVERT_KEEP_PP1", "1")
        args = Namespace(
            num_layers=num_layers,
            pipeline_model_parallel_size=pp_size,
            tensor_model_parallel_size=1,
            context_parallel_size=1,
            expert_tensor_parallel_size=None,
            expert_model_parallel_size=ep_size,
            decoder_last_pipeline_num_layers=None,
        )

        _configure_pipeline_parallel(args, world_size=world_size)

        assert args.pipeline_model_parallel_size == expected_pp
        assert args.decoder_last_pipeline_num_layers == expected_last


class TestConversionParser:
    def test_conversion_defaults_reach_backend_validation_and_model_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The real parser validates conversion topology and returns a usable model configuration."""
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("RANK", "0")
        monkeypatch.delenv("CONVERT_KEEP_PP1", raising=False)
        monkeypatch.delenv("MILES_BACKEND", raising=False)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "convert_hf_to_torch_dist",
                "--hf-checkpoint",
                "unused-hf-checkpoint",
                "--num-layers",
                "8",
                "--hidden-size",
                "128",
                "--num-attention-heads",
                "2",
                "--micro-batch-size",
                "3",
                "--save-interval",
                "7",
            ],
        )
        original_argv = sys.argv

        args = get_args()

        assert isinstance(args, _ConversionConfig)
        assert args.train_backend == "megatron"
        assert args.backend.world_size == 4
        assert args.backend.pipeline_model_parallel_size == 4
        assert args.backend.decoder_last_pipeline_num_layers == 2
        assert args.backend.global_batch_size == 4
        assert args.backend.micro_batch_size == 1
        assert args.backend.save_interval == 1
        assert sys.argv is original_argv
