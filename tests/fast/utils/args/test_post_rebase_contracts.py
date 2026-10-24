import argparse

import pytest
import torch

from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config

from miles.utils.args.configs.custom_megatron_plugins import DsaMegatronPluginsConfig
from miles.utils.args.runtime import AllConfig, TrainerConfig
from miles.utils.args.trainer_utils import compute_trainer_config


class TestRestoredArgumentContracts:
    @pytest.mark.parametrize("backend", ["fsdp", "torchtitan"])
    def test_native_backend_payload_roundtrips_with_its_real_owner(
        self, backend: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Native parsing and trainer serialization retain the selected backend and settings."""
        monkeypatch.setattr(torch, "__version__", "2.13.0")
        config = parse_fsdp_test_config(
            "--train-backend", backend, "--ci-event-root", "/events", "--rollout-max-response-len", "128"
        )
        restored = AllConfig.model_validate_json(config.model_dump_json())
        trainer = compute_trainer_config(restored, restored.raw_megatron.trainers[0])
        trainer = TrainerConfig.model_validate_json(trainer.model_dump_json())
        assert trainer.backend.backend_name == backend
        assert trainer.ci_event_root == "/events"
        assert trainer.rollout_top_logprobs_num == 0
        assert trainer.rollout_sampling_logprobs_mode == "selected"
        assert trainer.backend.micro_batch_size == 1

    def test_score_centering_cli_preserves_its_options(self) -> None:
        """Score centering remains selectable with its original sampling defaults."""
        config = parse_fsdp_test_config("--loss-type", "score_centering", "--rollout-top-logprobs-num", "3")
        assert config.loss_type == "score_centering"
        assert config.rollout_top_logprobs_num == 3
        assert config.score_centering_is == "none"
        assert config.score_centering_tis_clip == 2.0
        assert config.score_centering_mis_low == 0.5
        assert config.score_centering_mis_high == 5.0

    @pytest.mark.parametrize("argv,expected", [([], None), (["--cp-comm-type", "p2p"], ["p2p"])])
    def test_typed_dsa_distinguishes_omitted_and_explicit_cp(
        self, argv: list[str], expected: list[str] | None
    ) -> None:
        """Typed DSA registration preserves explicit communication choices."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--cp-comm-type", nargs="+", default=["p2p"])
        DsaMegatronPluginsConfig.add_arguments(parser)
        args = parser.parse_args(argv)
        assert args.cp_comm_type == expected
        assert args.dsa_impl == "miles"
        assert args.miles_dsa_topk_backend == "canonical"


class TestNativeCheckpointStateFlags:
    @pytest.mark.parametrize(
        "flag,field",
        [("--no-load-optim", "no_load_optim"), ("--no-load-rng", "no_load_rng"), ("--no-save-optim", "no_save_optim")],
    )
    def test_checkpoint_flags_reach_the_fsdp_backend_without_flat_duplicates(self, flag: str, field: str) -> None:
        """Shared checkpoint flags survive FSDP parsing and trainer payload construction."""
        config = parse_fsdp_test_config(flag)
        trainer = compute_trainer_config(config, config.raw_megatron.trainers[0])

        assert vars(trainer.backend)[field] is True
        assert field not in TrainerConfig.model_fields
        restored = TrainerConfig.model_validate_json(trainer.model_dump_json())
        assert vars(restored.backend)[field] is True
