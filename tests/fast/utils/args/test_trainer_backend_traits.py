import pytest
import torch
from tests.fast.fixtures.args_fixtures import _parse_test_config, parse_fsdp_test_config, parse_megatron_test_config

from miles.utils.args.runtime import TrainerConfig
from miles.utils.args.trainer_utils import compute_trainer_config


class TestTrainerBackendTraits:
    @pytest.mark.parametrize("backend", ["megatron", "fsdp"])
    def test_shared_training_traits_are_stored_once_in_the_transported_backend(self, backend: str) -> None:
        """Each backend receives shared training settings without duplicating their storage in the trainer."""
        parse = {"megatron": parse_megatron_test_config, "fsdp": parse_fsdp_test_config}[backend]
        config = parse(
            "--seed",
            "37",
            "--global-batch-size",
            "2",
            "--wandb-project",
            "backend-traits",
            "--distributed-timeout-minutes",
            "12",
        )
        before = config.model_dump(mode="json")
        trainer = compute_trainer_config(config, config.raw_megatron.trainers[0])
        restored = TrainerConfig.model_validate_json(trainer.model_dump_json())
        values = vars(restored.backend)
        expected = {
            "seed": 37,
            "global_batch_size": 2,
            "micro_batch_size": 1,
            "wandb_project": "backend-traits",
            "distributed_timeout_minutes": 12,
        }

        assert expected.items() <= values.items()
        assert not expected.keys() & TrainerConfig.model_fields.keys()
        assert config.model_dump(mode="json") == before

    @pytest.mark.parametrize(
        "parallel,splits",
        [
            ((), []),
            (
                (
                    "--pipeline-model-parallel-size",
                    "2",
                    "--decoder-first-pipeline-num-layers",
                    "3",
                    "--decoder-last-pipeline-num-layers",
                    "2",
                ),
                [3],
            ),
        ],
        ids=["pp1", "pp2-uneven"],
    )
    def test_the_replay_layer_splits_follow_the_megatron_pipeline_layout(self, parallel, splits) -> None:
        """The layers where pipeline stages meet are derived at launch and reach the rollout and every trainer."""
        config = parse_megatron_test_config("--num-layers", "5", "--actor-num-gpus-per-node", "2", *parallel)
        trainer = compute_trainer_config(config, config.raw_megatron.trainers[0])

        assert config.replay_layer_splits == splits
        assert trainer.backend.replay_layer_splits == splits

    def test_a_non_megatron_backend_has_no_replay_layer_splits(self) -> None:
        assert parse_fsdp_test_config().replay_layer_splits == []

    def test_a_non_megatron_output_store_run_resolves_its_trainer(self) -> None:
        """FSDP has no pipeline arguments, which resolving a store run's trainer must not read."""
        config = parse_fsdp_test_config(
            "--use-output-store", "--use-rollout-routing-replay", "--object-store-backend", "mooncake"
        )
        trainer = compute_trainer_config(config, config.raw_megatron.trainers[0])

        assert trainer.backend.replay_layer_splits == []


def _parse_torchtitan_test_config(monkeypatch: pytest.MonkeyPatch, *argv: str):
    monkeypatch.setattr(torch, "__version__", "2.14.0")  # the backend's floor; CI's torch is older
    base = ["--train-backend", "torchtitan", "--rollout-batch-size", "2", "--num-rollout", "1"]
    base += ["--actor-num-gpus-per-node", "1", "--rollout-max-response-len", "1024"]
    return _parse_test_config([*base, *argv])


class TestWandbExpName:
    @pytest.mark.parametrize("backend", ["megatron", "fsdp", "torchtitan"])
    @pytest.mark.parametrize("argv,expected", [((), None), (("--wandb-exp-name", "run-a"), "run-a")])
    def test_the_flag_reaches_the_orchestrator_and_every_trainer(self, monkeypatch, backend, argv, expected) -> None:
        """--wandb-exp-name is parsed for every backend and survives the trainer config round trip."""
        if backend == "torchtitan":
            config = _parse_torchtitan_test_config(monkeypatch, *argv)
        else:
            config = {"megatron": parse_megatron_test_config, "fsdp": parse_fsdp_test_config}[backend](*argv)
        trainer = compute_trainer_config(config, config.raw_megatron.trainers[0])
        restored = TrainerConfig.model_validate_json(trainer.model_dump_json())

        assert config.wandb_exp_name == expected
        assert restored.wandb_exp_name == expected
        assert restored.backend.wandb_exp_name == expected
