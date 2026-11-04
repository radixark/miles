import pytest
from megatron.training import arguments as megatron_arguments
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config

from miles.utils.args.runtime import TrainerConfig
from miles.utils.args.trainer_utils import compute_trainer_config


@pytest.mark.parametrize(
    "declared,independent_expected,ppo_expected",
    [
        (None, {"actor": 8}, {"actor": 1, "critic": 1}),
        (0, {"actor": 0}, {"actor": 0, "critic": 0}),
        (3, {"actor": 3}, {"actor": 3, "critic": 3}),
        ({"critic": 5, "actor": 2}, {"actor": 2}, {"actor": 2, "critic": 5}),
    ],
)
@pytest.mark.parametrize("independent_dp", [False, True])
def test_trainer_startup_counts_resolve_per_role_and_survive_transport(
    declared: int | dict[str, int] | None,
    independent_expected: dict[str, int],
    ppo_expected: dict[str, int],
    independent_dp: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A declared startup gate overrides role-specific GPU defaults and reaches each trainer as a scalar."""
    monkeypatch.setattr(megatron_arguments, "get_device_arch_version", lambda: 10)

    args = parse_megatron_test_config(
        "--advantage-estimator",
        "grpo" if independent_dp else "ppo",
        "--actor-num-nodes",
        "2",
        "--actor-num-gpus-per-node",
        "8",
        "--critic-num-nodes",
        "1",
        "--critic-num-gpus-per-node",
        "8",
        "--tensor-model-parallel-size",
        "2",
        *(["--indep-dp", "--use-dynamic-batch-size", "--max-tokens-per-gpu", "4096"] if independent_dp else []),
    ).model_copy(update={"trainer_init_expected_num_cells": declared})

    actual: dict[str, int] = {}
    for trainer in args.raw_megatron.trainers:
        config = compute_trainer_config(args, trainer)
        restored = TrainerConfig.model_validate_json(config.model_dump_json())
        actual[trainer.trainer_id] = restored.trainer_init_expected_num_cells

    assert actual == (independent_expected if independent_dp else ppo_expected)
    assert args.trainer_init_expected_num_cells == declared
