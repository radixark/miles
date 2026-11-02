from collections.abc import Callable
from typing import Any

import pytest
import torch
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config

from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace
from miles.utils.args.runtime import TrainerConfig
from miles.utils.args.trainer_utils import compute_trainer_config


def _parsed_trainer(*argv: str) -> TrainerConfig:
    config = parse_megatron_test_config(*argv)
    return compute_trainer_config(config, config.raw_megatron.trainers[0])


def _tokenizer_initialization(monkeypatch: pytest.MonkeyPatch) -> TrainerConfig:
    from miles.backends.megatron_utils import initialize

    args = _parsed_trainer()
    with args.backend.mutable():
        args.backend.rank = 0
    for name in (
        "set_args",
        "_initialize_distributed",
        "create_indep_dp_group",
        "create_megatron_parallel_state",
        "set_parallel_state",
        "set_random_seed_from_args",
        "register_hf_config_aliases",
        "init_num_microbatches_calculator",
    ):
        monkeypatch.setattr(initialize, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(initialize.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(initialize.dist, "get_world_size", lambda: 1)
    return args


def _tokenizer_builder(*, fails: bool) -> Callable[[MegatronArgsNamespace], None]:
    def build(backend: MegatronArgsNamespace) -> None:
        backend.padded_vocab_size = 128
        if fails:
            raise RuntimeError("tokenizer construction failed")

    return build


def _checkpoint_loader(*, backend: MegatronArgsNamespace, fails: bool) -> Callable[..., tuple[int, int]]:
    def load(**kwargs: Any) -> tuple[int, int]:
        backend.consumed_train_samples = 64
        if fails:
            raise RuntimeError("checkpoint loading failed")
        return 7, 0

    return load


def _model_loader(*, backend: MegatronArgsNamespace, fails: bool) -> Callable[..., list[torch.nn.Module]]:
    def load(provider: Callable[..., object], model_type: object) -> list[torch.nn.Module]:
        backend.model_type = model_type
        if fails:
            raise RuntimeError("model construction failed")
        return [torch.nn.Linear(1, 1)]

    return load
