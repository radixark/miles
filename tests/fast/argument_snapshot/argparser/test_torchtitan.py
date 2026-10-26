import sys
from argparse import ArgumentParser

import pytest
import torch

from miles.utils.arguments import parse_args_and_get_parser


def test_torchtitan_returns_its_parser_after_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parser capture must preserve TorchTitan startup and expose backend options."""
    monkeypatch.setattr(torch, "__version__", "2.13.0")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "snapshot-test",
            "--train-backend",
            "torchtitan",
            "--rollout-batch-size",
            "2",
            "--num-rollout",
            "1",
            "--actor-num-gpus-per-node",
            "1",
            "--micro-batch-size",
            "1",
            "--rollout-max-response-len",
            "1024",
        ],
    )
    args, parser = parse_args_and_get_parser()

    assert args.train_backend == "torchtitan"
    assert isinstance(parser, ArgumentParser)
    assert parser.parse_args(sys.argv[1:] + ["--titan-model-flavor", "8B"]).titan_model_flavor == "8B"
