from __future__ import annotations
import os
from miles.utils.args.runtime import AllConfig
from miles.utils.arguments import parse_args


import argparse
import contextlib
import functools
import sys
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

from miles.utils.arguments import get_miles_extra_args_provider
from miles.utils.run_uuid import RUN_UUID_LENGTH

# megatron's own parser adds these and miles' code reads them, but a unit test builds only the miles
# extras, so nothing else would put them on the namespace
_TRAIN_BACKEND_DEFAULTS: dict[str, Any] = dict(
    disable_param_buffers_cpu_backup=False,
    lr_warmup_iters=None,
    load=None,
    ckpt_step=None,
    exit_on_missing_checkpoint=False,
)

# declared with no default and resolved after parsing, so the raw parser value is one no production
# code has ever seen; these are what the resolution settles on for a plain single-deployment run
_RESOLVED_AFTER_PARSING: dict[str, Any] = dict(
    offload_train=False,
    offload_rollout=False,
    eval_uses_snapshots=False,
    starts_inference_engines=True,
    rollout_sampling_logprobs_mode="selected",
    run_uuid="0" * RUN_UUID_LENGTH,
)


@contextlib.contextmanager
def _with_relaxed_parser_required_args(parser: argparse.ArgumentParser) -> Iterator[None]:
    required = [action for action in parser._actions if action.required]
    for action in required:
        action.required = False
    try:
        yield
    finally:
        for action in required:
            action.required = True


@functools.cache
def parser_defaults() -> dict[str, Any]:
    # a hand-written defaults dict falls behind the moment production reads a new argument, and the
    # test then dies on AttributeError somewhere deep instead of on the thing it was written for
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)

    with _with_relaxed_parser_required_args(parser), patch.object(sys, "argv", ["test"]):
        parsed, _ = parser.parse_known_args([])
    return {**_TRAIN_BACKEND_DEFAULTS, **vars(parsed), **_RESOLVED_AFTER_PARSING}


_COMMON_TEST_ARGV = ['--rollout-batch-size', '2', '--num-rollout', '1', '--actor-num-gpus-per-node', '1', '--micro-batch-size', '1']


_MEGATRON_TEST_ARGV = ['--train-backend', 'megatron', *_COMMON_TEST_ARGV, '--num-layers', '1', '--hidden-size', '128', '--num-attention-heads', '2']


def parse_megatron_test_config(*argv: str) -> AllConfig:
    return _parse_test_config([*_MEGATRON_TEST_ARGV, *argv])


def _parse_test_config(argv: list[str]) -> AllConfig:
    environment = {'RANK': '0', 'WORLD_SIZE': '1', 'LOCAL_RANK': '0', 'MILES_SCRIPT_ENV_REPORT': ''}
    with patch.object(sys, 'argv', ['test', *argv]), patch.dict(os.environ, environment):
        return parse_args()


class ConfigNamespace(argparse.Namespace):
    def __iter__(self) -> Iterator[tuple[str, Any]]:
        return iter(vars(self).items())


def make_trainer_args(*, train_backend: str = "megatron", **values: Any) -> ConfigNamespace:
    return ConfigNamespace(**(parser_defaults() | values | {"train_backend": train_backend}))


def parse_fsdp_test_config(*argv: str) -> Any:
    return _parse_test_config([*['--train-backend', 'fsdp', '--rollout-batch-size', '2', '--num-rollout', '1', '--actor-num-gpus-per-node', '1', '--micro-batch-size', '1'], *argv])


_TRAIN_BACKEND_DEFAULTS.update(fp16=False)


def make_trainer_config(**values: Any) -> Any:
    from miles.utils.args.runtime import TrainerConfig

    args = make_trainer_args(**values)
    return TrainerConfig.model_construct(**vars(args))


def resolve_parse_boundary_configs(args: argparse.Namespace) -> argparse.Namespace:
    from miles.backends.megatron_utils.megatron_config import resolve_megatron_config
    from miles.backends.sglang_utils.sglang_config import SglangConfig
    from miles.utils.args.configs.router import RouterConfig

    args.raw_megatron = resolve_megatron_config(args, base_args={})
    args.sglang = SglangConfig.parse_args(args)
    vars(args).update(RouterConfig.from_args(args))
    return args
