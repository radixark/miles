from __future__ import annotations

import argparse
import contextlib
import functools
import os
import sys
from argparse import Namespace
from collections.abc import Iterator
from typing import Any, TypeVar
from unittest.mock import patch

from miles.backends.megatron_utils.megatron_config import resolve_megatron_config
from miles.backends.sglang_utils.sglang_config import SglangConfig
from miles.utils.args.configs.router import RouterConfig
from miles.utils.args.runtime import AllConfig, TrainerConfig
from miles.utils.arguments import _compute_init_expected_num_cells, get_miles_extra_args_provider, parse_args
from miles.utils.run_uuid import RUN_UUID_LENGTH

_ConfigT = TypeVar("_ConfigT", AllConfig, TrainerConfig)

# megatron's own parser adds these and miles' code reads them, but a unit test builds only the miles
# extras, so nothing else would put them on the namespace
_TRAIN_BACKEND_DEFAULTS: dict[str, Any] = dict(
    disable_param_buffers_cpu_backup=False,
    fp16=False,
    lr_warmup_iters=None,
    load=None,
    num_layers=None,
)

# declared with no default and resolved after parsing, so the raw parser value is one no production
# code has ever seen; these are what the resolution settles on for a plain single-deployment run
_RESOLVED_AFTER_PARSING: dict[str, Any] = dict(
    offload_train=False,
    offload_rollout=False,
    eval_uses_snapshots=False,
    starts_inference_engines=True,
    use_critic=False,
    rollout_external=False,
    multi_lora=False,
    use_sampling_support_replay=False,
    run_uuid="0" * RUN_UUID_LENGTH,
)


class ConfigNamespace(argparse.Namespace):
    def __iter__(self) -> Iterator[tuple[str, Any]]:
        return iter(vars(self).items())


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


def resolve_parse_boundary_configs(args: Namespace) -> Namespace:
    args.raw_megatron = resolve_megatron_config(args, base_args={})
    args.sglang, args.sglang_scaling = SglangConfig.parse_args(args)
    vars(args).update(RouterConfig.from_args(args))
    args.init_expected_num_cells = _compute_init_expected_num_cells(
        args, sglang=args.sglang, sglang_scaling=args.sglang_scaling
    )
    return args


_COMMON_TEST_ARGV = [
    "--rollout-batch-size",
    "2",
    "--num-rollout",
    "1",
    "--actor-num-gpus-per-node",
    "1",
    "--micro-batch-size",
    "1",
]

_MEGATRON_TEST_ARGV = [
    "--train-backend",
    "megatron",
    *_COMMON_TEST_ARGV,
    "--num-layers",
    "1",
    "--hidden-size",
    "128",
    "--num-attention-heads",
    "2",
]

_FSDP_TEST_ARGV = ["--train-backend", "fsdp", *_COMMON_TEST_ARGV]


def parse_megatron_test_config(*argv: str) -> AllConfig:
    return _parse_test_config([*_MEGATRON_TEST_ARGV, *argv])


def parse_fsdp_test_config(*argv: str) -> AllConfig:
    return _parse_test_config([*_FSDP_TEST_ARGV, *argv])


def _parse_test_config(argv: list[str]) -> AllConfig:
    environment = {"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0", "MILES_SCRIPT_ENV_REPORT": ""}
    with patch.object(sys, "argv", ["test", *argv]), patch.dict(os.environ, environment):
        return parse_args()


def make_trainer_args(*, train_backend: str = "megatron", **values: Any) -> ConfigNamespace:
    from miles.backends.fsdp_utils.config import FsdpArgsNamespace
    from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace
    from miles.utils.args.configs.backend_fields import TrainerBackendTraitConfig
    from miles.utils.args.runtime import TrainerConfig

    values = {**parser_defaults(), **values, "train_backend": train_backend}
    trainer_fields = TrainerConfig.model_fields.keys() - TrainerBackendTraitConfig.model_fields.keys()
    backend_cls = MegatronArgsNamespace if train_backend == "megatron" else FsdpArgsNamespace
    backend = backend_cls(**{name: value for name, value in values.items() if name not in trainer_fields})
    trainer = {name: value for name, value in values.items() if name in trainer_fields}
    return ConfigNamespace(**trainer, backend=backend)


def with_backend_values(args: ConfigNamespace, **values: Any) -> ConfigNamespace:
    backend = type(args.backend)(**(vars(args.backend) | values))
    return ConfigNamespace(**(vars(args) | {"backend": backend}))


def make_trainer_config(**values: Any) -> Any:
    from miles.utils.args.runtime import TrainerConfig

    args = make_trainer_args(**values)
    return TrainerConfig.model_construct(**vars(args))


def replace_config_values(config: _ConfigT, **updates: Any) -> _ConfigT:
    fields = type(config).model_fields
    backend_updates = {name: value for name, value in updates.items() if name not in fields}
    top_level = {name: value for name, value in updates.items() if name in fields}
    if backend_updates:
        top_level.update(_replace_backend_values(config, **backend_updates))
    return config.model_copy(update=top_level)


def _replace_backend_values(config: AllConfig | TrainerConfig, **updates: Any) -> dict[str, Any]:
    if isinstance(config, TrainerConfig):
        field, backend = "backend", config.backend
    elif config.train_backend == "megatron":
        field, backend = "raw_megatron", None
    else:
        field, backend = "raw_fsdp", config.raw_fsdp
    values = dict(config.raw_megatron.base_args) if backend is None else vars(backend)
    unknown = updates.keys() - values.keys()
    assert not unknown, f"{sorted(unknown)} are neither {type(config).__name__} fields nor backend arguments"
    if backend is None:
        return {field: config.raw_megatron.model_copy(update={"base_args": values | updates})}
    return {field: type(backend)(**(values | updates))}
