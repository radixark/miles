from argparse import Namespace
from typing import Any

from tests.fast.fixtures.args_fixtures import parser_defaults

from miles.backends.sglang_utils.sglang_api_client import WorkerType
from miles.backends.sglang_utils.sglang_config import (
    ModelConfig,
    ServerGroupConfig,
    SglangConfig,
    SglangScalingConfig,
    _compute_raw_sglang_config,
)


def resolve_sglang_config(args: Namespace) -> SglangConfig:
    config, _ = resolve_sglang_config_and_scaling(args)
    return config


def resolve_sglang_config_and_scaling(args: Namespace) -> tuple[SglangConfig, SglangScalingConfig]:
    return SglangConfig.resolve(raw=_compute_raw_sglang_config(args), args=args, base_args={})


def make_sglang_config(**base_args: Any) -> SglangConfig:
    group = ServerGroupConfig(worker_type=WorkerType.REGULAR, num_gpus_per_engine=1, needs_offload=False)
    model = ModelConfig(name="default", model_path=None, server_groups=[group], update_weights=True)
    return SglangConfig(models=[model], base_args=base_args)


def with_parser_defaults_and_sglang_config(values: dict[str, Any]) -> dict[str, Any]:
    values = {**parser_defaults(), **values}
    config, _ = SglangConfig.parse_args(Namespace(**values))
    return values | {"sglang": config}
