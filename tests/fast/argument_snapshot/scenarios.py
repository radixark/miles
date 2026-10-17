import argparse
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from tests.fast.argument_snapshot.schema import snapshot_parser


@dataclass(frozen=True)
class _Scenario:
    backend: str
    arguments: tuple[str, ...] = ()
    legacy: bool = False
    custom: bool = False


def capture_scenarios(selected: list[str] | None = None) -> dict[str, Any]:
    scenarios = {
        "megatron": _Scenario(backend="megatron"),
        "fsdp": _Scenario(backend="fsdp"),
        "fully_async": _Scenario(backend="megatron", arguments=("--fully-async",)),
        "legacy": _Scenario(backend="megatron", legacy=True),
        "custom_megatron": _Scenario(backend="megatron", custom=True),
        "custom_fsdp": _Scenario(backend="fsdp", custom=True),
    }
    for name, flag in {
        "rollout": "--rollout-function-path",
        "generate": "--custom-generate-function-path",
        "inference": "--custom-inference-engine-provider-path",
    }.items():
        scenarios[f"hook_{name}"] = _Scenario(
            backend="megatron", arguments=(flag, "tests.fast.argument_snapshot.scenarios._Hook")
        )
        scenarios[f"legacy_hook_{name}"] = _Scenario(
            backend="megatron", arguments=(flag, "tests.fast.argument_snapshot.scenarios._Hook"), legacy=True
        )
    scenarios["missing_hook_module"] = _Scenario(
        backend="megatron", arguments=("--custom-generate-function-path", "miles_snapshot_missing.module")
    )
    scenarios["hook_without_arguments"] = _Scenario(
        backend="megatron", arguments=("--custom-generate-function-path", "builtins.str")
    )
    scenarios["invalid_hook_path"] = _Scenario(
        backend="megatron", arguments=("--custom-generate-function-path", "invalid_snapshot_path")
    )
    scenarios["hook_function"] = _Scenario(
        backend="megatron",
        arguments=("--custom-generate-function-path", "tests.fast.argument_snapshot.scenarios._hook_function"),
    )
    scenarios["hook_fsdp"] = _Scenario(
        backend="fsdp",
        arguments=("--custom-inference-engine-provider-path", "tests.fast.argument_snapshot.scenarios._Hook"),
    )
    scenarios["megatron_repeat"] = _Scenario(backend="megatron")
    names = list(scenarios) if selected is None else selected
    if unknown := set(names) - scenarios.keys():
        raise ValueError(f"Unknown snapshot scenarios: {sorted(unknown)}; available: {list(scenarios)}")
    return {name: _capture_scenario(scenarios[name]) for name in names}


def _capture_scenario(scenario: _Scenario) -> dict[str, Any]:
    from miles.utils.arguments import parse_args_and_get_parser

    arguments = ["--rollout-batch-size", "2", "--train-backend", "fsdp" if scenario.backend == "fsdp" else "megatron"]
    arguments.extend(["--num-rollout", "1", "--actor-num-gpus-per-node", "1", "--micro-batch-size", "1"])
    if scenario.backend == "megatron":
        arguments.extend(["--num-layers", "1", "--hidden-size", "128", "--num-attention-heads", "2"])
    arguments.extend(scenario.arguments)
    with _environment(arguments=arguments, legacy=scenario.legacy):
        custom = _custom_arguments if scenario.custom else None
        _, parser = parse_args_and_get_parser(add_custom_arguments=custom)
        parsed = {"minimal": vars(parser.parse_args(arguments))}
        variants = {
            "lora_disabled": ["--no-sglang-lora-use-virtual-experts"],
            "sglang_alias": ["--sglang-tp-size", "2"],
            "eval_true": ["--eval-sglang-enable-metrics"],
            "eval_false": ["--no-eval-sglang-enable-metrics"],
        }
        for name, extra in variants.items():
            parsed[name] = vars(parser.parse_args(arguments + extra))
        return {"argv": arguments, "legacy": scenario.legacy, "schema": snapshot_parser(parser), "parsed": parsed}


def _custom_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--snapshot-custom", type=int, default=17)
    for action in parser._actions:
        if "--padded-vocab-size" in action.option_strings:
            action.default = 1024
            break
    else:
        parser.add_argument("--padded-vocab-size", type=int, default=1024)
    return parser


class _Hook:
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--snapshot-hook", type=int, default=23)


def _hook_function() -> None:
    pass


_hook_function.add_arguments = _Hook.add_arguments


@contextmanager
def _environment(*, arguments: list[str], legacy: bool) -> Iterator[None]:
    original_argv = sys.argv
    values = {
        "MILES_USE_LEGACY_ROLLOUT_V1": str(int(legacy)),
        "PROMETHEUS_PORT": "9090",
        "MILES_SCRIPT_ENV_REPORT": "",
    }
    original_environment = {name: os.environ.get(name) for name in values}
    sys.argv = ["argument-snapshot", *arguments]
    os.environ.update(values)
    try:
        yield
    finally:
        sys.argv = original_argv
        for name, value in original_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
