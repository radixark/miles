"""Feed a Primus experiment YAML into Miles' Megatron argument namespace.

Primus owns the config layer (preset inheritance, model presets, env substitution)
and Miles owns the Megatron install plus the RL loop, so the bridge reads only
Primus' YAML/params code. It never puts Primus' vendored `third_party/Megatron-LM`
on sys.path: every value is resolved against the Megatron that Miles already imports.

Precedence is explicit CLI flag > Primus YAML > Miles/Megatron default. That falls
out of injecting the YAML as argparse *defaults* before Megatron parses sys.argv.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Primus keys that name the config itself rather than a training knob.
_META_KEYS = frozenset({"name", "framework", "config", "model", "modules", "overrides", "stage"})

_SUPPORTED_FRAMEWORK = "megatron"

# Runs every Primus patch phase in the actor between Megatron init and model build.
PRIMUS_MEGATRON_INIT_PATH = "miles.backends.primus_utils.megatron_init.primus_megatron_init"

# Primus configs carry ~130 keys Miles' Megatron does not define; list a sample, not all.
_WARN_PREVIEW = 12

ExtraArgsProvider = Callable[[argparse.ArgumentParser], argparse.ArgumentParser]


def add_primus_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(title="primus")
    group.add_argument(
        "--primus-config",
        type=str,
        default=None,
        help="Path to a Primus experiment YAML (e.g. Primus/examples/megatron/configs/MI300X/"
        "llama3.1_8B-BF16-pretrain.yaml). Its resolved parameters become defaults for the "
        "Megatron arguments, so any flag passed explicitly on the command line still wins. "
        "Requires --train-backend megatron.",
    )
    group.add_argument(
        "--primus-module",
        type=str,
        default="pre_trainer",
        help="Module key to read from the Primus YAML's `modules` section.",
    )
    group.add_argument(
        "--primus-override",
        type=str,
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help="Primus-level overrides applied to the YAML parameters before they reach Megatron, "
        "using Primus' own override syntax (e.g. --primus-override train_iters=100 lr=1e-5).",
    )
    group.add_argument(
        "--primus-path",
        type=str,
        default=None,
        help="Primus checkout to import when the `primus` package is not installed. "
        "Falls back to the PRIMUS_PATH environment variable.",
    )
    group.add_argument(
        "--primus-strict-config",
        action="store_true",
        help="Fail instead of warning when the Primus YAML carries parameters that Miles' "
        "Megatron does not define.",
    )
    group.add_argument(
        "--primus-patches",
        type=str,
        default="kernels",
        help="Which Primus Megatron patches to apply: 'kernels' (default) for the curated "
        "kernel/numerics set, 'all' for every registered patch, 'none' to disable, or a "
        "comma-separated list of patch ids. Miles always keeps ownership of orchestration "
        "(checkpointing, data, logging), so 'all' is for experiments only.",
    )
    return parser


@dataclass
class PrimusConfigRequest:
    config_path: str
    module_name: str = "pre_trainer"
    overrides: list[str] = field(default_factory=list)
    primus_path: str | None = None
    strict: bool = False

    @classmethod
    def from_argv(cls, argv: list[str] | None = None) -> PrimusConfigRequest | None:
        """Read the --primus-* flags before the full parser exists, or None if unused."""
        parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        add_primus_arguments(parser)
        known, _ = parser.parse_known_args(argv)
        if not known.primus_config:
            return None
        return cls(
            config_path=known.primus_config,
            module_name=known.primus_module,
            overrides=list(known.primus_override),
            primus_path=known.primus_path,
            strict=known.primus_strict_config,
        )


@dataclass
class PrimusConfigReport:
    """What became of each Primus parameter."""

    applied: dict[str, Any] = field(default_factory=dict)
    carried: dict[str, Any] = field(default_factory=dict)


def ensure_primus_importable(primus_path: str | None) -> None:
    search_path = primus_path or os.environ.get("PRIMUS_PATH")
    if search_path:
        resolved = os.path.abspath(search_path)
        if not os.path.isdir(resolved):
            raise FileNotFoundError(f"Primus checkout not found: {resolved}")
        if resolved not in sys.path:
            sys.path.insert(0, resolved)

    importlib.invalidate_caches()
    if importlib.util.find_spec("primus") is None:
        raise ModuleNotFoundError(
            "--primus-config needs the `primus` package. Install it "
            '(pip install "primus" --extra-index-url https://amd-agi.github.io/Primus/simple/) '
            "or point --primus-path / PRIMUS_PATH at a Primus checkout."
        )


def _load_params(request: PrimusConfigRequest) -> dict[str, Any]:
    """Resolve the Primus YAML (presets, env substitution, overrides) to a flat param dict."""
    from pathlib import Path

    ensure_primus_importable(request.primus_path)

    from primus.core.config.merge_utils import deep_merge
    from primus.core.config.primus_config import (
        get_module_config,
        get_module_names,
        load_primus_config,
    )
    from primus.core.utils.arg_utils import parse_cli_overrides
    from primus.core.utils.yaml_utils import nested_namespace_to_dict

    config_path = Path(request.config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Primus config not found: {config_path}")

    primus_config = load_primus_config(config_path)
    module_config = get_module_config(primus_config, request.module_name)
    if module_config is None:
        raise ValueError(
            f"Primus config {config_path} has no module '{request.module_name}'. "
            f"Available modules: {', '.join(get_module_names(primus_config)) or 'none'}. "
            f"Select one with --primus-module."
        )

    framework = getattr(module_config, "framework", None)  # config-access-exempt: Primus module configs may omit framework
    if framework != _SUPPORTED_FRAMEWORK:
        raise ValueError(
            f"Primus module '{request.module_name}' targets the '{framework}' backend; Miles only "
            f"supports '{_SUPPORTED_FRAMEWORK}' through Primus."
        )

    params = nested_namespace_to_dict(module_config.params)
    if request.overrides:
        params = deep_merge(params, parse_cli_overrides(request.overrides))
    return {key: value for key, value in params.items() if key not in _META_KEYS}


def _split_params(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Partition Primus params into what Miles' Megatron defines and what it does not.

    Primus' own builder does the matching, so the split tracks Megatron's argparse
    rather than a mapping table that would drift from the installed Megatron.
    """
    from primus.backends.megatron.argument_builder import MegatronArgBuilder

    builder = MegatronArgBuilder()
    builder.update(params)
    megatron_params = dict(builder.overrides)
    primus_only = {key: value for key, value in params.items() if key not in megatron_params}
    return megatron_params, primus_only


class PrimusConfigBridge:
    """Applies one Primus experiment YAML to Miles' Megatron arguments."""

    def __init__(self, request: PrimusConfigRequest):
        self.request = request
        self.params = _load_params(request)
        self.report = PrimusConfigReport()

    @classmethod
    def from_argv(cls, argv: list[str] | None = None) -> PrimusConfigBridge | None:
        request = PrimusConfigRequest.from_argv(argv)
        return None if request is None else cls(request)

    def wrap_extra_args_provider(self, provider: ExtraArgsProvider) -> ExtraArgsProvider:
        """Compose with Miles' provider so the YAML lands after every argument is declared."""

        def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
            parser = provider(parser)
            self._apply_defaults(parser)
            return parser

        return add_arguments

    def _apply_defaults(self, parser: argparse.ArgumentParser) -> None:
        megatron_params, primus_only = _split_params(self.params)

        actions = {action.dest: action for action in parser._actions}
        applied = {}
        for key, value in megatron_params.items():
            action = actions.get(key)
            if action is None:
                primus_only[key] = value
                continue
            # A flag cannot hold None: YAML `null` leaves it at its default.
            if value is None and action.nargs == 0:
                continue
            # A YAML-supplied value satisfies the flag, so it must stop being mandatory.
            action.required = False
            applied[key] = value
        parser.set_defaults(**applied)

        self.report.applied = applied
        self.report.carried = primus_only

        logger.info(
            "Primus config %s (module '%s'): %d parameters applied to Megatron arguments, "
            "%d Primus-specific parameters carried through.",
            self.request.config_path,
            self.request.module_name,
            len(applied),
            len(primus_only),
        )
        if primus_only:
            names = sorted(primus_only)
            if self.request.strict:
                raise ValueError(
                    f"{len(names)} Primus parameters are not defined by Miles' Megatron: "
                    f"{', '.join(names)} (--primus-strict-config)"
                )
            preview = ", ".join(names[:_WARN_PREVIEW])
            if len(names) > _WARN_PREVIEW:
                preview += f", ... (+{len(names) - _WARN_PREVIEW} more)"
            logger.warning(
                "%d Primus parameters are not defined by Miles' Megatron and do not affect "
                "training: %s. The full list is on args.primus_params.",
                len(names),
                preview,
            )
            logger.debug("Primus-specific parameters: %s", ", ".join(names))

    def attach(self, args: argparse.Namespace) -> argparse.Namespace:
        """Park the Primus-specific parameters on `args` and route Megatron init through Primus.

        The parameters are deliberately not promoted to individual attributes: the Primus
        config is the only way in, so a consumer reads them from here rather than competing
        with Miles' own flags over a name (`offload` and `mlflow_run_name` are both taken).
        """
        args.primus_params = dict(self.report.carried)

        user_init = args.custom_megatron_init_path
        args.primus_chained_megatron_init_path = None if user_init == PRIMUS_MEGATRON_INIT_PATH else user_init
        args.custom_megatron_init_path = PRIMUS_MEGATRON_INIT_PATH
        self._route_primus_model_family(args)
        return args

    @staticmethod
    def _route_primus_model_family(args: argparse.Namespace) -> None:
        """Build, load and export a Primus-built model family through Primus' own code."""
        from miles.backends.primus_utils.models import (
            PRIMUS_MODEL_PROVIDER_PATH,
            primus_converter_name,
            primus_model_type,
        )

        model_type = primus_model_type(args)
        if model_type is None:
            return
        assert args.megatron_to_hf_mode == "raw", (
            f"Primus model_type '{model_type}' is built by Primus, which Megatron-Bridge cannot map; "
            f"use --megatron-to-hf-mode raw"
        )
        if args.custom_model_provider_path is None:
            args.custom_model_provider_path = PRIMUS_MODEL_PROVIDER_PATH
        if args.model_name is None:
            args.model_name = primus_converter_name(model_type)
        logger.info(
            "Primus model_type '%s': model provider %s, weight converter '%s'.",
            model_type,
            args.custom_model_provider_path,
            args.model_name,
        )
