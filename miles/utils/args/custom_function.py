import argparse

from miles.utils.arguments import resolve_rollout_function_paths
from miles.utils.environ import use_legacy_rollout_v1
from miles.utils.function_registry import load_function
from miles.utils.workers.argv_utils import with_relax_parser_required_args, with_suppressed_parser_help


def add_user_provided_function_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    try:
        with with_relax_parser_required_args(parser), with_suppressed_parser_help(parser):
            args_partial, _ = parser.parse_known_args()
    except SystemExit:
        return parser
    paths = [args_partial.custom_inference_engine_provider_path]
    if not use_legacy_rollout_v1():
        paths = [
            resolve_rollout_function_paths(args_partial)[0],
            args_partial.custom_generate_function_path,
            *paths,
        ]
    for path in paths:
        try:
            fn = load_function(path)
        except (ModuleNotFoundError, ValueError):
            continue
        if fn is not None and callable(
            getattr(fn, "add_arguments", None)
        ):  # config-access-exempt: custom hooks may optionally register CLI arguments
            fn.add_arguments(parser)
    return parser
