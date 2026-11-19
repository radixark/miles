from __future__ import annotations

import argparse
from typing import Any

try:
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import get_device
except ImportError:
    ServerArgs = None  # type: ignore[assignment,misc]
    get_device = None  # type: ignore[assignment]

from miles.utils.workers.argv_utils import render_cli_argv

_ALWAYS_RENDER_FIELDS = ("trust_remote_code", "model_path", "host", "port", "device")


def _require_server_args() -> Any:
    if ServerArgs is None:
        raise ImportError("sglang is required to serialize or parse SGLang ServerArgs.")
    return ServerArgs


def server_args_to_argv(server_args_dict: dict) -> list[str]:
    server_args_cls = _require_server_args()
    server_args_dict = {**server_args_dict, "device": _explicit_device(server_args_dict.get("device"))}
    return render_cli_argv(
        server_args_dict,
        expected_obj=server_args_cls(**server_args_dict),
        make_parser=_make_cli_parser,
        from_parsed=server_args_cls.from_cli_args,
        always_render_fields=_ALWAYS_RENDER_FIELDS,
    )


def parse_server_args_argv(argv: list[str]) -> Any:
    server_args_cls = _require_server_args()
    return server_args_cls.from_cli_args(_make_cli_parser().parse_args(argv))


def _make_cli_parser() -> argparse.ArgumentParser:
    server_args_cls = _require_server_args()
    parser = argparse.ArgumentParser()
    server_args_cls.add_cli_args(parser)
    return parser


def _explicit_device(device: str | None) -> str:
    if device:
        return device.split(":")[0]
    if get_device is None:
        raise ImportError("sglang is required to infer the SGLang device when device is unspecified.")
    return get_device().split(":")[0]
