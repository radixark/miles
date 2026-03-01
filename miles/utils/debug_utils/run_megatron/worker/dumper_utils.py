"""Dumper configuration for the standalone Megatron worker (env-var based)."""

import argparse
import os


def setup_dumper(args: argparse.Namespace) -> None:
    """Configure dumper from environment variables (set by cli.py)."""
    from sglang.srt.debug_utils.dumper import dumper

    dumper_dir: str | None = os.environ.get("DUMPER_DIR")
    if not dumper_dir:
        return

    dumper_enable: bool = os.environ.get("DUMPER_ENABLE", "0") == "1"
    if not dumper_enable:
        return

    dumper_filter: str = os.environ.get("DUMPER_FILTER", "")
    dump_grad: bool = os.environ.get("DUMPER_DUMP_GRAD", "0") == "1"

    dumper.configure(
        enable=True,
        dir=dumper_dir,
        exp_name=os.environ.get("DUMPER_EXP_NAME", "standalone"),
        filter=dumper_filter if dumper_filter else None,
        enable_model_grad=dump_grad,
    )


def finalize_dumper() -> None:
    """Step + disable dumper after forward/backward."""
    from sglang.srt.debug_utils.dumper import dumper

    if os.environ.get("DUMPER_ENABLE", "0") == "1":
        dumper.step()
        dumper.configure(enable=False)
