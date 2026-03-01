"""Dumper lifecycle helpers for the standalone Megatron worker."""

import os

from sglang.srt.debug_utils.dumper import dumper


def finalize_dumper() -> None:
    """Step + disable dumper after forward/backward."""
    if os.environ.get("DUMPER_ENABLE", "0") == "1":
        dumper.step()
        dumper.configure(enable=False)
