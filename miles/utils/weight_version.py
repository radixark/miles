import re
from argparse import Namespace
from typing import TYPE_CHECKING

from miles.utils.lora import is_lora_enabled

if TYPE_CHECKING:
    from miles.utils.types import Sample

SGLANG_DEFAULT_WEIGHT_VERSION = "default"

_NUMERIC_VERSION_PATTERN = re.compile(r"[0-9]+")


def assert_samples_weight_version_sane(args: Namespace, samples: list["Sample"]) -> None:
    if args.debug_rollout_only or args.debug_skip_weight_update or is_lora_enabled(args):
        return

    for sample in samples:
        for span in sample.all_weight_version_spans:
            assert span.version != SGLANG_DEFAULT_WEIGHT_VERSION, (
                f"sample index={sample.index} tokens [{span.abs_start}, {span.abs_end}) were generated under "
                f"weight version {SGLANG_DEFAULT_WEIGHT_VERSION!r}, the sglang placeholder for an engine whose "
                f"weights were never updated; training data must only come from engines that received a weight update"
            )
            assert _NUMERIC_VERSION_PATTERN.fullmatch(span.version), (
                f"sample index={sample.index} tokens [{span.abs_start}, {span.abs_end}) carry weight version "
                f"{span.version!r}, which is not the numeric version miles stamps on weight updates; "
                f"the engine serving this sample got its weights from somewhere miles does not know about"
            )
