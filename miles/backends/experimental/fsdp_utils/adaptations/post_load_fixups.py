"""Post-load weight fixups for the FSDP backend (re-assert on-disk values clobbered by from_pretrained)."""

import glob
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class PostLoadFixup:
    name: str
    applies_to: Callable  # (hf_config) -> bool
    apply: Callable  # (model, ckpt_path) -> int (count of params re-asserted)


_FIXUPS: list[PostLoadFixup] = []


def register_post_load_fixup(fixup: PostLoadFixup) -> None:
    _FIXUPS.append(fixup)


def apply_post_load_fixups(model, hf_config, ckpt_path) -> list[str]:
    """Run every registered fixup whose arch-predicate matches; return the names that fired."""
    fired = []
    for f in _FIXUPS:
        if f.applies_to(hf_config) and f.apply(model, ckpt_path):
            fired.append(f.name)
    return fired
