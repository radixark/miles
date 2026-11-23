"""A model vendor's own reference code as top_vjp's reference, for families HF transformers lacks or models poorly.

A vendor's inference file imports its kernels as sibling modules (DeepSeek: ``from kernel import ...``); the loader
serves those imports from PyTorch stand-ins, so the vendor's own module code runs in FP32 (or BF16) under autograd.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType, SimpleNamespace

VENDOR_DIR = Path(__file__).parent / "vendor"


def load_vendor_module(path: Path, *, name: str, shims: Mapping[str, ModuleType]) -> ModuleType:
    """Import ``path`` as module ``name``; while it executes, each import of a ``shims`` key returns its stand-in."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    saved = {key: sys.modules.get(key) for key in shims}
    sys.modules.update(shims)
    try:
        spec.loader.exec_module(module)
    finally:
        for key, previous in saved.items():
            if previous is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = previous
    return module


def read_config_json(model_path: str) -> SimpleNamespace:
    """A checkpoint's config.json as attributes, for model types transformers' AutoConfig does not know."""
    with open(Path(model_path) / "config.json") as f:
        return json.load(f, object_hook=lambda fields: SimpleNamespace(**fields))
