"""Weight-sync hooks a model plugin registers (TOP DeepSeek-V4) instead of Miles importing it.

``converter(args, model_name, name, param, quantization_config, packed_weight_basenames)`` returns the HF tensors
for a parameter the plugin owns, else None. ``checkpoint_units(args, model_name, quantization_config, *,
materialize, layers)`` yields serving state the trainer holds no parameters for; ``layers`` (0-based global decoder
indices) limits it to one pipeline stage, None means every layer. ``frozen_indexer`` marks the DS4 indexer as not
trained, so its compressor pair is never exported as a parameter.
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class ExportHooks:
    converter: Callable | None = None
    checkpoint_units: Callable | None = None
    frozen_indexer: bool = False


_HOOKS = ExportHooks()


def register_export_hooks(**hooks) -> None:
    for name, value in hooks.items():
        if not hasattr(_HOOKS, name):  # config-access-exempt: attribute selected at runtime from a table of names
            raise TypeError(f"unknown export hook {name!r}")
        setattr(_HOOKS, name, value)


def export_hooks() -> ExportHooks:
    return _HOOKS
