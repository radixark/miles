"""HuggingFace config loader with model-type alias registration and overrides.

`load_hf_config` is the single entry point miles uses to load an HF config from a
local checkpoint. It:

- Registers miles-local model_type aliases (e.g. `deepseek_v32` -> `DeepseekV3Config`)
  before calling AutoConfig, so checkpoints don't need to be mutated.
- Falls back to a plain namespace object built from `config.json` when
  AutoConfig can't parse the file (unknown model_type / non-HF-shaped configs).
- Accepts an `overrides` dict applied via setattr after loading, so callers can
  adjust fields like `max_position_embeddings` without touching the checkpoint.
"""

import importlib
import json
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = ["load_hf_config"]


@dataclass(frozen=True)
class _HFConfigAlias:
    model_type: str
    base_module: str
    base_class: str
    compat_class_name: str


_CONFIG_ALIASES: tuple[_HFConfigAlias, ...] = (
    _HFConfigAlias(
        model_type="deepseek_v32",
        base_module="transformers.models.deepseek_v3.configuration_deepseek_v3",
        base_class="DeepseekV3Config",
        compat_class_name="DeepseekV32Config",
    ),
)

_REGISTERED_ALIASES: set[str] = set()


def _register_hf_config_aliases() -> None:
    """Ensure miles' local model_type aliases are registered with AutoConfig.

    Idempotent: rerunning is a cheap set check. Called automatically by
    load_hf_config; not intended for external use.
    """
    from transformers import AutoConfig

    for alias in _CONFIG_ALIASES:
        if alias.model_type in _REGISTERED_ALIASES:
            continue
        try:
            module = importlib.import_module(alias.base_module)
        except ImportError:
            logger.info(
                "Skip HF config alias %s: base module %s not importable",
                alias.model_type,
                alias.base_module,
            )
            continue
        # base_class missing is a real problem (transformers rename or version skew);
        # let AttributeError surface instead of silently dropping the alias.
        base_config = getattr(module, alias.base_class)
        compat_config = type(
            alias.compat_class_name,
            (base_config,),
            {"model_type": alias.model_type, "__module__": __name__},
        )
        try:
            AutoConfig.register(alias.model_type, compat_config)
        except ValueError as exc:
            msg = str(exc).lower()
            if "already registered" not in msg and "already used" not in msg:
                raise
            # transformers added native support, or another path registered first
        _REGISTERED_ALIASES.add(alias.model_type)


def _fix_dtype_strings(d: dict) -> None:
    import torch

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    for key in ("torch_dtype", "dtype"):
        if key in d and d[key] in dtype_map:
            d[key] = dtype_map[d[key]]


def _namespace_fallback(checkpoint_path: str):
    """Build a namespace object from config.json when AutoConfig can't parse it."""
    config_path = os.path.join(checkpoint_path, "config.json")
    with open(config_path, encoding="utf-8") as f:
        config_dict = json.load(f)
    _fix_dtype_strings(config_dict)
    ns = type("HFConfig", (), config_dict)()
    if "text_config" in config_dict:
        _fix_dtype_strings(config_dict["text_config"])
        ns.text_config = type("TextConfig", (), config_dict["text_config"])()
    return ns


def load_hf_config(
    checkpoint_path: str,
    *,
    overrides: dict | None = None,
    trust_remote_code: bool = True,
    **autoconfig_kwargs,
):
    """Load an HF config from a local checkpoint.

    Parameters
    ----------
    checkpoint_path: path to a local HF checkpoint directory.
    overrides: optional dict of attributes to setattr on the returned config
        after loading. Lets callers patch fields without mutating the checkpoint.
    trust_remote_code: forwarded to AutoConfig.from_pretrained; defaults to True
        to match miles' existing call sites.
    autoconfig_kwargs: extra kwargs forwarded to AutoConfig.from_pretrained.

    Returns
    -------
    The HF Config object, or a namespace fallback if AutoConfig can't parse the
    config.json.
    """
    _register_hf_config_aliases()
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=trust_remote_code, **autoconfig_kwargs)
    except (ValueError, KeyError):
        config = _namespace_fallback(checkpoint_path)

    if overrides:
        for key, value in overrides.items():
            setattr(config, key, value)
    return config
