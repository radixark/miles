"""HuggingFace config loader with model-type alias registration and overrides.

`load_hf_config` is the single entry point miles uses to load an HF config from a
local checkpoint. It supports 2 customizations:

- Registers model_type aliases before calling AutoConfig, in case the model is
  not recognized in huggingface.
- Accepts an `overrides` dict applied via setattr after loading, so callers can
  adjust fields without touching the checkpoint.

The default behavior is exactly the same as `AutoConfig.from_pretrained`.
"""

import importlib
from dataclasses import dataclass

from transformers import AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

__all__ = ["load_hf_config"]


@dataclass(frozen=True)
class _HFConfigAlias:
    model_type: str
    base_module: str
    base_class: str
    compat_class_name: str
    # Set True to override transformers' native config.
    override_hf_native: bool = False


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
    """Ensure miles' local model_type aliases are registered with AutoConfig."""
    for alias in _CONFIG_ALIASES:
        if alias.model_type in _REGISTERED_ALIASES:
            continue
        if alias.model_type in CONFIG_MAPPING_NAMES and not alias.override_hf_native:
            raise RuntimeError(
                f"transformers now natively supports model_type={alias.model_type!r}; "
                f"set override_hf_native=True to override."
            )
        module = importlib.import_module(alias.base_module)
        base_config = getattr(module, alias.base_class)
        compat_config = type(
            alias.compat_class_name,
            (base_config,),
            {"model_type": alias.model_type, "__module__": __name__},
        )
        AutoConfig.register(alias.model_type, compat_config, exist_ok=alias.override_hf_native)
        _REGISTERED_ALIASES.add(alias.model_type)


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
    """
    _register_hf_config_aliases()
    config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=trust_remote_code, **autoconfig_kwargs)

    if overrides:
        for key, value in overrides.items():
            setattr(config, key, value)
    return config
