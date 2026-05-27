"""HuggingFace config loader with model-type alias registration and overrides.

`load_hf_config` is the single entry point miles uses to load an HF config from a
local checkpoint. It:

- Registers miles-local model_type aliases (e.g. `deepseek_v32` -> `DeepseekV3Config`)
  before calling AutoConfig, so checkpoints don't need to be mutated.
- Accepts an `overrides` dict applied via setattr after loading, so callers can
  adjust fields like `max_position_embeddings` without touching the checkpoint.

Unknown model_types raise the standard transformers error -- new types should
be added to `_CONFIG_ALIASES` rather than silently routed to a fallback.
"""

import importlib
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

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
    """Ensure miles' local model_type aliases are registered with AutoConfig.

    Idempotent: rerunning is a cheap set check. Called automatically by
    load_hf_config; not intended for external use.
    """
    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    for alias in _CONFIG_ALIASES:
        if alias.model_type in _REGISTERED_ALIASES:
            continue
        if alias.model_type in CONFIG_MAPPING_NAMES and not alias.override_hf_native:
            raise RuntimeError(
                f"transformers now natively supports model_type={alias.model_type!r}; "
                f"set override_hf_native=True to override."
            )
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
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=trust_remote_code, **autoconfig_kwargs)

    if overrides:
        for key, value in overrides.items():
            setattr(config, key, value)
    return config
