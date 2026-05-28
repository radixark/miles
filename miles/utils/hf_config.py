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

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES


@dataclass(frozen=True)
class _HFConfigAlias:
    model_type: str
    base_module: str
    base_class: str
    compat_class_name: str
    auto_model_classes: tuple = (AutoModelForCausalLM,)
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


def register_hf_config_aliases() -> None:
    """Register miles model_type aliases with transformers. Idempotent.

    Already called inside `load_hf_config` and `load_tokenizer`. Only call
    directly before a third-party entry point that won't go through either
    (e.g. megatron's `_build_tokenizer`).
    """
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
        for auto_cls in alias.auto_model_classes:
            base_model_cls = auto_cls._model_mapping[base_config]
            compat_model_cls = type(
                base_model_cls.__name__, (base_model_cls,), {"config_class": compat_config, "__module__": __name__}
            )
            auto_cls.register(compat_config, compat_model_cls, exist_ok=alias.override_hf_native)
        _REGISTERED_ALIASES.add(alias.model_type)


def load_hf_config(
    checkpoint_path: str,
    *,
    overrides: dict | None = None,
    trust_remote_code: bool = True,
    **autoconfig_kwargs,
):
    """Load an HF config from a local checkpoint.

    Registers model aliases first for pre-set aliases.

    overrides: optional dict of attributes to setattr on the returned config
        after loading. Lets callers patch fields without mutating the checkpoint.
    """
    register_hf_config_aliases()
    config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=trust_remote_code, **autoconfig_kwargs)

    if overrides:
        for key, value in overrides.items():
            setattr(config, key, value)
    return config
