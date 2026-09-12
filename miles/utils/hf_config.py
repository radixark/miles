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
from pathlib import Path

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
    # Name of a module-level function applied to the config kwargs before the base __init__.
    normalize_kwargs: str | None = None


_DEEPSEEK_V41_LEGACY_FIELDS = {
    "kv_source_layers": "kv_source_layer_ids",
    "index_source_layers": "index_source_layer_ids",
    "candidate_source_layer": "candidate_source_layer_id",
    "engram_pad_id": "engram_pad_token_id",
    "dspark_n_activated_experts": "dspark_num_experts_per_tok",
}


def _normalize_deepseek_v41_kwargs(kwargs: dict) -> dict:
    """Flatten the composite HF config (text_config holds the decoder) and accept the legacy key names."""
    kwargs = dict(kwargs)
    text = kwargs.pop("text_config", None)
    kwargs.pop("vision_config", None)
    if isinstance(text, dict):
        text = dict(text)
        text.pop("model_type", None)
        kwargs = {**text, **kwargs}
    for old, new in _DEEPSEEK_V41_LEGACY_FIELDS.items():
        if old in kwargs:
            kwargs.setdefault(new, kwargs.pop(old))
    return kwargs


_CONFIG_ALIASES: tuple[_HFConfigAlias, ...] = (
    _HFConfigAlias(
        model_type="deepseek_v32",
        base_module="transformers.models.deepseek_v3.configuration_deepseek_v3",
        base_class="DeepseekV3Config",
        compat_class_name="DeepseekV32Config",
        override_hf_native=True,
    ),
    _HFConfigAlias(
        model_type="deepseek_v4",
        base_module="transformers.models.deepseek_v3.configuration_deepseek_v3",
        base_class="DeepseekV3Config",
        compat_class_name="DeepseekV4Config",
        auto_model_classes=(),
        override_hf_native=True,
    ),
    _HFConfigAlias(
        model_type="deepseek_v41",
        base_module="transformers.models.deepseek_v3.configuration_deepseek_v3",
        base_class="DeepseekV3Config",
        compat_class_name="DeepseekV41Config",
        auto_model_classes=(),
        override_hf_native=True,
        normalize_kwargs="_normalize_deepseek_v41_kwargs",
    ),
    _HFConfigAlias(
        model_type="deepseek_v4.1",
        base_module="transformers.models.deepseek_v3.configuration_deepseek_v3",
        base_class="DeepseekV3Config",
        compat_class_name="LegacyDeepseekV41Config",
        auto_model_classes=(),
        override_hf_native=True,
        normalize_kwargs="_normalize_deepseek_v41_kwargs",
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
        attrs = {"model_type": alias.model_type, "__module__": __name__}
        if alias.normalize_kwargs is not None:
            normalize = globals()[alias.normalize_kwargs]
            model_type = alias.model_type

            def __init__(self, *args, _normalize=normalize, _model_type=model_type, _base=base_config, **kwargs):
                kwargs = _normalize(kwargs)
                kwargs["model_type"] = _model_type
                _base.__init__(self, *args, **kwargs)

            attrs["__init__"] = __init__
        compat_config = type(alias.compat_class_name, (base_config,), attrs)
        AutoConfig.register(alias.model_type, compat_config, exist_ok=alias.override_hf_native)
        for auto_cls in alias.auto_model_classes:
            base_model_cls = auto_cls._model_mapping[base_config]
            compat_model_cls = type(
                base_model_cls.__name__, (base_model_cls,), {"config_class": compat_config, "__module__": __name__}
            )
            auto_cls.register(compat_config, compat_model_cls, exist_ok=alias.override_hf_native)
        _REGISTERED_ALIASES.add(alias.model_type)

    try:
        import sglang.srt.configs.inkling  # noqa: F401
    except ImportError:
        pass


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


def is_dsa(hf_config) -> bool:
    return getattr(hf_config, "model_type", None) in ("deepseek_v32", "glm_moe_dsa")


# Written by HF exports after all ranks finish, so consumers can tell finished from partial.
HF_EXPORT_COMPLETE_MARKER = ".complete"


def is_complete_hf_export(path: str | Path) -> bool:
    return (Path(path) / HF_EXPORT_COMPLETE_MARKER).exists()
