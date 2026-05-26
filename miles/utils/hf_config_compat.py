import importlib
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_REGISTERED_CONFIG_ALIASES: set[str] = set()


@dataclass(frozen=True)
class HFConfigAlias:
    model_type: str
    base_module: str
    base_class: str
    compat_class_name: str


_CONFIG_ALIASES = (
    HFConfigAlias(
        model_type="deepseek_v32",
        base_module="transformers.models.deepseek_v3.configuration_deepseek_v3",
        base_class="DeepseekV3Config",
        compat_class_name="DeepseekV32Config",
    ),
)


def register_hf_config_compat() -> None:
    """Register local HF config aliases for model types missing in Transformers."""
    try:
        from transformers import AutoConfig
    except ImportError:
        return

    for alias in _CONFIG_ALIASES:
        if alias.model_type in _REGISTERED_CONFIG_ALIASES:
            continue

        try:
            module = importlib.import_module(alias.base_module)
            base_config = getattr(module, alias.base_class)
        except (AttributeError, ImportError):
            continue

        compat_config = type(
            alias.compat_class_name,
            (base_config,),
            {"model_type": alias.model_type, "__module__": __name__},
        )

        try:
            AutoConfig.register(alias.model_type, compat_config)
        except ValueError as exc:
            message = str(exc).lower()
            if "already registered" in message or "already used" in message:
                _REGISTERED_CONFIG_ALIASES.add(alias.model_type)
            else:
                logger.warning("Failed to register %s HF config alias: %s", alias.model_type, exc)
        else:
            _REGISTERED_CONFIG_ALIASES.add(alias.model_type)
