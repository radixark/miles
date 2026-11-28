"""Language-model-only bridge for Nemotron-H Omni checkpoints.

Nemotron 3.5 Super VL nests the Nemotron 3 Super language model under
``llm_config`` and prefixes its tensors with ``language_model.``. This plugin
admits the released ``NemotronH_Omni_Reasoning_V3`` architecture to AutoBridge
and reuses the Nemotron-H bridge for the language model. The vision tower and
projector have no Megatron counterpart and are not trained.
"""

from __future__ import annotations

import copy
import logging

logger = logging.getLogger(__name__)

OMNI_ARCHITECTURE = "NemotronH_Omni_Reasoning_V3"
LANGUAGE_MODEL_PREFIX = "language_model."

_LAYER_SYMBOLS = {"mamba": "M", "attention": "*", "moe": "E", "mlp": "-"}


def layer_pattern(block_types) -> str | None:
    """Encode an HF ``layers_block_type`` list as a Megatron hybrid layer pattern."""
    if not block_types:
        return None
    return "".join(_LAYER_SYMBOLS[block_type] for block_type in block_types)


class LanguageModelConfig:
    """The nested Nemotron-H config plus the hybrid-pattern fields Bridge reads."""

    def __init__(self, llm_config):
        self._llm_config = llm_config

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._llm_config, name)  # config-access-exempt: the wrapper forwards HF config fields by name

    @property
    def hybrid_override_pattern(self):
        pattern = getattr(
            self._llm_config, "hybrid_override_pattern", None
        )  # config-access-exempt: transformers versions either expose or derive the hybrid pattern
        return pattern or layer_pattern(self._llm_config.layers_block_type)

    @property
    def mtp_hybrid_override_pattern(self):
        pattern = getattr(
            self._llm_config, "mtp_hybrid_override_pattern", None
        )  # config-access-exempt: transformers versions either expose or derive the MTP pattern
        return pattern or layer_pattern(
            getattr(self._llm_config, "mtp_layers_block_type", None)
        )  # config-access-exempt: HF configs without MTP omit its block-type list


def language_model_config(hf_config):
    """Return the language-model view of an Omni config; other configs pass through."""
    llm_config = getattr(hf_config, "llm_config", None)  # config-access-exempt: only Omni configs nest llm_config
    if llm_config is None:
        return hf_config
    return LanguageModelConfig(llm_config)


class LanguageModelPretrained:
    """``hf_pretrained`` whose config is the language-model config; weights and artifacts are shared."""

    def __init__(self, hf_pretrained):
        self._hf_pretrained = hf_pretrained

    @property
    def config(self):
        return language_model_config(self._hf_pretrained.config)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(
            self._hf_pretrained, name
        )  # config-access-exempt: the wrapper forwards checkpoint object fields by name


def _prefixed(mapping, prefix: str):
    copied = copy.copy(mapping)
    if isinstance(mapping.hf_param, str):
        copied.hf_param = prefix + mapping.hf_param
    else:
        copied.hf_param = {key: prefix + value for key, value in mapping.hf_param.items()}
    return copied


def _registry_mappings(registry):
    if hasattr(
        registry, "mappings"
    ):  # config-access-exempt: bridge versions expose registry mappings under different names
        return registry.mappings
    return registry._mappings


def _allow_omni_architecture() -> None:
    from megatron.bridge.models.conversion import auto_bridge

    if OMNI_ARCHITECTURE in auto_bridge.SUPPORTED_HF_ARCHITECTURES:
        return
    auto_bridge.SUPPORTED_HF_ARCHITECTURES = (*auto_bridge.SUPPORTED_HF_ARCHITECTURES, OMNI_ARCHITECTURE)
    auto_bridge.SUPPORTED_HF_ARCHITECTURES_DISPLAY = " or ".join(
        f"'{name}'" for name in auto_bridge.SUPPORTED_HF_ARCHITECTURES
    )


def _build_bridge_subclass():
    from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
    from megatron.core.models.mamba import MambaModel

    from miles_plugins.megatron_bridge.nemotron_h import bridge_class

    _allow_omni_architecture()

    @MegatronModelBridge.register_bridge(source=OMNI_ARCHITECTURE, target=MambaModel)
    class MilesNemotronHOmniBridge(bridge_class()):
        """Nemotron-H Omni bridge that trains only the nested language model."""

        def provider_bridge(self, hf_pretrained):
            return super().provider_bridge(LanguageModelPretrained(hf_pretrained))

        def mapping_registry(self):
            omni_config = self.hf_config
            self.hf_config = language_model_config(omni_config)
            try:
                registry = super().mapping_registry()
            finally:
                self.hf_config = omni_config
            return MegatronMappingRegistry(
                *(_prefixed(mapping, LANGUAGE_MODEL_PREFIX) for mapping in _registry_mappings(registry))
            )

    return MilesNemotronHOmniBridge


_bridge_class = None


def bridge_class():
    """Return the registered :class:`MilesNemotronHOmniBridge`, building it on first use."""
    global _bridge_class
    if _bridge_class is None:
        _bridge_class = _build_bridge_subclass()
    return _bridge_class


def install() -> None:
    """Register the Nemotron-H Omni bridge subclass."""
    try:
        bridge_class()
    except Exception as e:  # best-effort; avoid breaking unrelated models
        logger.warning("miles nemotron_h_omni bridge not registered: %s", e)


install()
