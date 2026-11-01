"""Muse Spark Open's config for transformers releases that have no class for it (5.19 has none).

The checkpoint declares ``model_type: muse_spark`` with the decoder under ``text_config`` and the vision and
audio towers under their own sections. Only the text section needs attribute access (``hf_text_config``,
the flops table, the torchtitan model builder); the tower sections stay plain mappings.
"""

from transformers import AutoConfig, PretrainedConfig


class MuseSparkTextConfig(PretrainedConfig):
    model_type = "muse_spark_text"


class MuseSparkConfig(PretrainedConfig):
    model_type = "muse_spark"
    sub_configs = {"text_config": MuseSparkTextConfig}

    def __init__(self, text_config=None, **kwargs):
        if isinstance(text_config, dict):
            text_config = MuseSparkTextConfig(**text_config)
        self.text_config = text_config if text_config is not None else MuseSparkTextConfig()
        super().__init__(**kwargs)


def register_muse_spark_configs() -> None:
    # exist_ok: an sglang build that serves Muse Spark registers the same model types in the same process
    for config in (MuseSparkTextConfig, MuseSparkConfig):
        AutoConfig.register(config.model_type, config, exist_ok=True)
