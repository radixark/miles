"""Minimal transformers config classes for model_type=glm5_next.

transformers has no native glm5_next; serving parses it with sglang's own
registry. Training only needs attribute access to the raw json fields, so a
thin PretrainedConfig subclass per level is enough (nested dicts like
linear_attn_config stay plain dicts)."""

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig


class Glm5NextTextConfig(PretrainedConfig):
    model_type = "glm5_next_text"


class Glm5NextConfig(PretrainedConfig):
    model_type = "glm5_next"

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        if isinstance(text_config, dict):
            text_config = Glm5NextTextConfig(**text_config)
        if text_config is not None:
            self.text_config = text_config
        if vision_config is not None:
            self.vision_config = (
                PretrainedConfig(**vision_config) if isinstance(vision_config, dict) else vision_config
            )
        super().__init__(**kwargs)


def register_glm5_next_config() -> None:
    try:
        AutoConfig.register("glm5_next", Glm5NextConfig)
    except ValueError:
        pass
