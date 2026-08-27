from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig


class Glm5NextTextConfig(PretrainedConfig):
    model_type = "glm5_next_text"


class Glm5NextVisionConfig(PretrainedConfig):
    model_type = "glm5_next_vision"


class Glm5NextConfig(PretrainedConfig):
    model_type = "glm5_next"
    sub_configs = {"text_config": Glm5NextTextConfig, "vision_config": Glm5NextVisionConfig}

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        if isinstance(text_config, dict):
            text_config = Glm5NextTextConfig(**text_config)
        if isinstance(vision_config, dict):
            vision_config = Glm5NextVisionConfig(**vision_config)
        self.text_config = text_config
        self.vision_config = vision_config
        super().__init__(**kwargs)


def register_glm5_next_config() -> None:
    AutoConfig.register("glm5_next", Glm5NextConfig, exist_ok=True)
