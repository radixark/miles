from mbridge.core import register_model
from mbridge.models import DeepseekV3Bridge


@register_model("joyai_llm_flash")
class JoyAILLMFlashBridge(DeepseekV3Bridge):
    """Minimal JoyAI alias bridge with MTP explicitly disabled.

    JoyAI-LLM-Flash exposes `model_type=joyai_llm_flash` while architecture is
    DeepSeek-V3 compatible. We only register the model type alias so
    AutoBridge can instantiate. MTP conversion is intentionally disabled for now.
    """

    def _build_config(self):
        config = super()._build_config()

        # Keep JoyAI conversion on the non-MTP path for now.
        if hasattr(config, "mtp_num_layers"):
            config.mtp_num_layers = None
        elif isinstance(config, dict):
            config["mtp_num_layers"] = None

        return config
