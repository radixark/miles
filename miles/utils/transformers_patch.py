import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

_original_from_pretrained = None


@contextmanager
def with_transformers_patch():
    apply_transformers_patch()
    try:
        yield
    finally:
        unapply_transformers_patch()


def apply_transformers_patch():
    global _original_from_pretrained
    if _original_from_pretrained is not None:
        return

    from transformers.models.auto.configuration_auto import AutoConfig

    _original_from_pretrained = AutoConfig.from_pretrained

    @classmethod
    def _patched_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers.configuration_utils import PretrainedConfig

        config_dict, _ = PretrainedConfig.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )
        _patch_deepseek_config_dict(config_dict)

        return _original_from_pretrained.__func__(
            cls, pretrained_model_name_or_path, **kwargs
        )

    AutoConfig.from_pretrained = _patched_from_pretrained


def unapply_transformers_patch():
    global _original_from_pretrained
    if _original_from_pretrained is None:
        return

    from transformers.models.auto.configuration_auto import AutoConfig

    AutoConfig.from_pretrained = _original_from_pretrained
    _original_from_pretrained = None


def _patch_deepseek_config_dict(config_dict: dict):
    model_type = config_dict.get("model_type")

    if model_type == "deepseek_v32":
        logger.info(
            "Detected deepseek_v32 model, treating as deepseek_v3 for compatibility."
        )
        config_dict["model_type"] = "deepseek_v3"
        if "architectures" in config_dict:
            config_dict["architectures"] = [
                arch.replace("DeepseekV32", "DeepseekV3")
                for arch in config_dict["architectures"]
            ]
    elif model_type in ("deepseek_v4", "deepseek_ref"):
        logger.info(
            "Detected deepseek_v4 model, treating as deepseek_v3 for compatibility."
        )
        config_dict["model_type"] = "deepseek_v3"
        if "architectures" in config_dict:
            config_dict["architectures"] = [
                arch.replace("DeepseekV4", "DeepseekV3")
                for arch in config_dict["architectures"]
            ]
