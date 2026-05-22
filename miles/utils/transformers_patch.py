from contextlib import contextmanager


_config_registry_applied = False
_DEEPSEEK_V3_ALIASES = ("deepseek_v32", "deepseek_v4", "kimi_k2")


@contextmanager
def with_transformers_patch():
    apply_transformers_patch()
    yield


def apply_transformers_patch():
    """Register the DeepSeek V3-family HF config aliases used by SGLang.

    SGLang v0.5.12 registers these model types as thin aliases of
    ``transformers.DeepseekV3Config``. Mirroring that small registry locally
    avoids importing the whole SGLang package just to load Miles-side configs.
    The older private ``_load_deepseek_temp_model`` helper no longer exists on
    the target SGLang branch.
    """
    global _config_registry_applied
    if _config_registry_applied:
        return

    from transformers import DeepseekV3Config
    from transformers.models.auto.configuration_auto import AutoConfig

    for model_type in _DEEPSEEK_V3_ALIASES:
        config_cls = type(
            f"_{model_type.title().replace('_', '')}ConfigAlias",
            (DeepseekV3Config,),
            {"model_type": model_type},
        )
        try:
            AutoConfig.register(model_type, config_cls)
        except ValueError as e:
            err = str(e).lower()
            if "already registered" not in err and "already used" not in err:
                raise

    _config_registry_applied = True


def unapply_transformers_patch():
    # AutoConfig registrations are global and transformers does not expose a
    # reliable unregister path. Keep this function for existing callers.
    return
