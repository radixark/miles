from contextlib import contextmanager


_config_registry_applied = False


@contextmanager
def with_transformers_patch():
    apply_transformers_patch()
    yield


def apply_transformers_patch():
    """Register SGLang's custom HF config aliases with transformers.

    SGLang v0.5.12 registers DeepSeek V4 through
    ``sglang.srt.utils.hf_transformers.common``. Importing that module is the
    supported hook; the older private ``_load_deepseek_temp_model`` helper no
    longer exists on the target branch.
    """
    global _config_registry_applied
    if _config_registry_applied:
        return

    import sglang.srt.utils.hf_transformers.common  # noqa: F401

    _config_registry_applied = True


def unapply_transformers_patch():
    # AutoConfig registrations are global and transformers does not expose a
    # reliable unregister path. Keep this function for existing callers.
    return
