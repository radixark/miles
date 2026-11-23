"""Qwen3.5 on the plugin path: ``model`` (text spec, GDN, providers), ``vision`` and ``mm_towers``."""


def get_qwen3_5_spec(*args, **kwargs):
    # Imported on call, so importing the vision tower does not load Transformer Engine with the text model.
    from miles_plugins.models.qwen3_5.model import get_qwen3_5_spec as get_spec

    return get_spec(*args, **kwargs)


def qwen3_5_mm_model_provider(
    pre_process: bool = True, post_process: bool = True, vp_stage: int | None = None, *, args
):
    # The full signature: the provider loader passes args and vp_stage only when it sees them.
    from miles_plugins.models.qwen3_5.model import qwen3_5_mm_model_provider as provider

    return provider(pre_process, post_process, vp_stage, args=args)


__all__ = ["get_qwen3_5_spec", "qwen3_5_mm_model_provider"]
