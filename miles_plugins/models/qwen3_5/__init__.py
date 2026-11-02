"""Qwen3.5 on the plugin path: ``model`` (text spec, GDN, providers), ``vision`` and ``mm_towers``."""


def get_qwen3_5_spec(*args, **kwargs):
    # Imported on call, so importing the vision tower does not load Transformer Engine with the text model.
    from miles_plugins.models.qwen3_5.model import get_qwen3_5_spec as get_spec

    return get_spec(*args, **kwargs)
