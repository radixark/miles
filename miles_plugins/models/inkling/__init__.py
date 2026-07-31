def get_inkling_spec(*args, **kwargs):
    from .model import get_inkling_spec as get_spec

    return get_spec(*args, **kwargs)


def inkling_model_provider(*args, **kwargs):
    from .model import inkling_model_provider as provider

    return provider(*args, **kwargs)


__all__ = ["get_inkling_spec", "inkling_model_provider"]
