def get_step5_spec(args, config, vp_stage):
    # defer Transformer Engine imports for CPU-only layout and conversion callers
    from miles_plugins.models.step5.model import get_step5_spec as build_spec

    return build_spec(args, config, vp_stage)


__all__ = ["get_step5_spec"]
