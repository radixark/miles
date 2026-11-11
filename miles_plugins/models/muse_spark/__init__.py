"""Muse Spark plugins for Miles' raw Megatron backend."""


def muse_spark_1x_model_provider(pre_process=True, post_process=True, vp_stage=None, *, args):
    # Importing this package's other modules must not import the model code or Transformer Engine.
    from miles_plugins.models.muse_spark.provider import muse_spark_1x_model_provider as provider

    return provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage, args=args)
