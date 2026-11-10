from mbridge.core import register_model

from miles_plugins.models.muse_spark.checkpoint import load_hf_checkpoint


@register_model(["muse_spark"])
class MuseSpark1xBridge:
    """Offline HF initialization; Miles constructs and trains this model in raw mode."""

    def __init__(self, hf_config, make_vocab_size_divisible_by=None):
        # create_mbridge_for_model passes the model's vocab rows; the slice loader reads the checkpoint unpadded.
        self.hf_config = hf_config

    def load_weights(self, models, weights_path, memory_efficient=False):
        """Always read local slices, preserving FP32 routers and expert-bias buffers; the loader validates the
        checkpoint's config.json against the model."""
        load_hf_checkpoint(models, weights_path)
