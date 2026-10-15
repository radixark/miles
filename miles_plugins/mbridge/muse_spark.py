from mbridge.core import register_model

from miles_plugins.models.muse_spark.checkpoint import load_hf_checkpoint
from miles_plugins.models.muse_spark.config import parse_muse_spark_1x_config
from miles_plugins.models.muse_spark.hf_config import register_muse_spark_config

register_muse_spark_config()


@register_model(["muse_spark"])
class MuseSpark1xBridge:
    """Offline HF initialization; Miles constructs and trains this model in raw mode."""

    def __init__(self, hf_config):
        self.hf_config = hf_config
        source = hf_config.to_dict()
        for name, default in (("is_encoder_decoder", False), ("chunk_size_feed_forward", 0)):
            if source.pop(name, default) != default:
                raise ValueError(f"unsupported Spark HF option: {name}")
        # HF adds output metadata and derives SWA flags from global_attn_cfg.
        # Validate the original model fields rather than these duplicate views.
        for name in (
            "hybrid_layer_pattern",
            "is_hybrid_swa",
            "id2label",
            "label2id",
            "output_attentions",
            "output_hidden_states",
            "problem_type",
            "return_dict",
        ):
            source.pop(name, None)
        parse_muse_spark_1x_config(source)

    def load_weights(self, models, weights_path, memory_efficient=False):
        """Always read local slices, preserving FP32 routers and expert-bias buffers."""
        load_hf_checkpoint(models, weights_path)
