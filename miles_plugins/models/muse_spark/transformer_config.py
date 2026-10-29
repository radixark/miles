"""The Megatron TransformerConfig of Muse Spark, apart from ``config`` so that reading a
checkpoint's config (``hf_config``, imported by every miles process) does not import Megatron
Core, whose Transformer Engine needs the CUDA driver library at import time."""

from dataclasses import dataclass, field

from megatron.core.transformer.transformer_config import TransformerConfig

from miles_plugins.models.muse_spark.config import MuseSpark1xConfig


@dataclass
class MuseSpark1xTransformerConfig(TransformerConfig):
    spark: MuseSpark1xConfig = field(default_factory=MuseSpark1xConfig)
