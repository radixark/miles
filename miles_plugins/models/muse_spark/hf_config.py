"""Read Spark checkpoint metadata without loading its model implementation.

The checkpoint config nests the text model under ``text_config``; generic Miles code (``hf_validate_args``:
``hidden_size``, ``num_hidden_layers``, ``rope_theta``, ...) reads a flat attribute view, so the text fields are
lifted to the top level. ``to_dict`` returns the checkpoint's own config untouched, which is what the plugin's
parser and bridge consume.
"""

import copy

from transformers import AutoConfig, PretrainedConfig

from miles_plugins.models.muse_spark.config import ARCHITECTURE, MODEL_TYPE, text_fields_from_checkpoint_config

_NESTED_SECTIONS = ("text_config", "vision_config", "audio_config")


class MuseSparkHFConfig(PretrainedConfig):
    model_type = MODEL_TYPE

    def __init__(self, **kwargs):
        raw = copy.deepcopy(kwargs)
        for name in ("_commit_hash", "name_or_path", "attn_implementation"):
            raw.pop(name, None)
        for name in ("dtype", "torch_dtype"):
            if raw.get(name) is not None:
                raw[name] = str(raw[name]).removeprefix("torch.")
        raw.setdefault("model_type", MODEL_TYPE)
        raw.setdefault("architectures", [ARCHITECTURE])
        init_kwargs = {name: value for name, value in kwargs.items() if name not in _NESTED_SECTIONS}
        # transformers also instantiates the class without a checkpoint (defaults for ``to_diff_dict``).
        if "text_config" in raw:
            flat = text_fields_from_checkpoint_config(raw)
            init_kwargs.update(
                {name: value for name, value in flat.items() if name not in ("architectures", "model_type")}
            )
        # PretrainedConfig.__init__ may already serialize the config (to_dict), so the raw view exists first.
        self._spark_raw_config = raw
        super().__init__(**init_kwargs)
        self.architectures = raw["architectures"]
        self._spark_raw_config = raw

    def to_dict(self):
        raw = self.__dict__.get("_spark_raw_config")
        return copy.deepcopy(raw) if raw is not None else super().to_dict()


def register_muse_spark_config():
    AutoConfig.register(MuseSparkHFConfig.model_type, MuseSparkHFConfig, exist_ok=True)
