"""Last-four-layer Muse Spark text partial, extracted by scripts/muse_spark/extract_partial_checkpoint.py."""

from model_args_utils import load_sibling_model_args


def model_args() -> str:
    return load_sibling_model_args(__file__, "muse-spark-full", num_layers=4)
