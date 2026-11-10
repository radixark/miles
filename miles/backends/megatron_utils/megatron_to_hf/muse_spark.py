"""Convert already gathered MCore tensors to Spark's streaming HF weight names."""

import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=8)
def _checkpoint_layout(hf_checkpoint):
    # Other raw converters do not require Spark's MCore configuration dependency.
    from miles_plugins.models.muse_spark.config import parse_muse_spark_1x_config
    from miles_plugins.models.muse_spark.weight_mapping import build_mappings, hf_tensor_specs

    spark = parse_muse_spark_1x_config(json.loads((Path(hf_checkpoint) / "config.json").read_text()))
    return {mapping.megatron_param: mapping for mapping in build_mappings(spark)}, hf_tensor_specs(spark)


def convert_muse_spark_1x_to_hf(args, name, param):
    """One MCore tensor -> ``[(checkpoint_name, tensor), ...]``.

    Experts leave one at a time as ``mlp.experts.<E>.{gate_proj,up_proj,down_proj}.weight``; the engine writes
    them into its stacked ``gate_up_proj`` / ``down_proj`` parameters. q/k rows are re-ordered to the checkpoint's
    rotary-pair convention, so the engine's loader (which re-interleaves) receives exactly what it expects.
    """
    mappings, specs = _checkpoint_layout(args.hf_checkpoint)
    local_name = name.removeprefix("module.module.")
    if local_name not in mappings:
        raise ValueError(f"Unknown Spark parameter name: {name}")
    mapping = mappings[local_name]
    converted = mapping.megatron_to_hf(param)
    for hf_name, tensor in converted.items():
        spec = specs[hf_name]
        shape = spec.shape[1:] if mapping.expert_index is not None else spec.shape
        if tuple(tensor.shape) != shape or tensor.dtype != spec.dtype:
            raise ValueError(f"Spark raw tensor shape/dtype differs from checkpoint: {name}")
    if mapping.expert_index is None:
        return list(converted.items())

    hf_name, tensor = next(iter(converted.items()))
    prefix, source = hf_name.rsplit(".", 1)
    expert_prefix = f"{prefix}.{mapping.expert_index}"
    if source == "gate_up_proj":
        gate, up = tensor.chunk(2, dim=0)
        return [(f"{expert_prefix}.gate_proj.weight", gate), (f"{expert_prefix}.up_proj.weight", up)]
    return [(f"{expert_prefix}.down_proj.weight", tensor)]
