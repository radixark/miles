"""Convert a Muse Spark torch_dist checkpoint (a ``--save`` iteration) into the released HF checkpoint layout, on CPU.

The output keeps every shard name, header and size of the HF checkpoint it was trained from, so it can replace that
checkpoint for serving with SGLang or for further training; unchanged weights reproduce it byte for byte. The vision
and audio towers are never trained and are copied from it. Each worker process owns a subset of the shards and reads
only the torch_dist tensors those shards need.

Example:
    python tools/convert_muse_spark_torch_dist_to_hf.py --input-dir <save>/iter_0000029 \\
        --hf-checkpoint /root/models/Muse-Spark-Open-816B-A42B --output-dir <run_dir>/hf
"""

import argparse
import json
import logging
import multiprocessing
import re
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

from miles.backends.megatron_utils.megatron_to_hf.muse_spark import (
    convert_muse_spark_1x_to_hf,
    muse_spark_1x_checkpoint_layout,
)
from miles.backends.megatron_utils.megatron_to_hf.processors import remove_padding
from miles.backends.training_utils.checkpoint.hf_layout import CheckpointLayoutWriter
from miles_plugins.models.muse_spark.config import parse_muse_spark_1x_config
from miles_plugins.models.muse_spark.weight_mapping import build_mappings

logger = logging.getLogger(__name__)

# torch_dist stores a layer's experts stacked: ``...mlp.experts.experts.linear_fc1.weight`` is ``[experts, ...]``,
# and expert E is Megatron's ``...mlp.experts.linear_fc1.weight<E>``.
_EXPERT_PARAM = re.compile(r"(.+\.mlp\.experts)\.(linear_fc[12]\.weight)(\d+)")
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")


def _source(megatron_param: str) -> tuple[str, int | None]:
    """The torch_dist key holding a Megatron parameter, and its expert index inside that key."""
    match = _EXPERT_PARAM.fullmatch(megatron_param)
    if match is None:
        return megatron_param, None
    return f"{match[1]}.experts.{match[2]}", int(match[3])


def _hf_names(mapping) -> list[str]:
    return list(mapping.hf_param.values()) if isinstance(mapping.hf_param, dict) else [mapping.hf_param]


def _sources_for_shards(mappings, weight_map: dict[str, str], shards: set[str]) -> dict[str, list]:
    """torch_dist key -> the mappings that write into ``shards``, in checkpoint order."""
    sources: dict[str, list] = {}
    for mapping in mappings:
        if any(weight_map[name] in shards for name in _hf_names(mapping)):
            sources.setdefault(_source(mapping.megatron_param)[0], []).append(mapping)
    return sources


def _load(reader: dcp.FileSystemReader, metadata, key: str) -> torch.Tensor:
    entry = metadata.state_dict_metadata[key]
    state = {key: torch.empty(entry.size, dtype=entry.properties.dtype)}
    dcp.load(state, storage_reader=reader, planner=DefaultLoadPlanner(allow_partial_load=True), no_dist=True)
    return state[key]


def convert_worker(input_dir: Path, hf_checkpoint: Path, output_dir: Path, rank: int, world_size: int) -> None:
    """Write the output shards owned by ``rank``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    spark = parse_muse_spark_1x_config(json.loads((hf_checkpoint / "config.json").read_text()))
    writer = CheckpointLayoutWriter(
        output_dir,
        hf_checkpoint,
        muse_spark_1x_checkpoint_layout(str(hf_checkpoint)),
        rank=rank,
        world_size=world_size,
    )
    weight_map = json.loads((hf_checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    owned = set(sorted(set(weight_map.values()))[rank::world_size])
    reader = dcp.FileSystemReader(str(input_dir))
    metadata = reader.read_metadata()
    converter_args = SimpleNamespace(hf_checkpoint=str(hf_checkpoint))
    for key, mappings in _sources_for_shards(build_mappings(spark), weight_map, owned).items():
        source = _load(reader, metadata, key)
        for mapping in mappings:
            _, expert = _source(mapping.megatron_param)
            param = remove_padding(
                mapping.megatron_param, source if expert is None else source[expert], spark.vocab_size
            )
            for name, tensor in convert_muse_spark_1x_to_hf(converter_args, mapping.megatron_param, param):
                writer.write(name, tensor)
        del source
    writer.finish()


def convert(input_dir: Path, hf_checkpoint: Path, output_dir: Path, num_workers: int) -> None:
    output_dir.mkdir(parents=True)
    start = time.monotonic()
    jobs = [(input_dir, hf_checkpoint, output_dir, rank, num_workers) for rank in range(num_workers)]
    with multiprocessing.get_context("spawn").Pool(num_workers) as pool:
        pool.starmap(convert_worker, jobs)
    for path in hf_checkpoint.iterdir():
        if path.is_file() and path.suffix not in _WEIGHT_SUFFIXES:
            shutil.copy2(path, output_dir / path.name)
    logger.info(f"Converted {input_dir} into {output_dir} in {time.monotonic() - start:.1f} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True, help="A torch_dist iteration directory.")
    parser.add_argument("--hf-checkpoint", type=Path, required=True, help="The HF checkpoint training started from.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not exist yet.")
    parser.add_argument("--num-workers", type=int, default=16)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    convert(args.input_dir, args.hf_checkpoint, args.output_dir, args.num_workers)


if __name__ == "__main__":
    main()
