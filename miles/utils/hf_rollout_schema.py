"""Build metadata-only rollout schemas from Hugging Face checkpoints."""

import json
import re
import shutil
import struct
from pathlib import Path

from miles.utils.hf_parameter_names import get_param_name_remap
from miles.utils.mxfp8 import MXFP8_GROUP_SIZE

MXFP8_SKIP_WEIGHT_SUBSTRINGS = (
    "layernorm",
    "embed",
    "router",
    "mlp.gate.",
    "norm",
    "lm_head",
    "eh_proj",
    "weights_proj",
    "head.",
    "wo_a",
    "ffn.gate.",
    "compressor.",
)

_MXFP8_SOURCE_DTYPES = {"BF16", "F16", "F32", "F8_E4M3", "F8_E4M3FN", "F8_E4M3FNUZ"}
_SCHEMA_MARKER = ".miles-rollout-schema.json"
_WEIGHT_INDEX = "model.safetensors.index.json"


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as json_file:
        return json.load(json_file)


def _read_safetensors_header(path: Path) -> dict[str, dict]:
    """Read tensor dtype and shape metadata without mapping tensor payloads."""
    with path.open("rb") as tensor_file:
        header_size_bytes = tensor_file.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError(f"Invalid safetensors header in {path}")
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        header = tensor_file.read(header_size)
        if len(header) != header_size:
            raise ValueError(f"Truncated safetensors header in {path}")
    return json.loads(header)


def _load_tensor_metadata(source_dir: Path, weight_map: dict[str, str]) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    names_by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        names_by_shard.setdefault(shard, []).append(name)

    for shard, names in names_by_shard.items():
        header = _read_safetensors_header(source_dir / shard)
        for name in names:
            if name not in header:
                raise KeyError(f"{name} is listed in {_WEIGHT_INDEX} but missing from {shard}")
            metadata[name] = header[name]
    return metadata


def _natural_key(value: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part for part in re.findall(r"\d+|\D+", value)]


def _should_use_mxfp8(name: str, metadata: dict, skip_substrings: tuple[str, ...]) -> bool:
    if not name.endswith(".weight") or any(part in name for part in skip_substrings):
        return False
    shape = metadata.get("shape", [])
    if len(shape) < 2 or shape[-1] % MXFP8_GROUP_SIZE != 0:
        return False
    dtype = metadata.get("dtype")
    if dtype in _MXFP8_SOURCE_DTYPES:
        return True
    # Trainer-owned rollout replaces packed source experts with MXFP8 at the
    # first sync, so reserve the unpacked target layout in the schema.
    return dtype == "I8" and ".experts." in name


def build_mxfp8_quantization_config(
    source_dir: str | Path,
    *,
    extra_high_precision_layers_hf: tuple[str, ...] = (),
) -> dict:
    """Derive the SGLang MXFP8 allocation layout from checkpoint headers."""
    source = Path(source_dir)
    config_path = source / "config.json"
    index_path = source / _WEIGHT_INDEX
    if not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(f"Expected config.json and {_WEIGHT_INDEX} under {source}")

    weight_map = _load_json(index_path)["weight_map"]
    remap_name = get_param_name_remap(str(config_path), weight_map)
    tensor_metadata = _load_tensor_metadata(source, weight_map)
    skip_substrings = (*MXFP8_SKIP_WEIGHT_SUBSTRINGS, *extra_high_precision_layers_hf)

    ignored_modules = set()
    for source_name, metadata in tensor_metadata.items():
        if not source_name.endswith(".weight"):
            continue
        hf_name = remap_name(source_name)
        if not _should_use_mxfp8(hf_name, metadata, skip_substrings) and ".experts." not in hf_name:
            ignored_modules.add(hf_name.removesuffix(".weight"))

    return {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "mxfp8",
        "weight_block_size": [1, MXFP8_GROUP_SIZE],
        "scale_fmt": "ue8m0",
        "modules_to_not_convert": sorted(ignored_modules, key=_natural_key),
    }


def _copy_root_metadata(source: Path, destination: Path) -> None:
    for source_file in source.iterdir():
        if not source_file.is_file():
            continue
        if source_file.name == _WEIGHT_INDEX or source_file.suffix == ".safetensors":
            continue
        shutil.copy2(source_file, destination / source_file.name)


def create_mxfp8_rollout_schema(source_dir: str | Path, destination_dir: str | Path) -> Path:
    """Create a small MXFP8 config/tokenizer directory with no weight payloads."""
    source = Path(source_dir).resolve()
    destination = Path(destination_dir).resolve()
    if source == destination:
        raise ValueError("The rollout schema destination must differ from the source checkpoint")

    quantization_config = build_mxfp8_quantization_config(source)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.glob("*.safetensors")):
        raise ValueError(f"Refusing to use schema directory containing weight payloads: {destination}")
    _copy_root_metadata(source, destination)

    config = _load_json(source / "config.json")
    config["quantization_config"] = quantization_config
    config["expert_dtype"] = "fp8"
    with (destination / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
        config_file.write("\n")

    marker = {
        "format": "miles-metadata-only-rollout-schema-v1",
        "source": str(source),
        "weight_payloads": False,
    }
    with (destination / _SCHEMA_MARKER).open("w", encoding="utf-8") as marker_file:
        json.dump(marker, marker_file, indent=2)
        marker_file.write("\n")
    return destination
