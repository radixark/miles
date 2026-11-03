"""Keep the first N decoder layers of an HF safetensors checkpoint, for single-host runs of a large model.

python tools/prune_hf_layers.py --src /root/models/GLM-4.7-Flash --dst /root/models/GLM-4.7-Flash-4layer --num-layers 4
"""

import argparse
import json
import re
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

_LAYER_KEY = re.compile(r"(?:^|\.)layers\.(\d+)\.")
# multi-token-prediction layers sit after the decoder layers and would be cut off anyway
_MTP_CONFIG_KEYS = ("num_nextn_predict_layers",)


def _kept(key: str, num_layers: int) -> bool:
    match = _LAYER_KEY.search(key)
    return match is None or int(match.group(1)) < num_layers


def prune(src: Path, dst: Path, num_layers: int) -> None:
    config = json.loads((src / "config.json").read_text())
    if num_layers >= config["num_hidden_layers"]:
        raise ValueError(f"{src} has {config['num_hidden_layers']} layers; nothing to prune to {num_layers}")
    dst.mkdir(parents=True, exist_ok=True)

    weight_map = {}
    for shard in sorted(src.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as reader:
            tensors = {key: reader.get_tensor(key) for key in reader.keys() if _kept(key, num_layers)}
        if tensors:
            save_file(tensors, dst / shard.name, metadata={"format": "pt"})
            weight_map.update(dict.fromkeys(tensors, shard.name))
    (dst / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2))

    config["num_hidden_layers"] = num_layers
    for key in _MTP_CONFIG_KEYS:
        if key in config:
            config[key] = 0
    (dst / "config.json").write_text(json.dumps(config, indent=2))
    # tokenizer, chat template and generation config travel unchanged
    for path in src.iterdir():
        if (
            path.is_file()
            and path.suffix != ".safetensors"
            and path.name not in ("config.json", "model.safetensors.index.json")
        ):
            shutil.copy2(path, dst / path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    args = parser.parse_args()
    prune(args.src, args.dst, args.num_layers)


if __name__ == "__main__":
    main()
