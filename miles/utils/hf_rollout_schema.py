"""Build weightless rollout schemas from Hugging Face checkpoints."""

import argparse
import json
import shutil
from pathlib import Path

_WEIGHT_INDEX = "model.safetensors.index.json"


def create_mxfp8_expert_rollout_schema(source_dir: str | Path, destination_dir: str | Path) -> Path:
    """Copy a block-wise FP8 checkpoint's configuration and tokenizer with its routed experts set to MXFP8.

    SGLang allocates the other quantized weights as block-wise FP8 and the routed experts
    as MXFP8 from this directory; the trainer supplies every weight.
    """
    source = Path(source_dir).resolve()
    destination = Path(destination_dir).resolve()
    if source == destination:
        raise ValueError("The rollout schema destination must differ from the source checkpoint")

    with (source / "config.json").open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    quantization_config = config.get("quantization_config") or {}
    if quantization_config.get("quant_method") != "fp8":
        raise ValueError(f"Expected a block-wise FP8 checkpoint, got quantization_config={quantization_config}")

    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.glob("*.safetensors")):
        raise ValueError(f"Refusing to use schema directory containing weight payloads: {destination}")
    for source_file in source.iterdir():
        if not source_file.is_file():
            continue
        if source_file.name == _WEIGHT_INDEX or source_file.suffix == ".safetensors":
            continue
        shutil.copy2(source_file, destination / source_file.name)

    config["quantization_config"] = {**quantization_config, "routed_experts_quant_method": "mxfp8"}
    config["expert_dtype"] = "fp8"
    with (destination / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
        config_file.write("\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=create_mxfp8_expert_rollout_schema.__doc__)
    parser.add_argument("--model-dir", required=True, help="Block-wise FP8 Hugging Face checkpoint.")
    parser.add_argument("--save-dir", required=True, help="Directory for the weightless rollout schema.")
    args = parser.parse_args()
    schema = create_mxfp8_expert_rollout_schema(args.model_dir, args.save_dir)
    print(f"Created MXFP8 routed-expert rollout schema at {schema}")


if __name__ == "__main__":
    main()
