"""Expose a BF16 checkpoint to the reference converter through FP8 config exclusions.

Symlink the BF16 weights and exclude every MoE layer from quantization; otherwise
the converter expects FP8 scales that the BF16 checkpoint no longer contains.
"""

import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16", required=True, help="BF16 checkpoint (fp8_cast_bf16.py output)")
    parser.add_argument("--fp8", required=True, help="the FP8 checkpoint it was cast from")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for name in os.listdir(args.bf16):
        link = os.path.join(args.out, name)
        if name != "config.json" and not os.path.lexists(link):
            os.symlink(os.path.join(args.bf16, name), link)

    with open(os.path.join(args.bf16, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(args.fp8, "config.json")) as f:
        quantization = json.load(f)["quantization_config"]
    moe_layers = str(config["text_config"]["moe_layers_enum"]).split(",")
    experts = {f"model.layers.{layer}.moe.experts.{w}" for layer in moe_layers for w in ("w13_weight", "w2_weight")}
    quantization["modules_to_not_convert"] = sorted(set(quantization["modules_to_not_convert"]) | experts)
    config["quantization_config"] = quantization
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"wrote {args.out}: {len(moe_layers)} MoE layers marked BF16")


if __name__ == "__main__":
    main()
