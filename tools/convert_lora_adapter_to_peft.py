"""Write a 2D-per-expert bf16 PEFT adapter from a miles adapter dir, training or sampler (CPU only)."""

import argparse
import json
from pathlib import Path

from miles.backends.training_utils.checkpoint.adapter.adapter_exports import write_peft_adapter
from miles.backends.training_utils.checkpoint.adapter.readers import (
    AdapterSourceKind,
    MilesAdapterDirReader,
    classify_adapter_source,
)
from miles.utils.lora.hf_lora_targets import get_hf_lora_targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--hf-config", type=Path, required=True, help="the base model's config.json")
    parser.add_argument(
        "--lora-dropout", type=float, help="the training run's dropout; required when src holds no export of its own"
    )
    args = parser.parse_args()
    # in place, the PEFT tensors would replace the ones sglang serves from src
    assert not args.dst.exists(), f"{args.dst} exists; pass a new directory"
    targets = get_hf_lora_targets(json.loads(args.hf_config.read_text()))
    kind = classify_adapter_source(args.src, targets, load_optimizer=False)
    assert (
        kind == AdapterSourceKind.ADAPTER_DIR
    ), f"{args.src} is a {kind.value} source; load it in a training run and save it in the miles format first"
    reader = MilesAdapterDirReader(args.src, targets)
    # a sampler export records the training run's dropout; a training dir records none
    config = args.src / "adapter_config.json"
    lora_dropout = json.loads(config.read_text())["lora_dropout"] if config.is_file() else args.lora_dropout
    assert lora_dropout is not None, f"{args.src} records no lora_dropout; pass --lora-dropout"
    write_peft_adapter(
        args.dst, reader.read_weights(), reader.source.parametrization, targets, lora_dropout=lora_dropout
    )


if __name__ == "__main__":
    main()
