"""Deterministic token-id prompts for Step5 golden outputs.

Lengths cover a short prompt, one that crosses the 512-token sliding window, and one
past 4,103 tokens where the DSA layers start selecting regions instead of attending
everything. Ids avoid the special-token range near the top of the vocabulary.
"""

import json
import sys

import torch

LENGTHS = (32, 600, 5000)
LOW, HIGH = 1000, 120000


def main(out_path: str) -> None:
    generator = torch.Generator().manual_seed(20260928)
    prompts = [torch.randint(LOW, HIGH, (n,), generator=generator).tolist() for n in LENGTHS]
    with open(out_path, "w") as f:
        json.dump(prompts, f)
    print(f"wrote {out_path}: lengths {[len(p) for p in prompts]}")


if __name__ == "__main__":
    main(sys.argv[1])
