"""Record a DSA model's own indexer top-k on a long text, for tests/manual/bench_dsa.py --topk.

    python tests/manual/dump_dsa_topk.py <hf-model-dir> <text-file> <out-dir> [--max-tokens 16384]

Loads the checkpoint with HF transformers (SDPA attention: the indexer runs in torch and its top-k is
captured by a forward hook; a few layers of a model, like the Pinaster/*-4layer slices, are enough) and
writes ``<out-dir>/layer<i>.pt``, each the int32 [S, topk] indices of one indexer layer. Consecutive
queries' overlap is what the query-block backward's gain depends on, so the text should resemble the
training data; a long reasoning trace is a good choice.
"""

import argparse
import pathlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("text")
    parser.add_argument("out_dir")
    parser.add_argument("--max-tokens", type=int, default=16384)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda"
    ).eval()
    captured = {}
    for name, module in model.named_modules():
        if type(module).__name__.endswith("Indexer"):
            module.register_forward_hook(lambda m, i, o, name=name: captured.__setitem__(name, o))
    ids = tokenizer(pathlib.Path(args.text).read_text(), return_tensors="pt").input_ids[:, : args.max_tokens].cuda()
    with torch.no_grad():
        model(ids, use_cache=False)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    positions = torch.arange(ids.shape[1], device="cuda").view(-1, 1)
    for name, indices in captured.items():
        layer = next(part for part in name.split(".") if part.isdigit())
        # HF's top-k on early queries is padded with -inf-scored future keys, which miles marks -1
        indices = torch.where(indices[0] <= positions, indices[0], -1).to(torch.int32)
        torch.save(indices.cpu(), out_dir / f"layer{layer}.pt")
        print(f"{name}: {tuple(indices[0].shape)} -> {out_dir / f'layer{layer}.pt'}")


if __name__ == "__main__":
    main()
