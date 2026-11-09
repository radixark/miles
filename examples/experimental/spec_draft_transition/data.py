"""Prepare prompt-disjoint data and generate untruncated teacher trajectories.

The generation worker uses one GPU and one SGLang engine. The CLI arguments
are deliberately explicit so a manifest, rather than machine defaults, owns
the experiment's data partitioning and sampling settings.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import polars as pl
import sglang as sgl
from datasets import load_dataset
from huggingface_hub import HfApi
from tap import Tap
from transformers import AutoTokenizer


class Args(Tap):
    mode: str
    root: str
    teacher: str
    shard: int = 0
    shards: int = 40
    batch: int = 256
    maximum_response: int = 1024
    seed: int = 71319


def prepare(args: Args) -> None:
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    revision = HfApi().dataset_info("HuggingFaceH4/ultrachat_200k").sha
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_gen", revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    rows, seen = [], set()
    for item in dataset:
        messages = [{"role": "user", "content": item["messages"][0]["content"]}]
        identity = hashlib.sha256(messages[0]["content"].strip().encode()).hexdigest()
        if identity in seen:
            continue
        ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                             add_generation_prompt=True, enable_thinking=False)
        if len(ids) > 1024:
            continue
        seen.add(identity)
        number = len(rows)
        split = "validation" if number < 2000 else "test" if number < 4000 else "train"
        rows.append({"prompt_id": identity, "split": split, "input_ids": ids})
        if len(rows) == 84000:
            break
    assert len(rows) == 84000, len(rows)
    pl.DataFrame(rows).write_parquet(root / "prompts.parquet")
    (root / "prompt-provenance.json").write_text(json.dumps({"dataset": "HuggingFaceH4/ultrachat_200k", "revision": revision, "rows": len(rows), "splits": {"train": 80000, "validation": 2000, "test": 2000}}, indent=2))
    print("prepared", len(rows), flush=True)


def generate(args: Args) -> None:
    # The existing SGLang runtime imports Transformer Engine through an
    # optional Megatron debug plugin. A derived venv needs its library roots
    # explicitly because TE otherwise searches the derived venv's purelib.
    library_dirs = []
    for package, variable in [('cudnn','CUDNN_PATH'),('cublas','CUBLAS_PATH')]:
        found = next((Path(base)/'nvidia'/package for base in sys.path
                      if (Path(base)/'nvidia'/package/'lib').exists()),None)
        if found is not None:
            os.environ[variable] = str(found)
            library_dirs.append(str(found/'lib'))
    os.environ['LD_LIBRARY_PATH'] = ':'.join(library_dirs + [os.environ.get('LD_LIBRARY_PATH','')])
    root = Path(args.root)
    prompts = pl.read_parquet(root / "prompts.parquet").with_row_index("row")
    subset = prompts.filter(pl.col("row") % args.shards == args.shard).to_dicts()
    engine = sgl.Engine(model_path=args.teacher, dtype="bfloat16", mem_fraction_static=0.70,
                        context_length=2048, max_running_requests=128, random_seed=args.seed + args.shard,
                        disable_cuda_graph=False, log_level="warning")
    output = root / "generated"
    output.mkdir(exist_ok=True)
    for temperature in [0.6, 1.0, 1.4]:
        destination = output / f"t{temperature}-shard{args.shard:02d}.jsonl"
        if destination.exists() and destination.with_suffix(".done").exists():
            continue
        with destination.open("w") as handle:
            for start in range(0, len(subset), args.batch):
                batch = subset[start:start + args.batch]
                results = engine.generate(input_ids=[item["input_ids"] for item in batch],
                    sampling_params={"temperature": temperature, "top_p": 1.0, "top_k": -1,
                                     "max_new_tokens": args.maximum_response}, return_logprob=True)
                for prompt, result in zip(batch, results, strict=True):
                    # Use returned sampled IDs, never decode/re-tokenize model text.
                    output_ids = result.get("output_ids")
                    if output_ids is None:
                        output_ids = [entry[1] for entry in result["meta_info"]["output_token_logprobs"]]
                    assert output_ids and all(isinstance(token, int) for token in output_ids)
                    row = {"prompt_id": prompt["prompt_id"], "split": prompt["split"],
                           "prompt_ids": prompt["input_ids"], "response_ids": output_ids,
                           "temperature": temperature}
                    handle.write(json.dumps(row) + "\n")
                handle.flush()
                print(json.dumps({"shard": args.shard, "temperature": temperature, "completed": min(start + args.batch, len(subset)), "total": len(subset)}), flush=True)
        destination.with_suffix(".done").write_text("complete\n")
    engine.shutdown()


if __name__ == "__main__":
    config = Args().parse_args()
    if config.mode == "prepare":
        prepare(config)
    elif config.mode == "generate":
        generate(config)
    else:
        raise ValueError(config.mode)
