#!/usr/bin/env python3
"""Download and convert datasets to Miles format for agent training.

Supports any task type — SWE-bench, Terminal-Bench, custom datasets, etc.
Each record's metadata is enriched with ``agent_name`` so that
downstream components (server.py) can route to the correct Harbor agent.
All original fields are preserved in metadata so that
prepare_harbor_tasks.py can infer the right task directory layout.

Usage examples:

    # SWE-bench from HuggingFace (defaults)
    python download_and_process_data.py \\
        --input SWE-Gym/SWE-Gym --output /root/swe_train.jsonl

    # Terminal-Bench from local JSONL
    python download_and_process_data.py \\
        --input /data/tb_tasks.jsonl --output /root/tb_train.jsonl \\
        --agent-name terminus-2 --prompt-key instruction

    # Custom dataset
    python download_and_process_data.py \\
        --input /data/my_tasks.jsonl --output /root/custom_train.jsonl \\
        --agent-name my-agent --prompt-key task_description

    # Merge multiple outputs into one mixed JSONL
    cat /root/swe_train.jsonl /root/tb_train.jsonl > /root/mixed.jsonl
"""

import argparse
import json
import tempfile
from pathlib import Path

from datasets import load_dataset

_PROMPT_KEY_FALLBACKS = ("problem_statement", "instruction", "prompt")


def convert_to_miles_format(
    input_path: str,
    output_path: str,
    *,
    limit: int | None = None,
    split: str = "train",
    agent_name: str = "mini-swe-agent",
    prompt_key: str = "problem_statement",
    append: bool = False,
) -> int:
    """Convert JSONL to Miles format.

    Returns the number of records written.
    """
    count = 0
    mode = "a" if append else "w"
    with open(input_path) as fin, open(output_path, mode) as fout:
        for line in fin:
            if limit is not None and count >= limit:
                break

            instance = json.loads(line)

            metadata = dict(instance)
            metadata["agent_name"] = agent_name
            metadata["split"] = split

            prompt = instance.get(prompt_key, "")
            if not prompt:
                for fallback in _PROMPT_KEY_FALLBACKS:
                    prompt = instance.get(fallback, "")
                    if prompt:
                        break

            miles_sample = {
                "prompt": prompt,
                "metadata": metadata,
            }

            fout.write(json.dumps(miles_sample) + "\n")
            count += 1

    print(f"Converted {count} samples: {input_path} -> {output_path}")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Download dataset and convert to Miles format",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="HuggingFace dataset path or local JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split (default: train)",
    )
    parser.add_argument("--limit", type=int, help="Limit number of samples")
    parser.add_argument(
        "--agent-name",
        type=str,
        default="mini-swe-agent",
        help="Harbor agent name injected into metadata " "(default: mini-swe-agent)",
    )
    parser.add_argument(
        "--prompt-key",
        type=str,
        default="problem_statement",
        help="JSON key to use as prompt text " "(default: problem_statement)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to output file instead of overwriting",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    common_kwargs = dict(
        limit=args.limit,
        split=args.split,
        agent_name=args.agent_name,
        prompt_key=args.prompt_key,
        append=args.append,
    )

    if input_path.exists() and input_path.suffix == ".jsonl":
        print(f"Processing local file: {args.input}")
        convert_to_miles_format(args.input, args.output, **common_kwargs)
    else:
        print(f"Loading HuggingFace dataset: " f"{args.input} (split={args.split})")
        ds = load_dataset(args.input, split=args.split)

        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".jsonl",
                delete=False,
            ) as tmp:
                tmp_path = tmp.name

            print(f"Downloading to temporary file: {tmp_path}")
            ds.to_json(tmp_path)

            print(f"Converting to Miles format: {args.output}")
            convert_to_miles_format(tmp_path, args.output, **common_kwargs)
        finally:
            if tmp_path and Path(tmp_path).exists():
                Path(tmp_path).unlink()

    print("Done.")


if __name__ == "__main__":
    main()
