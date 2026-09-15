"""Run the official tinker-cookbook recipes against a running gateway.

The cookbook's sl_loop (SFT) and rl_loop (GRPO) are the executable definition
of the Tinker wire contract; passing them is the gateway's acceptance bar.

Requires, next to the pinned SDK (tinker==0.26.2):

    pip install git+https://github.com/thinking-machines-lab/tinker-cookbook@1f962eda3a2c

``--base-model`` must be both the name this gateway serves (--tinker-base-model)
and a HuggingFace name the cookbook can resolve a tokenizer and renderer for.
"""

import argparse
import os
import tempfile


def run_sft(base_url: str, base_model: str, steps: int, batch_size: int | None, lora_rank: int) -> None:
    from tinker_cookbook.recipes import sl_loop

    sl_loop.main(
        sl_loop.Config(
            base_url=base_url,
            model_name=base_model,
            log_path=tempfile.mkdtemp(prefix="cookbook-sl-"),
            batch_size=batch_size or 4,
            max_length=1024,
            lora_rank=lora_rank,
            save_every=0,
            ttl_seconds=None,
            max_steps=steps,
        )
    )


def run_rl(
    base_url: str,
    base_model: str,
    steps: int,
    batch_size: int | None,
    group_size: int,
    lora_rank: int,
    max_tokens: int,
) -> None:
    from tinker_cookbook.recipes import rl_loop

    rl_loop.main(
        rl_loop.Config(
            base_url=base_url,
            model_name=base_model,
            log_path=tempfile.mkdtemp(prefix="cookbook-rl-"),
            batch_size=batch_size or 2,
            group_size=group_size,
            lora_rank=lora_rank,
            save_every=0,
            ttl_seconds=None,
            max_tokens=max_tokens,
            max_steps=steps,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--mode", choices=["sft", "rl", "both"], default="both")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    os.environ.setdefault("TINKER_API_KEY", "tml-cookbook-acceptance")
    if args.mode in ("sft", "both"):
        run_sft(args.base_url, args.base_model, args.steps, args.batch_size, args.lora_rank)
    if args.mode in ("rl", "both"):
        run_rl(
            args.base_url,
            args.base_model,
            args.steps,
            args.batch_size,
            args.group_size,
            args.lora_rank,
            args.max_tokens,
        )
    print(f"cookbook acceptance passed: mode={args.mode} steps={args.steps}")


if __name__ == "__main__":
    main()
