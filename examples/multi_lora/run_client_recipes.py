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


def run_sft(base_url: str, base_model: str, steps: int) -> None:
    from tinker_cookbook.recipes import sl_loop

    sl_loop.main(
        sl_loop.Config(
            base_url=base_url,
            model_name=base_model,
            log_path=tempfile.mkdtemp(prefix="cookbook-sl-"),
            batch_size=4,
            max_length=1024,
            lora_rank=8,
            save_every=0,
            ttl_seconds=None,
            max_steps=steps,
        )
    )


def run_rl(base_url: str, base_model: str, steps: int) -> None:
    from tinker_cookbook.recipes import rl_loop

    rl_loop.main(
        rl_loop.Config(
            base_url=base_url,
            model_name=base_model,
            log_path=tempfile.mkdtemp(prefix="cookbook-rl-"),
            batch_size=2,
            group_size=4,
            lora_rank=8,
            save_every=0,
            ttl_seconds=None,
            max_tokens=512,
            max_steps=steps,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--mode", choices=["sft", "rl", "both"], default="both")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()

    os.environ.setdefault("TINKER_API_KEY", "tml-cookbook-acceptance")
    if args.mode in ("sft", "both"):
        run_sft(args.base_url, args.base_model, args.steps)
    if args.mode in ("rl", "both"):
        run_rl(args.base_url, args.base_model, args.steps)
    print(f"cookbook acceptance passed: mode={args.mode} steps={args.steps}")


if __name__ == "__main__":
    main()
