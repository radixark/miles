"""Plot eval scores across all rollout steps from dumped eval data.

Usage:
    # Original scores only
    python tools/plot_eval_scores.py --eval_path /path/to/rollout_data

    # With truncated-response scores side by side
    python tools/plot_eval_scores.py --eval_path /path/to/rollout_data \
        --hf_checkpoint /path/to/model --truncate_length 3000
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from miles.rollout.rm_hub.math_utils import grade_answer_verl


def _get_eval_files(eval_path: str):
    eval_dir = Path(eval_path)
    pattern = re.compile(r"^eval_(\d+)\.pt$")
    files = []
    for pt_file in sorted(eval_dir.glob("eval_*.pt")):
        m = pattern.match(pt_file.name)
        if m is not None:
            files.append((int(m.group(1)), pt_file))
    return files


def load_eval_data(eval_path: str) -> dict[int, list[float]]:
    """Load all eval_*.pt files and extract per-step reward lists."""
    step_rewards: dict[int, list[float]] = {}
    for rollout_id, pt_file in _get_eval_files(eval_path):
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        rewards = []
        for sample in data["samples"]:
            reward = sample.get("reward")
            if reward is None:
                continue
            if isinstance(reward, dict):
                rewards.append(sum(reward.values()))
            else:
                rewards.append(float(reward))
        step_rewards[rollout_id] = rewards
    return step_rewards


def load_eval_data_truncated(eval_path: str, tokenizer, truncate_length: int) -> dict[int, list[float]]:
    """Load eval data, truncate each response to truncate_length tokens, re-score."""
    step_rewards: dict[int, list[float]] = {}
    for rollout_id, pt_file in _get_eval_files(eval_path):
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        rewards = []
        for sample_dict in data["samples"]:
            tokens = sample_dict.get("tokens", [])
            response_length = sample_dict.get("response_length", 0)
            label = sample_dict.get("label")

            if response_length <= truncate_length:
                # No truncation needed, use original reward
                reward = sample_dict.get("reward")
                if reward is None:
                    continue
                if isinstance(reward, dict):
                    rewards.append(sum(reward.values()))
                else:
                    rewards.append(float(reward))
            else:
                # Truncate response tokens to truncate_length
                response_tokens = tokens[-response_length:][:truncate_length]
                truncated_response = tokenizer.decode(response_tokens, skip_special_tokens=False)
                reward = 1.0 if grade_answer_verl(truncated_response, label) else 0.0
                rewards.append(reward)

        step_rewards[rollout_id] = rewards
    return step_rewards


def plot_eval_scores_side_by_side(
    step_rewards_orig: dict[int, list[float]],
    step_rewards_trunc: dict[int, list[float]],
    truncate_length: int,
    output_path: str | None = None,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    steps = sorted(step_rewards_orig.keys())
    mean_orig = [sum(step_rewards_orig[s]) / len(step_rewards_orig[s]) for s in steps]
    ax1.plot(steps, mean_orig, marker="o", linewidth=2, markersize=4)
    ax1.set_xlabel("Rollout Step")
    ax1.set_ylabel("Mean Eval Reward (Accuracy)")
    ax1.set_title("Original Eval Score")
    ax1.grid(True, alpha=0.3)

    steps_t = sorted(step_rewards_trunc.keys())
    mean_trunc = [sum(step_rewards_trunc[s]) / len(step_rewards_trunc[s]) for s in steps_t]
    ax2.plot(steps_t, mean_trunc, marker="s", linewidth=2, markersize=4, color="tab:orange")
    ax2.set_xlabel("Rollout Step")
    ax2.set_title(f"Eval Score (truncated to {truncate_length} tokens)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if output_path is None:
        output_path = "eval_scores.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def plot_eval_scores(step_rewards: dict[int, list[float]], output_path: str | None = None):
    steps = sorted(step_rewards.keys())
    mean_rewards = [sum(step_rewards[s]) / len(step_rewards[s]) for s in steps]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, mean_rewards, marker="o", linewidth=2, markersize=4)
    ax.set_xlabel("Rollout Step")
    ax.set_ylabel("Mean Eval Reward")
    ax.set_title("Eval Score over Training Steps")
    ax.grid(True, alpha=0.3)

    if output_path is None:
        output_path = "eval_scores.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def _print_summary(label: str, step_rewards: dict[int, list[float]]):
    print(f"\n{label}: {len(step_rewards)} eval steps")
    for s in sorted(step_rewards.keys()):
        n = len(step_rewards[s])
        mean = sum(step_rewards[s]) / n if n > 0 else 0
        print(f"  step {s}: {n} samples, mean reward = {mean:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Plot eval scores from dumped eval data")
    parser.add_argument("--eval_path", type=str, required=True,
                        help="Path to the rollout_data directory containing eval_*.pt files")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (default: eval_scores.png)")
    parser.add_argument("--hf_checkpoint", type=str, default=None,
                        help="HF model checkpoint path (required for --truncate_length)")
    parser.add_argument("--truncate_length", type=int, default=None,
                        help="Truncate response to this many tokens and re-score (e.g. 3000)")
    args = parser.parse_args()

    step_rewards_orig = load_eval_data(args.eval_path)
    if not step_rewards_orig:
        print(f"No eval_*.pt files found in {args.eval_path}")
        return

    _print_summary("Original", step_rewards_orig)

    if args.truncate_length is not None:
        assert args.hf_checkpoint is not None, "--hf_checkpoint is required when using --truncate_length"
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

        step_rewards_trunc = load_eval_data_truncated(args.eval_path, tokenizer, args.truncate_length)
        _print_summary(f"Truncated ({args.truncate_length} tokens)", step_rewards_trunc)
        plot_eval_scores_side_by_side(step_rewards_orig, step_rewards_trunc, args.truncate_length, args.output)
    else:
        plot_eval_scores(step_rewards_orig, args.output)


if __name__ == "__main__":
    main()
