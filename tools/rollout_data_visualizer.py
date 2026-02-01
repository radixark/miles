import argparse
import json
from collections import defaultdict

import torch


def print_metadata(sample):
    print(f"\nSample fields: {list(sample.keys())}")

    if sample.get("metadata"):
        print(f"Metadata fields: {list(sample['metadata'].keys())}")

    if sample.get("train_metadata"):
        print(f"Train metadata fields: {list(sample['train_metadata'].keys())}")

def group_reward_visualize(samples, print_response_example: bool = False):
    groups = defaultdict(list)
    for sample in samples:
        prompt = sample["prompt"]
        key = json.dumps(prompt, ensure_ascii=False) if isinstance(prompt, list) else prompt
        groups[key].append(sample)

    for i, (prompt_key, group) in enumerate(groups.items()):
        rewards = [s["reward"] for s in group]
        avg_reward = sum(rewards) / len(rewards)

        best_sample = max(group, key=lambda s: s["reward"])
        worst_sample = min(group, key=lambda s: s["reward"])

        print(f"Group {i+1}")
        print(f"Prompt: {prompt_key[:200]}..." if len(prompt_key) > 200 else f"Prompt: {prompt_key}")
        print(f"Rewards: {rewards}, Average reward: {avg_reward:.4f}")

        if print_response_example:
            best_response = best_sample.get("response") or best_sample.get("output") or best_sample.get("completion", "")
            print(f"\n[BEST] Reward: {best_sample['reward']}")
            print(f"Response: {best_response[:500]}..." if len(str(best_response)) > 500 else f"Response: {best_response}")

            worst_response = worst_sample.get("response") or worst_sample.get("output") or worst_sample.get("completion", "")
            print(f"\n[WORST] Reward: {worst_sample['reward']}")
            print(f"Response: {worst_response[:500]}..." if len(str(worst_response)) > 500 else f"Response: {worst_response}")


def main():
    parser = argparse.ArgumentParser(description="Visualize rollout data")
    parser.add_argument("path", help="Path to the .pt data file")
    parser.add_argument("--print-response-example", action="store_true", help="Print the response example for the best and worst samples")
    args = parser.parse_args()

    data = torch.load(args.path, weights_only=False)

    print(f"Rollout ID: {data['rollout_id']}")
    print(f"Sample count: {len(data['samples'])}")

    print_metadata(data["samples"][0])

    group_reward_visualize(data["samples"], print_response_example=args.print_response_example)


if __name__ == "__main__":
    main()

