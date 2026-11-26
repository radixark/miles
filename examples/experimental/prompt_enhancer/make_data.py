"""Turn a prompt file (one user prompt per line, such as RePrompt's `data/generation_prompts_1k_filterd.txt`) into
Miles prompt data for `reward.py`.

Each row's `text` is RePrompt's training prompt: its system prompt, the user's prompt, and an open assistant turn that
already holds "Let me solve this step by step.\\n<think>", rendered with the policy's chat template and
`continue_final_message`. Miles' `--apply-chat-template` always adds a new generation prompt, so the string is rendered
here and passed with `--input-key text`. The user's prompt goes to `metadata`, where the reward reads it.

  python -m examples.experimental.prompt_enhancer.make_data RePrompt/data/generation_prompts_1k_filterd.txt \\
      /root/models/Qwen2.5-3B-Instruct /root/datasets/prompt_enhancer/train.jsonl
"""

import argparse
import json
from pathlib import Path

from examples.experimental.prompt_enhancer.reward import ASSISTANT_PREFIX, ORI_PROMPT_KEY
from transformers import AutoTokenizer

# Verbatim from RePrompt's train_reprompt.py, including its missing space before "The reasoning".
SYSTEM_PROMPT = (
    "You are Prompt Wizard assistant, designed to act as the ultimate creative muse for image generation model users. "
    "Your core purpose is to translate user request into an effective, detailed, imaginative, and optimized prompt "
    "that unlock the full potential of image generation model. "
    "You can reasoning style, subject's characteristics, background details, and interactions with color and lighting "
    "of the image."
    "The reasoning process and final prompt are enclosed within <think> </think> and<answer> </answer> tags, "
    "respectively, i.e., <think> reasoning process here </think><answer> a photo of detail </answer>."
)


def messages(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": ASSISTANT_PREFIX},
    ]


def render(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(messages(prompt), tokenize=False, continue_final_message=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompts", type=Path, help="One user prompt per line.")
    parser.add_argument("tokenizer", help="The policy's HF checkpoint or name.")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # RePrompt reads every line, stripped, blank or not.
    prompts = [line.strip() for line in args.prompts.read_text().splitlines()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for prompt in prompts:
            f.write(json.dumps({"text": render(tokenizer, prompt), "metadata": {ORI_PROMPT_KEY: prompt}}) + "\n")
    print(f"{len(prompts)} prompts in {args.out}")


if __name__ == "__main__":
    main()
