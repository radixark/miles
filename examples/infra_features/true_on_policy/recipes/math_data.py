"""Eval data of the math recipes: AIME-2024 in Miles' DAPO prompt wording, and the GLM-5.2 recipe's eval config.

The DAPO authors' AIME-2024 parquet (BytedTsinghua-SIA/AIME-2024) asks for "Answer: $Answer"; Miles' dapo-math-17k
prompts ask for "Answer: \\boxed{$Answer}", which ``--rm-type math`` grades. This swaps that instruction line and
keeps the rest of each prompt, dropping the parquet's repeated rows (30 problems remain).

Usage:
    python math_data.py aime <aime-2024.parquet> <aime-2024-boxed.jsonl>
    python math_data.py glm-eval-config <aime-2024-boxed.jsonl> <glm5_2_aime_eval.yaml>
"""

import json
import sys

import pandas as pd

_DAPO = (
    "Solve the following math problem step by step. The last line of your response should be of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem."
)
_MILES = (
    "Solve the following math problem step by step. The last line of your response should be of the form "
    "Answer: \\boxed{$Answer} where $Answer is the answer to the problem."
)


def write_aime(src: str, dst: str) -> None:
    rows = pd.read_parquet(src)
    seen, out = set(), []
    for prompt, reward in zip(rows["prompt"], rows["reward_model"], strict=True):
        messages = [{"role": m["role"], "content": m["content"]} for m in prompt]
        key = json.dumps(messages, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        last = messages[-1]
        assert last["content"].startswith(_DAPO + "\n"), last["content"][:120]
        last["content"] = _MILES + last["content"][len(_DAPO) :]
        out.append({"prompt": messages, "label": str(reward["ground_truth"])})
    with open(dst, "w") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} prompts to {dst}")


def write_glm_eval_config(aime: str, dst: str) -> None:
    """AIME-2024 x 8 at temperature 1.0: at the 4,096-token training cap every --eval-interval, and at 14,000 tokens
    every 40 rollouts (plus before training and at the end), scored by the run's reward."""
    config = f"""eval:
  defaults:
    n_samples_per_eval_prompt: 8
    temperature: 1.0
    top_p: 1.0
    top_k: -1
  datasets:
    - name: aime4k
      path: {aime}
      max_response_len: 4096
    - name: aime14k
      path: {aime}
      max_response_len: 14000
      interval: 40
"""
    with open(dst, "w") as f:
        f.write(config)
    print(f"wrote {dst}")


if __name__ == "__main__":
    {"aime": write_aime, "glm-eval-config": write_glm_eval_config}[sys.argv[1]](*sys.argv[2:])
