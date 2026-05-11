---
title: Agentic Chat Templates
description: How to verify and override the chat template applied during multi-turn rollout.
---

# Agentic Chat Templates

In agentic / multi-turn workflows, Miles uses SGLang's pretokenized prefix mechanism
so the conversation history is not re-tokenized every turn. That requires the chat
template to satisfy an **append-only invariant**: rendering messages `[1..N]` must
produce a string that is an exact prefix of rendering `[1..N+1]`.

Some community templates violate this. They use `loop.last` or other
context-dependent Jinja logic that flips bits across turns, and the result is silent
tokenization drift, divergent log-probabilities, and gradient blow-up after a few
iterations of multi-turn RL.

Miles ships a verifier and an autofix.

## Quick start

### Verify a HuggingFace template

```bash
python scripts/tools/verify_chat_template.py --model Qwen/Qwen3-0.6B
```

Failing output (illustrative):

```text
Template source: HuggingFace: Qwen/Qwen3-0.6B
Thinking cases:  disabled

  [FAIL] single_tool-N3                Prefix mismatch
  [PASS] single_tool-N3-no_tools
  [FAIL] multi_turn-N4                 Prefix mismatch
  ...
Verdict: FAIL - template is NOT append-only after last user message
```

### Apply Miles's autofix

If a fixed template ships for that model, `--autofix` swaps it in and re-runs the
suite:

```bash
python scripts/tools/verify_chat_template.py --model Qwen/Qwen3-0.6B --autofix
```

```text
Template source: fixed template: .../templates/qwen3_fixed.jinja
Verdict: PASS - template IS append-only after last user message
```

### Verify a local Jinja file

```bash
python scripts/tools/verify_chat_template.py --template path/to/my_template.jinja
```

### Include thinking-specific cases

For Qwen3.5, GLM-5, and other models that toggle `enable_thinking`, add `--thinking`
to also run thinking-specific trajectories.

```bash
python scripts/tools/verify_chat_template.py --model Qwen/Qwen3.5-0.8B --autofix --thinking
```

## CLI

```text
usage: verify_chat_template.py (--template PATH | --model MODEL_ID)
                               [--autofix] [--thinking]
```

| Flag | What |
|---|---|
| `--template PATH` | Local `.jinja` template. |
| `--model MODEL_ID` | HF model ID. |
| `--autofix` | Apply Miles's fixed template if available. |
| `--thinking` | Also run thinking-specific cases. |

Exit code is `0` on pass, `1` on fail.

## How it works

For each test case (a list of messages), the verifier renders progressive prefixes
and checks the invariant character by character:

```python
for n in range(1, len(messages)):
    full   = render(messages[: n + 1])
    prefix = render(messages[: n])
    assert full.startswith(prefix), f"break between turn {n} and {n+1}"
```

The trajectory specs and cases live in
`miles/utils/test_utils/chat_template_verify.py`. Standard cases cover
single-tool, multi-turn, parallel-tool, and long-chain trajectories; the thinking
suite adds variants that toggle `enable_thinking`.

A break almost always comes from `loop.last`, conditional whitespace, or a closing
token that's only emitted on the final turn.

## Using the fixed template at training time

Once you have the right template, point Miles at it:

```bash
ROLLOUT_ARGS+=(
   --chat-template-path /opt/miles/utils/chat_template_utils/templates/qwen3_fixed.jinja
)
```

Built-in fixed templates that ship with Miles live under
`miles/utils/chat_template_utils/templates/` (e.g. `qwen3_fixed.jinja`,
`qwen3.5_fixed.jinja`, `qwen3_thinking_2507_and_next_fixed.jinja`).

## What "append-only" buys you

| Without it | With it |
|---|---|
| Re-tokenize everything each turn | Tokenize only the new turn |
| O(N²) tokenization cost | O(N) tokenization cost |
| Subtle drift between turns | Bit-stable tokens |
| Multi-turn RL collapses after ~50 steps | Stable across thousands of steps |

Running the verifier as part of every model's pre-flight is recommended.
