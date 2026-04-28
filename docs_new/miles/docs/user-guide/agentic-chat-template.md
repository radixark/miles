---
title: Agentic Chat Templates
description: How to verify (and override) the chat template applied during multi-turn rollout.
---

# Agentic Chat Templates

In agentic / multi-turn workflows, Miles uses SGLang's **pretokenized prefix** mechanism so
the conversation history isn't re-tokenised every turn. That requires the chat template to
satisfy an **append-only invariant**: rendering messages `[1..N]` must produce a string
that is an exact prefix of rendering `[1..N+1]`.

A surprising number of community templates violate this — they use `loop.last` or other
context-dependent Jinja logic that flips bits across turns. The result: silent
tokenisation drift, divergent log-probabilities, and gradient blow-up after a few
iterations of multi-turn RL.

Miles ships a one-click verifier and an autofix.

## Quick start

### Verify a HuggingFace template

```bash
python scripts/tools/verify_chat_template.py --model Qwen/Qwen3-0.6B
```

Failing output:

```text
Template source: HuggingFace: Qwen/Qwen3-0.6B
Thinking cases:  disabled

  [FAIL] single_tool-N3                -- Prefix mismatch!
  [PASS] single_tool-N3-no_tools
  [FAIL] multi_turn-N4                 -- Prefix mismatch!
  [FAIL] multi_tool_single_turn-N3     -- Prefix mismatch!
  ...
Results: 2/13 passed, 11 failed
Verdict: FAIL - template is NOT append-only after last user message
```

### Apply Miles's autofix

If we ship a fix for that model, `--autofix` swaps the template and re-runs the suite:

```bash
python scripts/tools/verify_chat_template.py --model Qwen/Qwen3-0.6B --autofix
```

```text
Template source: fixed template: .../templates/qwen3_fixed.jinja
Results: 13/13 passed, 0 failed
Verdict: PASS - template IS append-only after last user message
```

### Verify a local Jinja file

```bash
python scripts/tools/verify_chat_template.py --template path/to/my_template.jinja
```

### Include thinking-specific cases

For Qwen3.5 / GLM-5 / models that toggle `enable_thinking`, add `--thinking` (29 cases
total = 13 standard + 16 thinking).

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

Exit code is **0** on pass, **1** on fail — easy to wire into CI.

## How it works

For each test case (a list of messages), the verifier renders progressive prefixes and
checks the invariant character-by-character:

```python
for n in range(1, len(messages)):
    full   = render(messages[: n + 1])
    prefix = render(messages[: n])
    assert full.startswith(prefix), f"break between turn {n} and {n+1}"
```

A break almost always comes from `loop.last`, conditional whitespace, or a closing token
that's only emitted "if this is the final turn."

## Using the fixed template at training time

Once you've identified the right template, point Miles at it:

```bash
ROLLOUT_ARGS+=(
   --chat-template-path /opt/miles/utils/chat_template_utils/templates/qwen3_fixed.jinja
)
```

If the fix is built in for the model you're using, `--apply-chat-template` automatically
picks it up.

## What "append-only" buys you

| Without it | With it |
|---|---|
| Re-tokenise everything each turn | Tokenise only the new turn |
| O(N²) tokenisation cost | O(N) tokenisation cost |
| Subtle drift between turns | Bit-stable tokens |
| Multi-turn RL collapses after ~50 steps | Stable for thousands of steps |

This is one of those "boring infra fixes that quietly determines whether your run works."
We strongly recommend running the verifier as part of every model's pre-flight.
