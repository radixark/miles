---
title: Agentic Chat Templates (TITO)
description: How to turn on and verify Token-In-Token-Out (TITO) for multi-turn agentic rollout.
---

# Agentic Chat Templates (TITO)

Multi-turn agentic rollout in Miles runs on **TITO** (Token-In-Token-Out): each turn's token sequence is a bit-perfect prefix of the next, so the trainer sees exactly the tokens the engine produced — no re-tokenization, no drift. The *why* is in the blog ([No Token Left Behind](https://lmsys.org/blog/2026-05-13-no-token-left-behind/)); this page is *how*.

Your harness only ever sends and receives **OpenAI chat messages**, never tokens. Miles keeps the per-trajectory append-only token buffer (ids + logprobs + routed experts) internally and ships it straight to training.

## Prerequisites

Your rollout loop must keep two invariants, or TITO is rejected at runtime:

- **Append-only messages.** Each turn = previous messages + new ones on the tail; past turns are never edited. The only exception is retrying the latest turn — a single-step rollback to the last assistant checkpoint. Diverging earlier, or rolling back more than one turn, is rejected.
- **Declared roles match the flag.** `--tito-allowed-append-roles` must list exactly the roles your harness appends after the first assistant turn (`tool` is always implied) — Miles resolves a prefix-stable template for that exact set, and appending any role outside it is rejected at runtime.

## Pick your `--tito-model`

No auto-detection — pick the family matching your model. Miles then auto-resolves the fixed template from `(--tito-model, --tito-allowed-append-roles)`; pass `--chat-template-path` only to override. The table below lists common models; the full set and verified role surfaces live in [issue #712](https://github.com/radixark/miles/issues/712).

| Your model | `--tito-model` | Max `--tito-allowed-append-roles` |
|---|---|---|
| Qwen3.5 | `qwen35` | `tool user` |
| GLM-4.7 / GLM-5 | `glm47` | `tool user system` |
| NVIDIA Nemotron 3 Super / Ultra | `nemotron3` | `tool user system` |
| Kimi K2.5 / K2.6 | `kimi25` / `kimi26` | `tool user` |
| DeepSeek-V3.2 / V4 | `deepseekv32` / `deepseekv4` | `tool` |
| anything else | `default` | `tool` |

More models: [issue #712](https://github.com/radixark/miles/issues/712).

## Turn it on

```bash
ROLLOUT_ARGS+=(
   --use-session-server          # entry point; required for the two flags below
   --hf-checkpoint Qwen/Qwen3-4B
   --tito-model qwen3
   --tito-allowed-append-roles tool user
)
```

## Example

A full multi-turn agentic setup on the session-server TITO path lives in [`examples/experimental/swe-agent-v2`](https://github.com/radixark/miles/tree/main/examples/experimental/swe-agent-v2): its launchers wire `--use-session-server` + `--tito-model glm47` + `--tito-allowed-append-roles user tool` against a real SWE agent.

## Add a new model

Models in the table are verified by Miles maintainers — just pick the family. To support a new model (or a new append-role surface), register a `TITOTokenizer` subclass plus its fixed Jinja template (or HF-native + kwargs) and `SUPPORTED_TEMPLATES` rows in [`tito_tokenizer.py`](../../miles/utils/chat_template_utils/tito_tokenizer.py), then verify with both scripts — either failing blocks it. Each prints `Verdict: PASS/FAIL`.

```bash
# CPU / fast — rendered token sequence is append-only
python scripts/tools/verify_chat_template.py \
    --model <hf-id> --tito-model <family> --tito-allowed-append-roles tool user

# GPU / e2e — still holds under real model inference
python scripts/tools/verify_session_tito_tokenizer.py \
    --hf-checkpoint <hf-id> --tito-model <family> --tito-allowed-append-roles tool user \
    --sglang-reasoning-parser <rp> --sglang-tool-call-parser <tcp> --rollout-num-gpus-per-engine 1
```
