---
title: Agentic Rollout (TITO)
description: How to turn on and verify Token-In-Token-Out (TITO) for multi-turn agentic rollout.
---

# Agentic Rollout (TITO)

Multi-turn agentic rollout in Miles runs on **TITO** (Token-In-Token-Out): each turn's token sequence is a bit-perfect prefix of the next, so the trainer sees exactly the tokens the engine produced — no re-tokenization, no drift. The *why* is in the blog ([No Token Left Behind](https://lmsys.org/blog/2026-05-13-no-token-left-behind/)); this page is *how*.

Your harness only ever sends and receives **OpenAI chat messages**, never tokens. Miles keeps the per-trajectory append-only token buffer (ids + logprobs + routed experts) internally and ships it straight to training.

## Prerequisites

Your rollout loop must keep two invariants, or TITO is rejected at runtime:

- **Append-only messages.** Each turn = previous messages + new ones on the tail; past turns are never edited. The only exception is retrying the latest turn — a single-step rollback to the last assistant checkpoint, or to an empty session when the retried turn is the first one. Diverging earlier, or rolling back more than one turn, is rejected.
- **Appended roles follow the chat template.** After the first assistant message, the selected model's chat template determines which roles may be appended; users do not configure this separately.

## Pick your `--tito-model`

No auto-detection — pick the family matching your model. For every family, Miles resolves one `FIXED_TEMPLATE` registration from `--tito-model` alone. The registration owns the bundled Jinja template (or HuggingFace-native template) and fixed kwargs. A non-default family rejects `--chat-template-path` overrides and conflicting fixed kwargs; use `--tito-model default` for a custom renderer.

| Your model | `--tito-model` |
|---|---|
| Qwen3 | `qwen3` |
| Qwen3.5 | `qwen35` |
| Qwen3-Next | `qwennext` |
| GLM-4.7 / GLM-5 | `glm47` |
| NVIDIA Nemotron 3 Super / Ultra | `nemotron3` |
| Kimi K2.5 / K2.6 | `kimi25` / `kimi26` |
| MiniMax M2.5 / M2.7 | `minimax_m25` / `minimax_m27` |
| DeepSeek-V3.2 / V4 | `deepseekv32` / `deepseekv4` |
| anything else | `default` |

More models and verification history live in [issue #712](https://github.com/radixark/miles/issues/712).

## Turn it on

```bash
ROLLOUT_ARGS+=(
   --use-session-server          # entry point for TITO session tracking
   --hf-checkpoint Qwen/Qwen3-4B
   --tito-model qwen3
)
```

## Example

A full multi-turn agentic setup on the session-server TITO path lives in [`examples/swe-agent-harbor-docker`](https://github.com/radixark/miles/tree/main/examples/swe-agent-harbor-docker): its launchers wire `--use-session-server` + `--tito-model glm47` against a real SWE agent.

## Add a new model

Models in the table are verified by Miles maintainers. To support a new model, register its `TITOTokenizer` and `FIXED_TEMPLATE` in [`tito_tokenizer.py`](https://github.com/radixark/miles/blob/main/miles/utils/chat_template_utils/tito_tokenizer.py), then run both checks below; either failure blocks support.

```bash
# CPU / fast — rendered token sequence is append-only
python scripts/tools/verify_chat_template.py \
    --model <hf-id> --tito-model <family>

# GPU / e2e — still holds under real model inference
python scripts/tools/verify_session_tito_tokenizer.py \
    --hf-checkpoint <hf-id> --tito-model <family> \
    --sglang-reasoning-parser <rp> --sglang-tool-call-parser <tcp> --rollout-num-gpus-per-engine 1
```
