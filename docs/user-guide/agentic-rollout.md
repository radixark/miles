---
title: Agentic Rollout (TITO)
description: Configure an OpenAI-compatible agent loop with Token-In-Token-Out trajectory assembly.
---

Multi-turn agentic rollout in Miles runs through the Token-In-Token-Out (TITO)
session server. Your agent exchanges OpenAI-compatible chat messages, while Miles
preserves the exact token IDs, logprobs, and routed experts produced during
inference and assembles them into training samples. For the design rationale, see
[No Token Left Behind](https://lmsys.org/blog/2026-05-13-no-token-left-behind/).

This page owns the agentic path: wrapper setup, the custom agent contract,
session behavior, token ownership, model-family selection, and verification.
Use [Rollout Endpoints](/user-guide/rollout-endpoints) for the lower-level,
stateless `/generate` interface.

## Configure the wrapper

Select `agentic_tool_call.generate` as the custom generate function. The wrapper
registers `--custom-agent-function-path` and `--max-seq-len`, creates a TITO
session for each rollout, invokes your agent, and collects the resulting samples.

```bash
export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1

AGENTIC_ARGS=(
   --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
   --custom-agent-function-path    my_agent.run
   --use-session-server
   --hf-checkpoint                 Qwen/Qwen3-4B
   --tito-model                    qwen3
)
```

A bare `--use-session-server`, or `--use-session-server v1`, selects the linear
v1 server. Use `--use-session-server v2` when one session must retain multiple
trajectory branches.

<Warning>

**Do not apply the chat template to prompt data.** Do not pass
`--apply-chat-template`: `Sample.prompt` must remain a `messages` list. The
session server renders the first turn and incrementally appends later turns with
the selected `--tito-model` implementation.

</Warning>

## Write the agent loop

Use `--custom-agent-function-path` to name an async function with this contract:

```python
async def run_agent(
    base_url: str,
    prompt,
    request_kwargs: dict,
    metadata: dict,
    **kwargs,
) -> dict | None:
    ...
```

Send OpenAI-compatible chat requests to the session-scoped endpoint:

```python
from miles.utils.http_utils import post


async def run_agent(base_url, prompt, request_kwargs, metadata, **kwargs):
    payload = {"model": "default", "messages": prompt, **request_kwargs}
    await post(f"{base_url}/v1/chat/completions", payload)
    return None
```

- `base_url` already includes `/sessions/<id>`; do not append the session path.
- `prompt` is the input sample's OpenAI `messages` list.
- `request_kwargs` contains the rollout sampling settings in
  `ChatCompletionRequest`-compatible form. For example, Miles maps
  `max_new_tokens` to `max_tokens`.
- `metadata` contains the sample metadata, session identifiers, and configured
  `max_seq_len`. Forward only the fields your environment needs.
- Return a dictionary to merge rewards, reports, or metrics into each output
  sample's metadata, or return `None` when there is nothing to add.

For structured parsing, the payload may use SGLang's
`ChatCompletionRequest`-compatible fields, which extend the OpenAI format.

## Leave token ownership to Miles

Send the full `messages` history on every turn. On the first request, the
session server renders the selected template into `input_ids`. After a
successful completion, it checkpoints those prompt IDs together with the output
token IDs and logprobs returned by SGLang.

On later requests, the server reuses the deepest applicable checkpoint,
tokenizes only the appended suffix, and sends the joined `input_ids` to SGLang.
During collection, Miles aligns the turn outputs against the accumulated TITO
sequence, trims model-specific boundary tokens, and builds the training sample.

<Warning>

**Do not send TITO control fields.** The session server replaces client
`input_ids` and forces `logprobs=True`, `return_meta_info=True`, and the response
metadata needed for TITO. Do not set `logprob_start_len=0`; scoring the entire
prompt defeats prefix caching and hurts performance.

</Warning>

## Choose the session behavior

History handling depends on the selected server version:

- **v1 is linear.** Each request must extend the previous messages at the tail.
  Retrying the latest turn may roll back one assistant checkpoint, including to
  an empty session when retrying the first turn. Earlier divergence or a larger
  rollback is rejected.
- **v2 is an append-only tree.** A request attaches to the deepest checkpoint
  whose complete message path prefixes the request. Any unmatched suffix creates
  a branch, and existing branches are never deleted. A path whose last generation
  ended with `finish_reason=length` cannot be extended.
- **Appended roles follow the template.** After a checkpoint, the selected
  model's fixed template determines which roles may be appended.

The v1 wrapper returns one `Sample`. The v2 wrapper returns a `list[Sample]`, one
for each selected tree leaf. Consequently, v2 rejects `--group-rm`,
`--partial-rollout`, and `--recompute-logprobs-via-prefill`. Both versions reject
`--pause-generation-mode=abort`.

Set `--max-seq-len` to cap the combined prompt, model output, and environment
response tokens in each assembled sample. Miles also includes this value in the
metadata passed to your agent so an external environment can stop early.

## Optional teardown hook

The module named by `--custom-agent-function-path` may expose an `abort` function
alongside the agent entry point:

```python
async def abort(args) -> None:
    ...  # cancel this agent's in-flight external work
```

Miles calls this hook during oversampling abort after it stops in-flight SGLang
generation. Use it when the agent drives an external sandbox or agent server that
would otherwise keep issuing completion requests until its own length limit or
timeout. The hook is optional; modules without it continue to work.

See [`swe_agent_function.abort`](https://github.com/radixark/miles/blob/main/examples/swe-agent-harbor-docker/swe_agent_function.py)
for an implementation that flushes the Harbor agent server.

## Pick your `--tito-model`

There is no auto-detection. Pick the family matching your model. Each named
family resolves a maintainer-verified `FIXED_TEMPLATE` registration from
`--tito-model` alone. The registration owns the bundled Jinja or
HuggingFace-native template, fixed template arguments, and the bundled SGLang
reasoning and tool-call parsers.

A named family rejects `--chat-template-path` overrides and conflicting fixed
arguments. Use `--tito-model default` for a custom or checkpoint-native renderer,
but treat it as best-effort until it passes the checks below.

| Your model | `--tito-model` |
|---|---|
| Qwen3 | `qwen3` |
| Qwen3.5 | `qwen35` |
| Qwen3-Thinking-2507 / Qwen3-Next | `qwennext` |
| GLM-4.7 / 5 / 5.1 / 5.2 | `glm47` |
| NVIDIA Nemotron 3 Nano / Super / Ultra | `nemotron3` |
| Kimi K2.5 / K2.6 | `kimi25` / `kimi26` |
| MiniMax M2.5 / M2.7 | `minimax_m25` / `minimax_m27` |
| DeepSeek-V3.2 / V4 | `deepseekv32` / `deepseekv4` |
| Inkling / Inkling-Small | `inkling` |
| Unregistered model or custom template (best-effort) | `default` |

More model families and verification history live in
[issue #712](https://github.com/radixark/miles/issues/712).

## Verify a new model family

To add a named family, register its `TITOTokenizer` and `FIXED_TEMPLATE` in
[`tito_tokenizer.py`](https://github.com/radixark/miles/blob/main/miles/utils/chat_template_utils/tito_tokenizer.py),
then run both checks. Either failure blocks support.

```bash
# CPU / fast: rendered token sequences remain append-only
python scripts/tools/verify_chat_template.py \
    --model <hf-id> --tito-model <family>

# GPU / end to end: the invariant holds under real model inference
python scripts/tools/verify_session_tito_tokenizer.py \
    --hf-checkpoint <hf-id> --tito-model <family> \
    --sglang-reasoning-parser <rp> --sglang-tool-call-parser <tcp> \
    --rollout-num-gpus-per-engine 1
```

## Troubleshooting

| Symptom | Check |
|---|---|
| Backend response lacks `meta_info.output_token_logprobs` | Use the supported SGLang build. Miles already forces `logprobs=True` and `return_meta_info=True`. |
| Prefix-cache hit rate drops to zero | Remove `logprob_start_len=0`. |
| v1 rejects changed history | Keep messages append-only or use v2 when the session must preserve multiple lineages. |
| v2 creates an unexpected branch | Replay every message in the intended parent path exactly. |
| Appended-role validation fails | Select the matching `--tito-model` and use only roles supported by its fixed template. |
| The agent hits the wrong URL | Use the supplied `base_url`; it already contains `/sessions/<id>`. |

## Complete example

[`examples/swe-agent-harbor-docker`](https://github.com/radixark/miles/tree/main/examples/swe-agent-harbor-docker)
wires a multi-turn SWE agent, TITO session server, model-family registration,
reward, length limit, and environment teardown into production launchers.
