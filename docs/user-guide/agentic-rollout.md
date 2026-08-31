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
Use [Generate Endpoint](/user-guide/generate-endpoint) for the lower-level,
stateless `/generate` interface.

<Warning>

**No VLM support yet.** Currently the TITO session path cannot carry image or video inputs. For vision-language models, use the
[Generate Endpoint](/user-guide/generate-endpoint) path instead.

</Warning>

## Configure the wrapper

Select `agentic_tool_call.generate` as the custom generate function. The wrapper
registers `--custom-agent-function-path` and `--max-seq-len`, creates a TITO
session for each rollout, invokes your agent, and collects the resulting samples.

```bash
AGENTIC_ARGS=(
   --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
   --custom-agent-function-path    my_agent.run
   --use-session-server
   --hf-checkpoint                 Qwen/Qwen3-4B
   --tito-model                    qwen3
)
```

<Warning>

**Do not apply the chat template to prompt data manually.** Do not pass
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


### Optional teardown hook

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

## TITO

### Leave token ownership to Miles

Send the full `messages` history on every turn. On the first request, the
session server renders the selected template into `input_ids`. After a
successful completion, it checkpoints those prompt IDs together with the output
token IDs and logprobs returned by SGLang.

On later requests, the server reuses the deepest applicable checkpoint,
tokenizes only the appended suffix, and sends the joined `input_ids` to SGLang.
During collection, Miles aligns the turn outputs against the accumulated TITO
sequence, trims model-specific boundary tokens, and builds the training sample.

<Warning>

**Do not set TITO control fields.** The session server replaces client
`input_ids` and forces `logprobs=True`, `return_meta_info=True`, and the response
metadata needed for TITO. Do not set `logprob_start_len=0`; scoring the entire
prompt defeats prefix caching and hurts performance.

</Warning>

### Choose the session behavior

History handling depends on the selected server version:

- **v1 is linear.** Each request must extend the previous messages at the tail.
  Retrying the latest turn may roll back one assistant checkpoint, including to
  an empty session when retrying the first turn. Earlier divergence or a larger
  rollback is rejected.
- **v2 (Experimental) is an append-only tree.** A request attaches to the deepest checkpoint
  whose complete message path prefixes the request. Any unmatched suffix creates
  a branch, and existing branches are never deleted. A path whose last generation
  ended with `finish_reason=length` cannot be extended.

Whether a replayed message counts as "the same" as the stored one is decided by
`--session-message-matcher` (default `strict`); see
[Choose replay matching](#choose-replay-matching).

The v1 wrapper returns one `Sample`. The v2 wrapper returns a `list[Sample]`, one for each selected tree leaf. Both versions reject `--partial-rollout`. With R3 replay, colocated `abort` and `in_place` request only additional R3 rows, while `retract` returns full R3 data on every turn and emits a warning because the payloads can become very large. Non-colocated session servers cannot use `abort`.

Set `--max-seq-len` to cap the context length. Miles also includes this value in the
metadata passed to your agent so an external environment can stop early.

### Pick your `--tito-model`

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
| Qwen3.6 | `qwen36` |
| Qwen3.8-27B | `qwen38small` |
| Qwen3.8-Flash-Next | `qwen4exp` |
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

### Verify a new model TITO

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

### Choose replay matching

Some agent harnesses do not replay model messages verbatim: they may reserialize tool-call arguments, replace empty `arguments` with `"{}"`, or omit `reasoning_content` on the next request. Under the default matcher those replays count as divergence — v1 rolls back (or rejects), v2 branches a new lineage.

`--session-message-matcher` is process-wide and defaults to `strict`. It accepts a built-in selector or a trusted dotted import path.

| Selector | Behavior |
|---|---|
| `strict` | Preserves the existing comparison of `role`, `content`, `reasoning_content`, and `tool_calls`, including empty-value and tool-call `index` normalization. |
| `loose_tool_call` | Accepts everything `strict` accepts, plus equivalent JSON-object representations of `tool_calls[].function.arguments`. Call IDs, types, function names, order, unknown fields, and `reasoning_content` still have to match. |
| `role_content_only` | Compares only normalized `role` and `content`. **High risk:** different tool-call or reasoning histories can collapse into one session lineage. |
| dotted import path | Loads a trusted synchronous custom matcher; see [Customization](/user-guide/customization#session-message-matcher). |

The matcher only decides whether the message a client replays and the message stored at the same position in the session count as the same one.

- On a mismatch, the existing paths apply: v1 rolls back (or rejects), v2 branches.
- On a match, the stored messages and token snapshot stay authoritative inside the reusable prefix; only the suffix beyond it is tokenized anew from the client input.

Miles does not reconcile tool-call IDs across that boundary: deployments choosing `role_content_only` must themselves keep a stored call ID `A` followed by a replayed tool result referencing `B` protocol-compatible.

## Example

[`examples/swe-agent-harbor-docker`](https://github.com/radixark/miles/tree/main/examples/swe-agent-harbor-docker)
wires a multi-turn SWE agent, TITO session server, model-family registration,
reward, length limit, and environment teardown into production launchers.
