---
title: Rollout Endpoints
description: How Miles talks to SGLang. The /generate endpoint and the OpenAI-format /v1/chat/completions endpoint.
---
Miles supports two ways for a custom rollout function to talk to SGLang. The `/generate` endpoint is the most direct interface and leaves tokenization to your code. The OpenAI-format `/v1/chat/completions` endpoint runs through Miles' session server: an agent exchanges `messages`, while Miles owns TITO tokenization and trajectory assembly across turns.

| | `/generate` | OpenAI `/v1/chat/completions` |
|---|---|---|
| Input | Text or tokens | `messages` list |
| Tokenization | Your code | Miles' TITO session server |
| Session state | Stateless | Session server (`base_url` includes `/sessions/<id>`) |
| Best for | Tool use with custom token handling, benchmarking | Agentic loops, multi-turn dialogue |
| Reference generator | `generate_hub/single_turn.py`, `generate_hub/multi_turn.py` | `generate_hub/agentic_tool_call.py` |

Both entry points are wired up through `--custom-generate-function-path`.

---

## The `/generate` endpoint

### What `generate_hub` is

`miles/rollout/generate_hub/` ships reusable generate functions that conform to the
refactored rollout interface (`GenerateFnInput` / `GenerateFnOutput`). They compose
with custom agents, tool use, or multi-turn logic.

Key modules:

| Path | Purpose |
|---|---|
| `miles/rollout/base_types.py` | `GenerateFnInput` / `GenerateFnOutput` |
| `miles/rollout/inference_rollout/inference_rollout_common.py` | Builds a `GenerateState` and calls the generate function |
| `MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1` | Enables the new path (see `examples/swe-agent-harbor-docker`) |

### Generate function basics

The runtime contract:

1. The rollout engine passes a `GenerateFnInput` containing:
    - `state`: tokenizer, processor, args, sampling defaults.
    - `sample`: the prompt, current tokens, response, status.
    - `sampling_params`: `max_new_tokens`, `temperature`, `top_p`, etc.
2. Your function:
    - Builds a request from the prompt.
    - Executes it against SGLang.
    - Updates the `Sample` with tokens, logprobs, loss mask, status.

Minimal skeleton:

```python
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.types import Sample


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    args = input.args
    sample = input.sample
    sampling_params = input.sampling_params

    # 1) build request from prompt and sampling params
    # 2) call backend
    # 3) update sample.tokens, sample.response, sample.rollout_log_probs,
    #    sample.loss_mask, sample.status

    return GenerateFnOutput(samples=sample)


def _add_arguments(parser):
    parser.add_argument("--your-arg", type=str)


generate.add_arguments = _add_arguments
```

<Tip>

**Custom CLI flags.** `generate.add_arguments = _add_arguments` registers extra CLI flags. They are
parsed into `input.args` and available everywhere in your generator.

</Tip>

Helpers:

- `compute_prompt_ids_from_sample` and `compute_request_payload` from
  `miles/rollout/generate_utils/generate_endpoint_utils.py` build `/generate` requests.
- A generate function can set `GenerateFnOutput.samples` to a `Sample` or `list[Sample]`.

### Reference generators

- **`single_turn.py`**: single-turn generation via `/generate`. Text or multimodal prompts.
- **`multi_turn.py`**: multi-turn tool calling via `/generate`. Adds CLI flags
  `--generate-max-turns`, `--generate-tool-specs-path`, `--generate-tool-call-parser`,
  `--generate-execute-tool-function-path`.
- **`benchmarkers.py`**: forces random output sequence length for benchmarking.

---

## The OpenAI chat endpoint

### Minimal `run_agent`

A `run_agent` receives a session-scoped `base_url`. Send OpenAI-format chat requests
to `base_url/v1/chat/completions` and pass the `messages` list as the prompt.

```python
from miles.utils.http_utils import post


async def run_agent(
    base_url: str,
    prompt,
    request_kwargs: dict | None = None,
    metadata: dict | None = None,
    **kwargs,
) -> dict | None:
    payload = {"model": "default", "messages": prompt, **(request_kwargs or {})}
    await post(f"{base_url}/v1/chat/completions", payload)
    return None
```

<Tip>

**What's already handled.**
- `base_url` already includes `/sessions/<id>`. Don't append it manually.
- `request_kwargs` already contains sampling defaults from
  `agentic_tool_call.build_chat_request_kwargs`.
- `max_new_tokens` from Miles's rollout params is mapped to OpenAI's `max_tokens`
  before the request is sent.
- `metadata` contains the input sample's metadata plus session identifiers and, when configured, `max_seq_len`; pass through the fields your external environment needs.
- The session server replaces any client `input_ids` and forces the response metadata required for TITO. The agent does not manage token IDs or response-logprob flags.
- For structured parsing, use SGLang's `ChatCompletionRequest`-compatible
  format, a superset of OpenAI plus SGLang extras.

</Tip>

### OpenAI chat messages

Standard OpenAI format:

```json
{
  "model": "default",
  "messages": [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user",   "content": "Answer with one word: 2+2?"}
  ]
}
```

<Warning>

**Leave TITO fields to the session server.** Do not send `input_ids` or set `logprob_start_len=0`. Miles constructs the exact prompt IDs, forces `logprobs=True` and `return_meta_info=True`, and records the output token IDs and logprobs. Setting `logprob_start_len=0` makes SGLang score the whole prompt, destroys the prefix-cache benefit, and hurts performance.

</Warning>

### Quickstart

Generator entry point:

- `miles/rollout/generate_hub/agentic_tool_call.py`: OpenAI-format agent loop via the TITO session server.

Example:

- [`examples/swe-agent-harbor-docker`](https://github.com/radixark/miles/tree/main/examples/swe-agent-harbor-docker):
  multi-turn agentic SWE agent on the session-server TITO path, with ready-to-run launchers.

Minimal wiring for a Qwen3 agent function:

```bash
export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1

CUSTOM_ARGS=(
   --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
   --custom-agent-function-path    my_agent.run
   --use-session-server
   --hf-checkpoint                 Qwen/Qwen3-4B
   --tito-model                    qwen3
)
```

Add the reasoning and tool-call parser flags required by your model and agent. For a production launcher with those settings, reward wiring, and environment integration, use the SWE-agent example above.

<Warning>

**Don't apply the chat template to prompt data.** For the OpenAI path, do **not** pass `--apply-chat-template`: `Sample.prompt` must remain a `messages` list. The Miles session server renders the first turn and incrementally appends later turns with the selected `--tito-model` implementation.

</Warning>

<Warning>

**Session server v2 output is a `list[Sample]`.** With `--use-session-server v2`, `agentic_tool_call.generate` returns one sample for each selected tree leaf. The v1 session server returns one scalar `Sample`.

A custom reward model (`--custom-rm-path`) receives the v2 samples in batch form. `--group-rm`, `--partial-rollout`, and `--recompute-logprobs-via-prefill` are not supported with this v2 agentic output and are rejected explicitly.

</Warning>

### Optional teardown: the `abort` hook

The module named by `--custom-agent-function-path` may expose an optional `abort`
function alongside the agent entry point:

```python
async def abort(args) -> None:
    ...  # tell this agent's backend to cancel its in-flight work
```

Miles calls it during **oversampling abort**. When dynamic sampling has collected
enough groups, the rollout aborts in-flight SGLang generation (see
[Async / partial rollout](/user-guide/cli-reference#async--partial-rollout)).
An external agent loop doesn't observe that abort on its own — it keeps issuing
fresh completion requests until it hits its own `max_seq_len` or timeout. If your
agent drives an external backend (e.g. a sandbox/agent server), define `abort` to
tell that backend to tear down the trials tied to this rollout.

The hook is **entirely optional and safe to omit**:

- If the module defines no `abort`, nothing is called — existing plugins are
  unaffected and their in-flight generations simply drain as before.
- It only fires when `--custom-agent-function-path` is set, so non-agentic runs
  never invoke it.

See [`swe_agent_function.abort`](https://github.com/radixark/miles/blob/main/examples/swe-agent-harbor-docker/swe_agent_function.py)
for a reference implementation that flushes the Harbor agent server.

### Customizing the wrapper

[`agentic_tool_call.generate`](https://github.com/radixark/miles/blob/main/miles/rollout/generate_hub/agentic_tool_call.py)
is a thin wrapper around the custom agent. It:

1. Creates a session on one Miles session-server instance and builds a session-scoped `base_url`.
2. Calls the custom agent (from `--custom-agent-function-path`) to send one or more
   chat requests.
3. Collects server-assembled `Sample` objects via `OpenAIEndpointTracer.collect_samples`
   (the session server converts records into samples, truncates and merges on the
   owning instance; records never leave the server).

For broader customization beyond the OpenAI wrapper, see the `/generate` path above.

### TITO (token-in / token-out)

The agent sends the full OpenAI `messages` history on every turn, but Miles does not re-tokenize the full history. On the first turn, the session server renders the selected `--tito-model` template into `input_ids`. After a successful completion, it checkpoints those exact prompt IDs plus the output token IDs returned in SGLang's `meta_info.output_token_logprobs`.

On each later turn, the session server reuses a stored checkpoint, tokenizes only the appended suffix, joins it to the checkpoint, and sends the resulting `input_ids` to SGLang. History selection depends on the version:

- **v1:** requires an append-only extension, except for a one-assistant-step retry rollback.
- **v2:** attaches the request to its deepest complete matching checkpoint and creates a branch from any unmatched suffix; it never deletes an existing branch.

Both versions enforce the appended-role surface registered by `--tito-model`. The session server forces `logprobs=True` and `return_meta_info=True`, so the agent does not request prompt IDs or manage tokens itself.

During collection, [`merge.py`](https://github.com/radixark/miles/blob/main/miles/rollout/session/samples/merge.py) aligns each turn's output tokens and logprobs against the accumulated TITO sequence, trims model-specific boundary tokens, and assembles the training sample. The [Agentic Rollout (TITO)](/user-guide/agentic-chat-template) guide lists the verified model families and template checks.

### Common pitfalls

| Pitfall | Fix |
|---|---|
| Backend response lacks `meta_info.output_token_logprobs` | Use the supported SGLang build; the session server already forces `logprobs=True` and `return_meta_info=True`. |
| Prefix cache hit rate drops to 0 | Remove `logprob_start_len=0`. |
| v1 rejects changed history | Keep messages append-only; only a one-assistant-step retry rollback is supported. Use v2 when one session must preserve multiple lineages. |
| v2 creates an unexpected branch | Replay every message in the intended parent path exactly; v2 attaches at the deepest complete match. |
| Appended-role validation error | Pick the matching `--tito-model` and use only roles supported by that family's fixed template. |
| Agent sends its own `input_ids` | Remove them; the session server owns prompt tokenization. |
| Custom agent hitting the wrong URL | `base_url` already has `/sessions/<id>`. Don't append it. |

---

## Next

- [Customization](/user-guide/customization): the full catalog of `--*-path` hooks.
- [Agentic Rollout (TITO)](/user-guide/agentic-chat-template): verifying that a template is
  append-only across turns.
- [Multi-agent example](https://github.com/radixark/miles/tree/main/examples/experimental/multi_agent):
  full agentic walkthrough.
