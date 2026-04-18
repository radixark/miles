---
title: Rollout Endpoints
description: How Miles talks to SGLang — the /generate endpoint and the OpenAI-format /v1/chat/completions endpoint.
---

# Rollout Endpoints

Miles supports two ways for a custom rollout function to talk to SGLang. Use the
**`/generate` endpoint** when you control tokenisation and want the most direct
interface. Use the **OpenAI chat endpoint** when you're building an agent loop that
benefits from server-side session state.

| | `/generate` | OpenAI `/v1/chat/completions` |
|---|---|---|
| Input | Text or tokens | `messages` list |
| Tokenisation | Your code | SGLang |
| Session state | Stateless | Router sessions (base_url includes `/sessions/<id>`) |
| Best for | Tool use with custom token handling, benchmarking | Agentic loops, multi-turn dialogue |
| Reference generator | `generate_hub/single_turn.py`, `generate_hub/multi_turn.py` | `generate_hub/agentic_tool_call.py` |

Both entry points are wired up through `--custom-generate-function-path`.

---

## The `/generate` endpoint

### What `generate_hub` is

`miles/rollout/generate_hub/` ships reusable generate functions that conform to the
refactored rollout interface (`GenerateFnInput` / `GenerateFnOutput`). They're designed
to be composed with custom agents, tool use, or multi-turn logic.

Key modules:

| Path | Purpose |
|---|---|
| `miles/rollout/base_types.py` | `GenerateFnInput` / `GenerateFnOutput` |
| `miles/rollout/inference_rollout/inference_rollout_common.py` | Builds a `GenerateState` and calls the generate function |
| `MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1` | Enables the new path (see `examples/openai_format/*.sh`) |

### Generate function basics

The runtime contract:

1. The rollout engine passes a `GenerateFnInput` containing:
    - `state` — tokenizer, processor, args, sampling defaults.
    - `sample` — the prompt, current tokens, response, status.
    - `sampling_params` — `max_new_tokens`, `temperature`, `top_p`, etc.
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

!!! tip "Custom CLI flags"
    `generate.add_arguments = _add_arguments` is the hook for custom CLI flags. They're
    parsed into `input.args` and available everywhere in your generator.

Helpers:

- `compute_prompt_ids_from_sample` and `compute_request_payload` from
  `miles/rollout/generate_utils/generate_endpoint_utils.py` build `/generate` requests.
- For multi-sample outputs, set `--generate-multi-samples` and return a list.

### Reference generators

- **`single_turn.py`** — Single-turn generation via `/generate`. Text or multimodal prompts.
- **`multi_turn.py`** — Multi-turn tool calling via `/generate`. CLI flags:
  `--generate-max-turns`, `--generate-tool-specs-path`, `--generate-tool-call-parser`,
  `--generate-execute-tool-function-path`, `--generate-multi-samples`.
- **`benchmarkers.py`** — Forces random output sequence length for benchmarking.

### Radix-tree middleware (full TITO for `/generate`)

For **token-in / token-out** caching on `/generate`, turn on the radix-tree middleware.
It's unrelated to the OpenAI session middleware and only affects the `/generate` and
`/retrieve_from_text` routes.

What it does:

- Caches token ids and logprobs by prompt text in a radix tree.
- Lets `/generate` requests include `input_tokens` — skipping re-tokenisation.
- Enables `update_sample_from_response` to fetch tokens via `/retrieve_from_text`
  during training.

Enable it:

```bash
--use-miles-router
--miles-router-middleware-paths miles.router.middleware_hub.radix_tree_middleware.RadixTreeMiddleware
```

Make sure `--sglang-router-ip` and `--sglang-router-port` point at Miles Router so
`/retrieve_from_text` is reachable during rollout.

See [Miles Router (R3)](../advanced/miles-router.md) for the broader router picture.

---

## The OpenAI chat endpoint

### Minimal `run_agent`

Your `run_agent` receives a session-scoped `base_url`. Send OpenAI-format chat requests
to `base_url/v1/chat/completions` and pass the `messages` list as the prompt.

```python
from miles.utils.http_utils import post


async def run_agent(base_url: str, prompt, request_kwargs: dict | None = None) -> None:
    payload = {"model": "default", "messages": prompt, **(request_kwargs or {})}
    await post(f"{base_url}/v1/chat/completions", payload)
```

!!! tip "What's already handled"
    - `base_url` already includes `/sessions/<id>` — don't add it manually.
    - `request_kwargs` already contains sampling defaults from
      `agentic_tool_call.build_chat_request_kwargs`.
    - `max_new_tokens` from Miles's rollout params is mapped to OpenAI's `max_tokens`
      before the request is sent.
    - For structured parsing, use SGLang's `ChatCompletionRequest`-compatible format —
      a superset of OpenAI plus SGLang extras.

### OpenAI chat messages

Standard OpenAI format:

```json
{
  "model": "default",
  "messages": [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user",   "content": "Answer with one word: 2+2?"}
  ],
  "logprobs": true,
  "return_prompt_token_ids": true
}
```

!!! warning "Leave `logprob_start_len` alone"
    `logprobs=True` and `return_prompt_token_ids=True` are set by default — they're
    what enables TITO. **Do not** set `logprob_start_len=0`: it forces SGLang to
    compute logprobs for every prompt token, destroys the prefix cache, and hurts
    performance. `return_prompt_token_ids=True` returns prompt token ids at zero
    cost with full caching.

### Quickstart

Generator entry point:

- `miles/rollout/generate_hub/agentic_tool_call.py` — OpenAI-format agent loop via
  router sessions.

Examples:

- [`examples/openai_format/dapo_math.py`](https://github.com/radixark/miles/blob/main/examples/openai_format/dapo_math.py) —
  single-turn OpenAI-format agent (DAPO math).
- [`examples/openai_format/run-qwen3-4B.sh`](https://github.com/radixark/miles/blob/main/examples/openai_format/run-qwen3-4B.sh) — launcher.

Wire-up:

```bash
CUSTOM_ARGS=(
   --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
   --custom-agent-function-path    examples.openai_format.dapo_math.run_agent
)
```

!!! warning "Don't apply chat template"
    For OpenAI format, do **not** pass `--apply-chat-template`. The prompt must remain
    a `messages` list — SGLang handles templating server-side.

### Customising the wrapper

[`agentic_tool_call.generate`](https://github.com/radixark/miles/blob/main/miles/rollout/generate_hub/agentic_tool_call.py)
is a thin wrapper around your custom agent. It:

1. Creates a session on Miles Router and builds a session-scoped `base_url`.
2. Calls the custom agent (from `--custom-agent-function-path`) to send one or more
   chat requests.
3. Collects session records via `OpenAIEndpointTracer`.
4. Converts records into `Sample` objects via `compute_samples_from_openai_records`.

For broader customisation (beyond the OpenAI wrapper), see the `/generate` path above.

### TITO (token-in / token-out)

TITO needs two things from every SGLang response:

1. **Prompt token ids** — extracted from `response.choices[0].prompt_token_ids`.
   Returned when the request sets `return_prompt_token_ids=True`.
2. **Output token ids + logprobs** — from `response.choices[0].logprobs.content[*]`
   (`token_id`, `logprob`). Returned when `logprobs=True`.

By default, `build_chat_request_kwargs` sets **both** flags. The session middleware
forwards raw `messages` to SGLang, which tokenises the prompt and returns the response.
`_compute_sample_from_openai_record` in
[`openai_endpoint_utils.py`](https://github.com/radixark/miles/blob/main/miles/rollout/generate_utils/openai_endpoint_utils.py)
extracts prompt + output ids from the response and concatenates them into
`sample.tokens`. You don't need to provide `input_ids` yourself.

Multi-turn samples can be saved within a single session, but tokens are **not**
inherited across turns — each request is tokenised independently.

### Common pitfalls

| Pitfall | Fix |
|---|---|
| Missing logprobs / prompt token ids | Ensure `logprobs=True` and `return_prompt_token_ids=True`. |
| Prefix cache hit rate drops to 0 | Remove `logprob_start_len=0`. |
| Tokenisation drift across turns | Expected — tokens aren't inherited. |
| Custom agent hitting the wrong URL | `base_url` already has `/sessions/<id>` — don't add it. |

---

## Next

- [Customization](customization.md) — the full catalogue of `--*-path` hooks.
- [Agentic Chat Templates](agentic-chat-template.md) — verifying your template is
  append-only across turns.
- [Multi-agent example](../examples/multi-agent.md) — full agentic walkthrough.
