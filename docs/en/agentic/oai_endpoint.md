# OAI Endpoint Usage

This document explains how to use the OpenAI-format chat endpoint through Miles
Router sessions. For the `/generate` endpoint, see
`docs/en/get_started/gen_endpoint.md`.

## 1. Minimal `run_agent` loop

Your `run_agent` receives a session-scoped `base_url`. Send OpenAI-format chat
requests to `base_url/v1/chat/completions` and pass the `messages` list as the
prompt.

Minimal custom agent example:

```python
from miles.utils.http_utils import post

async def run_agent(base_url: str, prompt, request_kwargs: dict | None = None) -> None:
    payload = {"model": "default", "messages": prompt, **(request_kwargs or {})}
    await post(f"{base_url}/v1/chat/completions", payload)
```

Notes for `run_agent`:

- `base_url` already includes the session path (e.g. `/sessions/<id>`), so you
  should not manually add the session id. Just append the OpenAI route.
- `request_kwargs` already contains the default sampling settings from
  `agentic_tool_call.build_chat_request_kwargs`, so you can directly expand it
  into the chat request payload.
- If you pass rollout sampling params, `max_new_tokens` will be mapped to the
  OpenAI `max_tokens` field before the request is sent.
- If you need structured parsing payloads, use SGLang's
  `ChatCompletionRequest`-compatible format. It is compatible with native OpenAI
  fields, plus extra SGLang parameters.

## 2. OpenAI chat messages and the basic request

The OpenAI-format chat API uses a list of `messages`, each with a `role` and
`content`.

Minimal request shape:

```json
{
  "model": "default",
  "messages": [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "Answer with one word: 2+2?"}
  ]
}
```

You can pass any OpenAI-compatible parameters in the payload, or any
SGLang-compatible `ChatCompletionRequest` parameters. Note: with
`--use-session-server`, the session middleware sets the token-tracking fields TITO needs.
Do **not** set `logprob_start_len=0` — it would disable SGLang's prefix
cache.

## 3. Quickstart index

If you just want something runnable, start here.

Generator entry point:

- `miles/rollout/generate_hub/agentic_tool_call.py` — OpenAI-format agent
  loop via router sessions.

OpenAI-format examples that use `agentic_tool_call.generate`:

- Single-turn (DAPO math):
  - `examples/openai_format/dapo_math.py` — custom agent.
  - `examples/openai_format/run-qwen3-4B.sh` — launcher.
- Multi-turn (SWE agent):
  - `examples/experimental/swe-agent-v2/run.py` — drives a multi-turn
    tool-calling loop with `--tito-model glm47 --tito-allowed-append-roles tool user`.

Key flags for an OpenAI-format agentic run:

| Flag | Description |
| :--- | :--- |
| `--custom-generate-function-path` | Set to `miles.rollout.generate_hub.agentic_tool_call.generate`. The OpenAI wrapper that creates the session and collects records. |
| `--custom-agent-function-path` | Path to your `run_agent` function. It receives `base_url` and `request_kwargs` and sends chat requests. |
| `--use-session-server` | Enables the session-server middleware that forces the SGLang flags TITO needs and tracks token prefixes across turns. Required for TITO. |
| `--tito-model` | TITO tokenizer family (`qwen3`, `qwen35`, `qwennext`, `glm47`, ...). Use `default` to keep the model's HF-native chat template untouched. |
| `--tito-allowed-append-roles` | Roles the session may append after assistant turns. Default `tool`; add `user` / `system` if your conversation pattern needs them. |

For the list of supported `--tito-model` families and the role surfaces
pre-wired for each, see
[`miles/utils/chat_template_utils/tito_tokenizer.py`](../../../miles/utils/chat_template_utils/tito_tokenizer.py)
or [issue #712](https://github.com/radixark/miles/issues/712).

Customize like:

```
CUSTOM_ARGS=(
   --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
   --custom-agent-function-path examples.openai_format.dapo_math.run_agent
)
```

For OpenAI format, do not add `--apply-chat-template`; the
prompt must remain a `messages` list.

## 4. Further customization (OpenAI wrapper generate function)

For OpenAI-format rollout, the key generate function is
`miles/rollout/generate_hub/agentic_tool_call.generate`. It is a thin wrapper
around your custom agent:

1. Create a session on Miles Router and build a session-scoped `base_url`.
2. Call the custom agent (from `--custom-agent-function-path`) to send one or
   more chat requests to `base_url/v1/chat/completions`, typically using
   `prompt` and `request_kwargs`.
3. Collect session records via `OpenAIEndpointTracer`.
4. Convert records into `Sample` objects with
   `compute_samples_from_openai_records`.

If you want general generate-function customization beyond the OpenAI wrapper,
see `docs/en/get_started/gen_endpoint.md`.

## 5. TITO (token-in token-out)

TITO is the layer that gives miles per-turn token ids and logprobs across a
multi-turn session, without re-tokenizing the full conversation on every
request. It requires `--use-session-server`; with that on, miles handles
three things on your behalf:

- Hardcodes the SGLang flags TITO needs on every chat request
  (`logprobs=True`, `return_meta_info=True`, `no_stop_trim=False`); these are
  set by the middleware in `miles/rollout/session/sessions.py` and override any
  agent-passed values.
- Reuses the token prefix from previous turns by injecting Miles-owned `input_ids` on every proxied chat request. 
- Accumulates per-turn records into the `Sample` you receive at the end of
  the session, with `tokens` and `rollout_log_probs` already populated.

You do not extract token ids from the response yourself.

### Configuring TITO for your model and conversation

`--use-session-server` turns TITO on, but you also need to set two more
flags so TITO renders with the right chat template and accepts your
conversation's roles.

- `--tito-model` (default `default`) selects the TITO tokenizer family. Set
  it to your model's family — `qwen3`, `qwen35`, `qwennext`, `glm47`, ... —
  so miles auto-resolves a bundled fixed chat template verified against
  TITO's append-only invariants. The `default` value skips auto-resolution
  and uses the model's HF-native chat template; use it only if that native
  template already satisfies append-only.
- `--tito-allowed-append-roles` (default `tool`) restricts which roles your
  `run_agent` may append after the first assistant message. Extend it to
  match your conversation pattern; see "Session server message constraints"
  below for valid values and typical role sets. Adding `user` or `system`
  typically selects a different bundled template (or different kwargs)
  within the same family, since broader role surfaces need stricter
  rendering rules to keep append-only valid.

### Common pitfalls

- Do **not** set `logprob_start_len=0` — it forces SGLang to compute
  logprobs for every prompt token, which destroys the prefix cache and
  hurts performance.

## 6. Session server message constraints

The session server enforces two invariants on every request's `messages`.
Violations raise `MessageValidationError`.

- **Append-only.** Each new request's `messages` must contain the previous
  turn's full `messages` as an exact prefix (same roles, same
  template-relevant content). You can extend at the tail; you cannot edit,
  reorder, or drop earlier messages. This is how TITO can reuse tokens
  across turns: the stored token prefix is only valid if the client
  confirms the same message prefix on every request.

- **Allowed append roles.** After the first assistant message, every newly
  appended message must have a role in `--tito-allowed-append-roles`
  (default `{tool}`; valid values: `tool`, `user`, `system`). Typical usage:

  - Pure tool-calling agent — `tool` (default).
  - Multi-turn chat with mid-conversation user follow-ups — add `user`.
  - Mid-conversation system reminders — also add `system`.

Both invariants depend on the chat template's rendering behavior under the
chosen role surface. Adding miles support for a new model family therefore
involves both updating the chat template and adapting the TITO tokenizer
subclass for that family.
