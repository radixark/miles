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
creates a TITO session for each rollout, invokes the function named by
`--custom-agent-function-path`, and collects the resulting samples.

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
  sample's metadata, or return `None` when there is nothing to add. Either way,
  and also when the function raises, the recorded session becomes a sample the
  reward model scores.
- Raise `miles.rollout.agentic.agent_function.InfraAbort(exit_status)` to
  discard the sample instead. Its group is dropped before any dynamic-sampling
  filter runs and counted in
  `rollout/dynamic_filter/drop_group_has_aborted_<exit_status>`. Evaluation
  scores the attempt 0, like any other with no reward, and counts it in
  `eval/<dataset>-none_reward_ratio`.

The function owns one episode and these rules:

- One call is one episode. Miles opens the session before the call and collects
  its samples once the call returns.
- The returned dictionary is the episode's verdict, merged as above.
- Raise `InfraAbort` only on signals the policy cannot write, such as the
  harness's own result event or the session's count of backend failures. An exit
  code alone does not count, and neither does error text that an agent with a
  shell could print. The next section says which failures qualify.
- A cancelled call stops and cleans up: free the sandbox, then let
  `asyncio.CancelledError` propagate. Every mode delivers the cancel, and a
  rollout abort sends one (see below). A harness that cannot honour it runs
  under `subproc`, whose SIGTERM and backstop kill end it. An adapter that
  spawns a CLI also stops the CLI's descendants, since a tool can sit in its own
  process group.
- Run blocking code with `asyncio.to_thread`, or the event loop cannot deliver
  the cancel while it runs.
- Hold a wall-clock cap, such as `asyncio.timeout`, around the episode, because
  a harness's retry loop need not end by itself. A timeout after the agent
  started is the policy's: return `reward: 0`. Give a CLI its stdin from
  `/dev/null`, or it can wait on it forever.

For structured parsing, the payload may use SGLang's
`ChatCompletionRequest`-compatible fields, which extend the OpenAI format.

### Which failures to discard

A discarded sample contributes no gradient, so the policy is never penalized for
whatever led to it. Discard only what the policy cannot bring about; anything it
can, it would learn to trigger to escape a low reward.

| Outcome | Can the policy cause it? | Do |
|---|---|---|
| The sandbox was never ready, or the harness could not be set up, before the agent started | No | raise `InfraAbort` |
| The model service or the environment host became unreachable | No | raise `InfraAbort` |
| A wall-clock timeout after the agent started | Yes, by stalling | return `reward: 0` |
| A step or turn limit, or `max_seq_len` | Yes | return `reward: 0` |
| The sandbox was lost mid-episode, or the verifier crashed | Yes, the agent controls the sandbox | return `reward: 0` |

Miles itself discards a session that recorded no model call (`NoModelCalls`),
whose samples could not be collected (`CollectFailed`), or whose last model call
failed at the engine with a 5xx or a transport error (`ModelCallFailed`), whatever
the agent function returned. A failed call the agent retried successfully does
not count, and a 4xx, whether the engine's context-length error or the session
rejecting the request, is the policy's own outcome.


### What the session URL accepts

The session server trains only on what it records, so it refuses what it cannot record instead of passing it through. An agent framework or CLI pointed at `base_url` must fit these rules; where it does not by default, the agent function configures it.

- **Dialects.** `POST /v1/chat/completions` (OpenAI chat) and `POST /v1/messages` (Anthropic Messages) are recorded. `GET /v1/models`, `GET /get_model_info` and `POST /v1/messages/count_tokens` are forwarded to the engine unrecorded, since they generate nothing. Any other path returns HTTP 404 naming these routes, so a client that speaks OpenAI Responses or Gemini `generateContent` needs a conversion to chat.
- **Streaming.** `stream: true` gets the complete reply as one block in that dialect's stream format, ending with its terminal event.
- **Refusals.** Each is a client error, since resending the same request cannot succeed: `n` other than 1 is HTTP 400, as the session records one choice per request; a request the chat template cannot render, such as `system` content given as parts to a template that expects a string, is HTTP 400; on v2, a request to a session that already holds 1,024 tree nodes is HTTP 400.
- **History.** Send the full history on every request; see [Leave token ownership to Miles](#leave-token-ownership-to-miles). Use v2 for anything that is not strictly one linear conversation, such as subagents, parallel conversations, side calls or compaction; v1 is for strictly linear agents only. See [Choose the session behavior](#choose-the-session-behavior).
- **Matcher and picker.** A recipe names its `--session-message-matcher` when its client re-serializes tool-call arguments or drops `reasoning_content` ([Choose replay matching](#choose-replay-matching)), and its `--session-sample-picker-path` when the client makes side calls, such as title requests, that cannot be turned off. A picker is specific to a harness.
- **Sampling.** The rollout owns sampling: pass `request_kwargs` through, and keep the client from setting its own `temperature`, `top_p` or `top_k`. A training request whose temperature differs from the session's, or whose `top_p` or `top_k` is bounded without sampling-support replay, is HTTP 400.
- **Other traffic.** Turn the client's telemetry and tracing off. A model the client uses for anything other than the policy, such as a judge or a memory model, goes to another endpoint, not to `base_url`.


### Where the agent runs

By default (`--custom-agent-function-mode subproc`) every agent call runs in a
child process of its own, started by the rollout process. One episode that blocks
on a slow step, or crashes, then affects only itself; on the rollout's single
event loop it would stall every other episode, long enough under many episodes
for their sandbox requests to time out. The contract changes in three ways:

- The arguments and the return value cross a process boundary, so both must be
  picklable.
- Module-level state lasts one call. A module-level semaphore no longer limits
  anything; use `rollout_semaphore` or `rollout_lock` from
  `miles.rollout.agentic.rollout_limits`, which hold across the rollout's agent
  processes. The number of concurrent calls is already bounded by the rollout's
  batch settings.
- Cleanup the agent leaves in a thread must be a non-daemon thread. The call
  returns without waiting for it, and the process waits up to ten minutes for it
  before exiting.

Cancelling a call cancels the agent inside its process, so cleanup such as closing
a sandbox still runs; an agent still running two minutes later is stopped.

`--custom-agent-function-mode inline` runs the agent on the rollout's event loop,
as before; use it for debugging and for agents that keep state across calls.

`--custom-agent-function-mode worker` runs the same child process on an agent
worker on another host, so the agents' CPU work leaves the rollout host. Start one
with `python -m miles.rollout.agentic.worker --port 11100` and name it with
`--custom-agent-function-worker http://<host>:11100`. The agent function must be
importable on the worker under the same module path, and its arguments and return
value must be JSON-serializable. Each call carries each sandbox provider's key-file
and address variables and the env vars matching the patterns in a `WORKER_ENV`
tuple the agent function's module may expose beside it; credentials are never sent
by value, and paths among them must exist on the worker. A worker that cannot be reached, or that no longer knows the call, discards
the sample (`exit_status` `WorkerUnreachable` or `WorkerLost`). A worker cancels a
call nobody has polled for two minutes. To keep anyone else from submitting calls,
set the same secret on the worker and on the rollout host, in
`MILES_AGENT_WORKER_SECRET` or in a file named by `MILES_AGENT_WORKER_SECRET_FILE`,
as one line of printable ASCII;
the worker then refuses requests without it, and a call it refuses discards the
sample (`exit_status` `WorkerUnauthorized`). Without a secret the worker accepts
any request, so run it on a private network.

A rollout abort, such as the one that ends an oversampled round, cancels this
rollout's calls in every mode and discards their samples (`exit_status`
`Cancelled`), however the agent then ends; an episode whose session opens after
the abort never starts its agent. Under `subproc` the abort does not wait for
the agent process's cleanup, which runs beside the next round to the deadlines
above, and under `worker` it waits only until the worker has the cancel. An
`inline` agent's cleanup runs on the rollout's loop: the abort hook
waits up to 30 seconds for it before running the teardown hook below, and the
rollout then waits for the episode to end, however long its cleanup takes. An
evaluation on its own engines (`--eval-num-gpus`) keeps its calls through a
training abort.

### Optional teardown hook

The module named by `--custom-agent-function-path` may expose an `abort` function
alongside the agent entry point:

```python
async def abort(args) -> None:
    ...  # cancel this agent's in-flight external work
```

Miles calls this hook in the rollout process during oversampling abort after it
stops in-flight SGLang generation and cancels the agent calls, so under `subproc`
it cannot reach their module state; tell the external backend to stop, as below.
Use it when the agent drives an external sandbox or agent server whose work outlives
the cancelled call and would otherwise keep issuing completion requests until its
own length limit or timeout. The hook is optional; modules without it continue to
work.

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

**Do not set TITO control fields.** Leave `input_ids`, `routed_experts_start_len`, and `logprob_start_len` unset. The server selects `lora_path` and rejects conflicting control values with HTTP 400. It forces `logprobs=True`, `return_meta_info=True`, and the response metadata needed for TITO.

</Warning>

Chat responses from both session-server versions omit `choices[*].meta_info`. Ordinary JSON replies retain the message and `usage`, and carry standard `logprobs` only when the request set `logprobs`. On training sessions that record score-centering candidates, their `top_logprobs` lists are empty: the candidates travel as arrays in `meta_info`. The server keeps the full metadata and logprobs in session records for training sample collection and inspection through `GET /sessions/{id}`. Streaming chunks omit both.

### Choose template options per session

- Ordinary `chat_template_kwargs` use request > continued turn > launch defaults. Fields required to preserve the reused prompt retain their recorded values; fixed model settings override conflicts. Qwen3.8 selects `reasoning_effort` on a new root and preserves its value or absence on continuation. When the field is absent, the template defaults to `xhigh`.
- Omitted `tools` inherit on continuation; incompatible changes return HTTP 400 by default. Sampling parameters remain per-request. New roots resolve without turn history.
- Each successful turn records the full resolved request as `turn_args`. Exported session, tree-node and sample metadata omit `input_ids` and `messages` from this snapshot; the stored history remains complete.

### Choose the session behavior

Use `{"evaluation": true}` in `POST /sessions` for evaluation; omitting it defaults to training. The agentic generator sets this automatically, and the purpose stays fixed across turns, retries, and branches.

`temperature`, `top_p`, and `top_k` in `POST /sessions` provide defaults for omitted or `null` chat fields. Training requests must match the registered temperature or receive HTTP 400; eval temperature and `top_p`/`top_k` remain overridable. The agentic generator registers the sample's resolved values automatically.

Evaluation forces `return_sampling_mask`, `return_routed_experts`, and `return_indexer_topk` off and ignores `routed_experts_start_len`. TITO, logprobs, and sample collection still apply; engine-internal capture may remain enabled.

History handling depends on the selected server version:

- **v1 is linear.** Each request must extend the previous messages at the tail.
  Retrying the latest turn may roll back one assistant checkpoint, including to
  an empty session when retrying the first turn. Earlier divergence or a larger
  rollback is rejected.
- **v2 (Experimental) is an append-only tree.** A request attaches to the deepest checkpoint
  whose complete message path prefixes the request. Any unmatched suffix creates
  a branch, and existing branches are never deleted. A path whose last generation
  ended with `finish_reason=length` cannot be extended. By default, a leaf becomes
  no sample when a later request re-sent its exact prompt tokens; a later request
  that differs is a separate branch and its own sample
  (`--session-sample-picker-path`).

Whether a replayed message counts as "the same" as the stored one is decided by
`--session-message-matcher` (default `strict`); see
[Choose replay matching](#choose-replay-matching).

The v1 wrapper returns one `Sample`. The v2 wrapper returns a `list[Sample]`, one
for each selected tree leaf. Both versions reject `--partial-rollout`. With R3
replay, every pause mode except `retract` requests only additional R3 rows.
`retract` returns full R3 data on every turn and emits a warning because the
payloads can become very large.

The session server leaves the harness's control flow alone. It never lowers a turn's
`max_tokens`, refuses a turn for its length, or cuts a turn short: how long an episode runs
is the harness's decision. A request the session cannot serve, such as one that sets
`input_ids`, fails loud with an HTTP error.

`--max-seq-len` follows this rule. Miles passes it to your agent as
`metadata["max_seq_len"]` and truncates each collected trajectory to it after the episode,
so samples fit training memory. It does not stop the agent: a harness that reads the value
can end the episode there, and one that does not runs on past it. The agentic generator
logs a warning once when the limit is set.

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
| GLM-4.7 / 5 / 5.2 | `glm47` |
| GLM-5.3 / GLM-5.3-Flash (text sessions) | `glm53` |
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
| `loose_tool_call_ignore_reasoning` | Accepts everything `loose_tool_call` accepts, and leaves `reasoning_content` out of the comparison, so a replay that drops or rewrites it still matches. Content, tool calls and call IDs still have to match. |
| `role_content_only` | Compares only normalized `role` and `content`. **High risk:** different tool-call or reasoning histories can collapse into one session lineage. |
| dotted import path | Loads a trusted synchronous custom matcher; see [Customization](/user-guide/customization#session-message-matcher). |

The matcher only decides whether the message a client replays and the message stored at the same position in the session count as the same one.

- On a mismatch, the existing paths apply: v1 rolls back (or rejects), v2 branches.
- On a match, the stored messages and token snapshot stay authoritative inside the reusable prefix; only the suffix beyond it is tokenized anew from the client input.

Under v2 a replay the matcher rejects is still answered with a 200 and the session branches; a client that rewrites every assistant turn opens a new root on each turn, so each turn trains as its own sample. Rollout metrics report `rollout/session/roots_per_episode`, `rollout/session/leaves_per_episode` and `rollout/session/one_turn_per_root_ratio` (multi-turn sessions in which every turn opened a root), and the rollout logs a warning when that ratio passes one half. A clean single conversation is one root and one leaf.

Choose the matcher for the agent framework a recipe runs:

- Keep `strict` for a client that replays model messages verbatim.
- Choose `loose_tool_call` for a client that re-serializes tool-call arguments, for example to compact JSON.
- Choose `loose_tool_call_ignore_reasoning` when the engine runs a reasoning parser and the client resends assistant turns without `reasoning_content`. The stored reasoning stays in the training tokens.

Miles does not reconcile tool-call IDs across that boundary: deployments choosing `role_content_only` must themselves keep a stored call ID `A` followed by a replayed tool result referencing `B` protocol-compatible.

## Example

[`examples/swe-agent-harbor-docker`](https://github.com/radixark/miles/tree/main/examples/swe-agent-harbor-docker)
wires a multi-turn SWE agent, TITO session server, model-family registration,
reward, length limit, and environment teardown into production launchers.
