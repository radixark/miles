---
title: Agentic Rollout Internals
description: Code-level tour of Miles multi-turn rollout, TITO sessions, sandboxes, rewards, and failure handling, with a Terminal-RL port guide.
---

This page follows one agentic sample through Miles at the level of functions and
line numbers: who calls the generate function, how the multi-turn token sequence is
built, where tool output gets masked, how an external agent harness running in a
sandbox ends up as a token-exact training sample, and what happens to that sample
when the sandbox, the agent, or the verifier fails. The last sections map the
concepts onto a bash-tool-in-a-container Terminal-RL setup and list what is missing
to port one.

It complements the task-oriented pages; read those for setup, and this one when you
need to know what the code actually does:

- [Agentic Rollout (TITO)](/user-guide/agentic-rollout): wrapper setup, agent contract, `--tito-model` table
- [Generate Endpoint](/user-guide/generate-endpoint): the token-level `/generate` path
- [Agentic Environments](/user-guide/environments): the three plug-in layers and connector table
- [Harbor](/user-guide/harbor), [OpenEnv](/user-guide/openenv), [NeMo Gym](/user-guide/nemo-gym), [Verifiers](/user-guide/verifiers): connectors
- [Sandbox Providers](/user-guide/sandbox-providers), [Adding a Sandbox Provider](/developer/adding-a-sandbox-provider)
- [SWE-Agent with Harbor on Docker](/examples/swe-agent-harbor-docker), [Retool v2](/examples/retool-v2), [TMax Terminal-RL](/examples/tmax): worked examples

Line numbers were checked against `edf2c7ae3e` (rebased onto upstream `main` at
`e30d9a8ec6`, October 2026). They drift; function names do not. Paths without a
prefix are relative to the repository root. Paths prefixed `harbor/` are in the
Harbor repository at the branch the Harbor example pins, `harbor-miles-v0.20.0`;
every Harbor claim on this page was verified against commit `43e944d6`.

## 1. The call chain

### 1.1 Three nested layers

Every rollout, agentic or not, goes through the same three layers. Replacing an
outer one takes over everything inside it
([Integration shapes](/user-guide/environments#integration-shapes)).

```text
RolloutManager (Ray actor)                       train step N asks for rollout_batch_size groups
 └─ rollout function         --rollout-function-path   default InferenceRolloutFn
     generate_rollout_async                            inference_rollout_train.py:99
      ├─ data_source(over_sampling_batch_size)          -> list[list[Sample]]  (groups of n_samples_per_prompt)
      ├─ generate_and_rm_group(group)  one asyncio task per group            inference_rollout_common.py:142
      │   └─ generate_and_rm(sample)   one asyncio task per sample           inference_rollout_common.py:60
      │        async with generate_fn_semaphore                              inference_rollout_common.py:89
      │        └─ generate function   --custom-generate-function-path        single_turn / multi_turn / agentic_tool_call
      │             └─ [agentic only] agent function  --custom-agent-function-path (own process by default)
      │        reward: async_rm / batched_async_rm (unless --group-rm)       inference_rollout_common.py:117-136
      │   [--group-rm] batched_async_rm over the whole group                inference_rollout_common.py:186
      ├─ apply_preput_filters(group): aborted -> missing reward -> dynamic   common_filters.py:70
      └─ abort(): /abort_request on every engine + agent abort hook         inference_rollout_train.py:27
```

`GenerateState` (`inference_rollout_common.py:32`) is built once per rollout
function. It loads the tokenizer, the default sampling params
(`compute_sampling_params`, `inference_rollout_common.py:191`), the generate function
(`load_generate_function`, `compatibility.py:75`), and the one concurrency limiter
that exists at this layer:

```python
self.generate_fn_semaphore = asyncio.Semaphore(
    args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
)   # inference_rollout_common.py:41; --sglang-server-concurrency defaults to 512 (sglang_utils/arguments.py:153)
```

The semaphore is held for the **whole** generate call, so in the agentic path it
bounds concurrent *episodes* (and therefore concurrent sandboxes), not concurrent
HTTP requests. In practice the tighter bound is the submission pacing: with the
default group-level scheduler (`submission_scheduler.py:14`),
`generate_rollout_async` submits `over_sampling_batch_size` groups (default
`rollout_batch_size`, `arguments.py:4060`) whenever fewer groups are pending than
the remaining budget, so roughly `rollout_batch_size × n_samples_per_prompt`
episodes run at once.

### 1.2 The generate-function contract

`load_generate_function` accepts a class (instantiated), a new-form function, or an
old-form one. It detects old form purely by signature: three or more parameters and
the first is not named `input` (`compatibility.py:88`); old-form functions are
wrapped by `LegacyGenerateFnAdapter` (`compatibility.py:58`). A sample can override
the function per row via `Sample.generate_function_path`
(`inference_rollout_common.py:98`), which is how an eval dataset can use a different
generator.

```python
@dataclass(frozen=True)
class GenerateFnInput:          # base_types.py:114
    state: GenerateState        # tokenizer, processor, args, sampling defaults
    sample: Sample
    sampling_params: dict       # deep-copied per call; max_new_tokens = --rollout-max-response-len
    evaluation: bool

@dataclass(frozen=True)
class GenerateFnOutput:         # base_types.py:126
    samples: Sample | list[Sample]   # a list: tree/compaction/multi-agent outputs
```

`Sample` (`miles/utils/types.py:165`) fields a generate function reads and writes:

| Field | In | Out | Notes |
|---|---|---|---|
| `prompt` | str or OpenAI `messages` | unchanged | Agentic path requires a messages list; do not pass `--apply-chat-template` |
| `metadata` | dataset `--metadata-key` column | + agent's returned dict, session metadata | Where `instance_id`, `task_id`, and env-computed `reward` travel |
| `label` | dataset label | unchanged | Used by rule-based RMs |
| `tokens` | empty | prompt ids + full response ids | The complete trained sequence |
| `response_length` | 0 | tokens after the first prompt | Includes tool/env tokens |
| `loss_mask` | None | 1 = model token, 0 = injected | Length must equal `response_length` (`types.py:353`) |
| `rollout_log_probs` | None | one per response token | 0.0 on injected tokens |
| `status` | `PENDING` | `COMPLETED` / `TRUNCATED` / `ABORTED` / `FAILED` | Set from the last engine `finish_reason` (`types.py:505`) |
| `reward` | None | usually left None | If set in generate, the RM step is skipped (`inference_rollout_common.py:135`) |
| `weight_versions` | [] | one entry per engine call | Per-token policy version spans, used for staleness |
| `rollout_routed_experts`, `rollout_indexer_topk`, `rollout_sampling_mask`, `rollout_topk_*` | None | optional replay payloads | R3, sampling-support, and top-k logprob replay |
| `kv_cache_namespace` | set by the rollout function under `--namespaced-radix-cache` | unchanged | Sent to SGLang as `extra_key`, partitioning the radix cache per rollout call; on the session path only v2 accepts it (`arguments.py:4000-4004`) |
| `remove_sample` | False | usually False | Set later by `--rollout-sample-filter-path` |

`FAILED` exists in the enum but no built-in path sets it and no built-in filter
reads it; treat it as reserved.

### 1.3 Two transports to SGLang

There are two ways a generate function reaches the engines, and they are the two
styles of multi-turn rollout.

```text
                       ┌──────────────────────────────── rollout process ─────────────────────────────┐
 token style           │ multi_turn.generate ──POST /generate {input_ids}──────────────────────┐       │
 (/generate)           │                                                                       ▼       │
                       │                                                    router (SGLang or Miles)   │
 message style         │ agentic_tool_call.generate                                   │  ▲            │
 (TITO session)        │   ├─ POST /sessions ─► session server i (1 of 32 procs)     ▼  │            │
                       │   └─ agent fn ─► harness (maybe in a sandbox)             SGLang engines     │
                       │        POST {base}/sessions/<id>/v1/chat/completions {messages}             │
                       │             session server renders input_ids ──POST /v1/chat/completions──► router
                       └──────────────────────────────────────────────────────────────────────────────┘
```

- **`/generate`** (`generate_endpoint_utils.py:57`): the generate function owns the
  token list and posts `input_ids`; SGLang returns `output_token_logprobs` as
  `(logprob, token_id)` pairs (`generate_endpoint_utils.py:103`). No
  re-tokenization of model output ever happens.
- **Session server** (`miles/rollout/session/`; v2 in `session/v2/`): a standalone
  FastAPI process per instance (`--session-server-workers`, default 32; launched from
  `spec_session_server`, `miles/ray/specs/inference.py:219`). `wait_session_server_ready`
  (`miles/ray/rollout/router_manager.py:68`) waits for them and publishes
  `args.session_server_instances`, each with a cluster address, an external address,
  and an instance id. It speaks OpenAI Chat Completions (and Anthropic Messages,
  `sessions.py:136`) to the agent, and forwards to the router's
  `/v1/chat/completions` with `input_ids` already filled in (`core.py:326`). It is
  the only way an unmodified external harness can produce token-exact samples.

The router is either SGLang's router or `MilesRouter` (`miles/router/router.py:36`,
least-active-requests selection at `router.py:228`, background health checks that
quarantine dead workers at `router.py:82`). The session server tags every upstream
request with `X-SMG-Routing-Key: <session_id>` (`core.py:369`), so under a
consistent-hashing router policy all turns of one episode land on the same engine
and hit its prefix cache.

## 2. The token-level loop: `multi_turn.generate`

`miles/rollout/generate_hub/multi_turn.py` is Miles' built-in equivalent of
open-instruct's `process_request`. Retool v2 uses it. It is short enough to read in
full; the essentials:

```python
prompt_ids = compute_prompt_ids_from_sample(state, sample, tools=tool_specs)   # multi_turn.py:47
sample.tokens = prompt_ids.copy()
for _turn in range(args.generate_max_turns):                                   # default 16, multi_turn.py:98
    payload, halt = compute_request_payload(args, sample.tokens, input.sampling_params, ...)
    if payload is None: sample.status = halt; break                            # context exhausted -> TRUNCATED
    output = await post(router + "/generate", payload)
    await update_sample_from_response(..., update_loss_mask=True)              # model tokens: mask 1, real logprob
    if finish_reason in ("abort", "length"): break                             # multi_turn.py:76
    _, tool_calls = tool_call_parser.parse_non_stream(output["text"])          # multi_turn.py:81
    if not tool_calls: break                                                   # no tool call = final answer
    tool_messages = await execute_tool_calls(tool_calls, execute_tool)         # sequential, multi_turn.py:86
    update_sample_with_tool_responses(sample, tool_messages, tokenizer)        # tool tokens: mask 0, logprob 0.0
```

**Prompt construction.** `apply_chat_template(messages, add_generation_prompt=True,
tools=tool_specs)` runs once (`generate_endpoint_utils.py:35`), exactly like
open-instruct's preprocessing step, except it happens at rollout time.

**Tool-call parsing.** `create_tool_call_parser` (`tool_call_utils.py:23`) builds
SGLang's own `FunctionCallParser` from the OpenAI tool specs and
`--generate-tool-call-parser` (e.g. `qwen25`, `glm47`). Parsing is on the decoded
text of the turn.

**Tool execution.** `execute_tool_calls` (`tool_call_utils.py:30`) awaits each call
in order and requires `execute_one(name, params) -> str`
(`tool_call_utils.py:54`). The function receives **no sample, metadata, or episode
handle** (see `examples/retool_v2/tool_sandbox.py:359`), so it cannot address a
per-episode container. An exception raised inside it propagates out of the
generate function (section 4.5 covers what that does to the group).

**Observation tokens.** `_tokenize_postfix_messages` (`tool_call_utils.py:80`)
renders `[dummy_user, dummy_assistant_with_matching_tool_calls]` with and without
the tool messages (the "with" render uses `add_generation_prompt=True`) and keeps the
token suffix. The suffix therefore contains the tool-role wrapper **and** the next
assistant opener (for Qwen, `<|im_start|>assistant\n`), all at `loss_mask=0` and
`rollout_log_probs=0.0` (`tool_call_utils.py:68-69`). This is the same role-template
approach as open-instruct's `format_tool_outputs`, but derived from the chat
template instead of hardcoded, and it asserts append-only rendering
(`tool_call_utils.py:97`). Thinking templates that rewrite earlier turns fail that
assertion; use the session path for those.

**Budgets.** These differ from open-instruct in a way that matters for long
episodes:

| Budget | Where | Semantics |
|---|---|---|
| `--rollout-max-response-len` | `sampling_params["max_new_tokens"]` | **Per turn.** The same value is passed on every turn (`multi_turn.py:57`); there is no running total |
| `--rollout-max-context-len` | `compute_request_payload`, `generate_endpoint_utils.py:68` | Caps `max_new_tokens` at `context - len(tokens)`; when that reaches 0 the loop ends with `TRUNCATED` (`generate_endpoint_utils.py:71`) |
| `--generate-max-turns` | loop bound | Model turns, not tool calls |

Without `--rollout-max-context-len`, total length is bounded only by turns ×
per-turn budget. Tool output is appended untruncated, so one large observation can
push `sample.tokens` past the context; the next turn then halts as `TRUNCATED` (if
the flag is set) or SGLang rejects the request (if not).

**Termination order:** payload refusal (context) → `abort`/`length` finish → no tool
call → turn limit. Hitting the turn limit leaves the sample ending on masked tool
tokens with the last turn's status (normally `COMPLETED`).

**Known gaps in the code:** `rollout_routed_experts` is reassigned rather than
concatenated each turn (`generate_endpoint_utils.py:138-139`, a `TODO`), partial rollout
is rejected, and there is no format-error nudge or per-call timeout; add those in
your `execute_tool`.

## 3. The message-level loop: `agentic_tool_call` and the TITO session server

This is the path every sandboxed connector (Harbor, OpenEnv, NeMo Gym) uses. The
agent (or an external harness) speaks plain OpenAI chat; Miles records tokens on the
side. The design goal, "Token-In-Token-Out", is that the model's own output tokens
are never re-tokenized and injected text is tokenized exactly once.

### 3.1 One sample, end to end

`agentic_tool_call.generate` (`generate_hub/agentic_tool_call.py:51`):

1. `OpenAIEndpointTracer.create` (`openai_endpoint_utils.py:53`) picks one of
   `args.session_server_instances` at random (`openai_endpoint_utils.py:67`; this is
   the only routing decision, every later call reuses that instance), and
   `POST /sessions` with the sample's sampling params, `evaluation` flag, and
   `extra_key = sample.kv_cache_namespace` (set only under
   `--namespaced-radix-cache`). The server registers the temperature; a
   training request with a different temperature gets HTTP 400
   (`request_args.py:130`). v2 adds the `extra_key` to every upstream request; v1
   refuses one, which is why `--namespaced-radix-cache` requires v2.
2. It calls your agent function through `call_agent_function`
   (`miles/rollout/agentic/agent_function.py:57`). With
   `--custom-agent-function-mode subproc`, the default, each call runs in a fresh
   child process: arguments and return value must pickle, module-level state lasts
   one call, and limits shared across calls go through `rollout_limits` (section
   4.3). `inline` awaits it on the rollout's event loop. The arguments are
   `base_url = tracer.agent_base_url` (`http://<external addr>/sessions/<id>`,
   section 4), `prompt = sample.prompt`,
   `request_kwargs = build_chat_request_kwargs(sampling_params)`
   (`agentic_tool_call.py:206`, maps `max_new_tokens` to `max_tokens`), and
   `metadata = sample.metadata + {max_seq_len, session_server_instance_id,
   session_server_id}`, where `session_server_id` is the owning instance's cluster
   `ip:port`.
3. **Any exception from the agent is caught and logged as a warning**
   (`agentic_tool_call.py:98`), including an agent process that dies or returns an
   unpicklable value; collection runs regardless.
4. `collect_samples` posts `/sessions/<id>/samples` with `max_seq_len` (v2 also
   sends the agent's returned dict as `metadata`), then always deletes the session
   (`openai_endpoint_utils.py:90-118`, 120 s timeout each). The collect is one
   aiohttp POST with `Connection: close`; a reply cut short of its
   `Content-Length` counts as a transport error (`post_buffer_no_retry`,
   `miles/utils/http_utils.py:276`).
5. Outcomes:
   - collect timeout or transport error → a copy of the input sample with
     `status=ABORTED` (`agentic_tool_call.py:110-124`);
   - collect returns no samples (no model call recorded, or everything truncated
     away) → `ABORTED` (`agentic_tool_call.py:126-135`);
   - collect returns a non-2xx (e.g. 422 on an assembly assertion) →
     `RuntimeError` propagates out of the generate function;
   - otherwise v1 merges the agent's returned dict into `sample.metadata`
     (`agentic_tool_call.py:159-164`) and returns one `Sample`; v2 returns a list
     whose metadata and reward the server already applied (section 3.5).
6. `agent_metrics.total_tool_time`, if the agent reports it, becomes
   `Sample.non_generation_time` for throughput accounting
   (`agentic_tool_call.py:169-172`).

The agent function must return a dict (merged into metadata) or `None`. Nothing in
the contract lets it say "drop this sample"; section 4.5 explains the consequences.

### 3.2 One chat turn inside the session server

`SessionCore.chat_completions` (`session/core.py:326`; v2 overrides it in
`session/v2/core.py`) runs in three phases so the per-session lock is never held
across inference:

```text
Phase 1 (lock):  parse body; resolve args (request_args.py); pick checkpoint; render input_ids
Phase 2 (none):  POST router/v1/chat/completions {input_ids, messages, logprobs:true, return_meta_info:true, ...}
Phase 3 (lock):  validate response; commit checkpoint = input_ids + output_ids; append SessionRecord
```

- **Forced request fields** (`resolve_request_args_by_config`,
  `request_args.py:136-197`): `logprobs=True`, `return_meta_info=True`,
  `no_stop_trim=False`, R3/indexer replay per launch flags. A client-supplied
  `input_ids`, `logprob_start_len`, or training `routed_experts_start_len` is
  rejected with 400. `stream=true` is honored as fake streaming: the upstream call
  stays non-streaming and the full response is re-emitted as one SSE chunk
  (`core.py:119`). The reply to the agent omits `meta_info`, and omits `logprobs`
  unless the agent's own request asked for them; the stored record keeps both.
- **Tool calls and reasoning are parsed by SGLang**, not by Miles, using the
  engine's `--sglang-tool-call-parser` / `--sglang-reasoning-parser`. A named
  `--tito-model` family declares the pair it expects (`tito_tokenizer.py`,
  `reasoning_parser` / `tool_call_parser` class attributes; e.g. Qwen3 →
  `qwen3`/`qwen25`, GLM-4.7 → `glm45`/`glm47`), but **nothing sets the engine
  flags from it**: `resolve_reasoning_and_tool_call_parser` is only called from
  `test_utils/session_verify_runner.py`, and both default to `None`. Pass them
  explicitly; otherwise the model's tool-call markup comes back as plain `content`
  with empty `tool_calls` (verified with `--tito-model qwen35`: every TMax episode
  ended after one turn until `--sglang-reasoning-parser qwen3
  --sglang-tool-call-parser qwen3_coder` were added; `examples/tmax` and
  `swe-agent-harbor-docker` now pass the pair). The agent receives structured
  `tool_calls` and `reasoning_content` and must echo them back verbatim on the next
  request (see 3.4).
- **Validation** (`extract_completion`, `core.py:159`): `finish_reason=abort` →
  503; missing `output_token_logprobs` or a length mismatch against
  `completion_tokens` → 502. Any non-200 from upstream (e.g. 400 "context too
  long") is passed back to the agent **unrecorded** (`core.py:376`), so the agent
  can retry or stop.
- **Commit** (`LinearTrajectory.update_pretokenized_state`,
  `linear_trajectory.py:206`) asserts that the previous checkpoint is a prefix of
  the new `prompt_ids + completion_ids`, tolerating `max_trim_tokens` trailing
  differences (`assert_pretokenized_prefix`, `linear_trajectory.py:26`).

### 3.3 How input_ids are built, and where drift can creep in

`_render_token_ids` (`linear_trajectory.py:152`):

- **First turn:** `tito_tokenizer.apply_chat_template(messages,
  add_generation_prompt=True)` with the resolved template args, including `tools`
  from the request (`linear_trajectory.py:178`). This is where tool schemas enter
  the system prompt.
- **Later turns:** the stored checkpoint tokens (the exact prompt ids plus the
  exact sampled ids of the previous turn) are reused verbatim, and only the appended
  messages are tokenized (`merge_tokens`, `linear_trajectory.py:199`).
  `tokenize_additional_messages` (`tito_tokenizer.py:313`) renders
  `[dummy_system, dummy_assistant(prev tool_calls)]` with and without the new
  messages and encodes the text suffix, including the generation prompt
  (`tito_tokenizer.py:344`). Appended roles must be in the family's
  `allowed_append_roles`.
- **Boundary fixes per family:** Qwen3 stops at `<|im_end|>` but the template
  renders `<|im_end|>\n`, so `Qwen3TITOTokenizer.merge_tokens` appends the missing
  newline (`tito_tokenizer.py:420`). GLM-4.7's `<|observation|>`/`<|user|>` are both
  stop tokens and next-message openers, so the trailing one is stripped
  (`tito_tokenizer.py:535`, `max_trim_tokens = 1` at `:505`).

So model tokens have zero retokenization drift by construction. What *can* differ
from a canonical `apply_chat_template` of the final message list is the rendering of
injected messages in context, and anything the model emitted that the template
would render differently (a degenerate stop token, a non-canonical tool-call
spelling). Miles does not reject these; it reports them.
`compute_session_mismatch` (`linear_trajectory.py:418`) diffs the accumulated ids
against a canonical render and ships the result as
`sample.metadata["tito_session_mismatch"]`. That is the metric to watch for silent
template problems. `--tito-model default` gets a startup warning that this
invariant is best-effort (`arguments.py:3534`).

### 3.4 Retries, rollback, and branching

Harnesses do not always extend history at the tail: they retry a turn, edit a tool
result, or compact the context. Which stored checkpoint a request continues is
decided by comparing messages with `--session-message-matcher` (default `strict`;
role, content, `reasoning_content`, `tool_calls`).

- **v1 (`--use-session-server` or `--use-session-server v1`)** is linear.
  `_find_rollback_checkpoint` (`linear_trajectory.py:243`) allows discarding at
  most one generated checkpoint (`MAX_ASSISTANT_ROLLBACK_STEPS = 1`,
  `linear_trajectory.py:23`); anything deeper is a 400. Context compaction or
  summarization in the harness therefore fails on v1.
- **v2 (`--use-session-server v2`)** keeps an append-only tree; every divergence is
  a branch. Extending a node that ended with `finish_reason=length` is a 409
  (`v2/session_state.py:75`). At collect time a picker
  (`--session-sample-picker-path`, default `drop_same_prompt_retries`,
  `v2/picker_hub/drop_same_prompt_retries.py:7`) chooses which leaves become
  samples, and `default_postprocess` (`v2/postprocessor_hub/default_postprocess.py:15`)
  masks shared prefix completions so each completion is trained in exactly one leaf,
  merges the agent's other keys into each kept sample's `metadata`, and assigns
  `agent_metadata["reward"]` to every kept leaf as `Sample.reward`
  (`default_postprocess.py:53-58`). v2 rejects `--group-rm` and
  `--recompute-logprobs-via-prefill` (`arguments.py:3474`).

A harness that re-serializes the assistant message (drops `reasoning_content`,
rewrites tool-call argument JSON, sends `{role, content}` only) looks like
divergence under `strict`. The OpenEnv agent echoes `message.model_dump()` for this
reason (`examples/experimental/openenv/openenv_agent_function.py:295`).

### 3.5 From records to one training sample

Both versions assemble on the owning instance, because the records never leave it.
v1's `SessionCore.collect_samples` (`core.py:278`) runs the three steps below over
the session's linear record list and returns one sample. v2's
`SessionCoreV2.collect_samples` (`v2/core.py:78`) runs them once per leaf, over that
leaf's root-to-leaf records (`build_leaf_material`, `v2/utils.py:45`), then applies
the picker and postprocessor of section 3.4 and returns one sample per kept leaf.

1. `compute_samples_from_openai_records` (`session/samples/merge.py:27`) builds one
   `Sample` per recorded turn: `tokens = input_ids + output_ids`,
   `rollout_log_probs` from `output_token_logprobs`, `loss_mask = [1] *
   len(output_ids)` (`merge.py:140`). It walks a cursor along the accumulated token
   sequence and trims trailing output tokens that the next turn's render consumed
   (at most `max_trim_tokens`, zero on the last turn, `merge.py:76`), then asserts
   the whole accumulated sequence was consumed (`merge.py:100`). Status comes from
   `finish_reason`: `stop`/`tool_calls` → `COMPLETED`, `length` → `TRUNCATED`,
   `abort` → `ABORTED` (`merge.py:161-166`).
2. `truncate_samples_by_total_tokens` (`merge.py:225`) enforces `--max-seq-len`
   (prompt + everything). Turns that fit are kept; the first overflowing turn is cut
   to the budget and marked `TRUNCATED`; if nothing of its output would remain it is
   dropped (`merge.py:243-244`); later turns are dropped.
3. `merge_samples` (`generate_utils/sample_utils.py:19`) folds turn samples
   left to right. The gap between turn *i*'s tokens and turn *i+1*'s output is the
   injected observation plus the next assistant opener; it gets
   `loss_mask=0` and logprob `0.0` (`sample_utils.py:147-167`). The merged status is
   the last merged turn's (`sample_utils.py:177`).

Two consequences worth internalizing:

- **A non-`COMPLETED` turn ends the trajectory for training** (`sample_utils.py:25`).
  If turn 5 hit `max_tokens` (`length`) and the harness kept going (v1 does not stop
  it), turns 6+ are silently dropped from the sample while the episode's reward,
  computed on the full episode, still applies. Harbor's `harbor-miles` branch
  configures terminus-2 to abort on truncation for this reason
  (`harbor/agent_server/trial_runner.py:68,76-78`, mirrored in
  `harbor_agent_function.py:136,147-150`). The abort raises
  `SingleTurnMaxSeqLenExceededError`, which ends the trial **without running the
  verifier**, so the sample scores 0 (section 4.5).
- **The last observation is never in the sample.** The sample ends with the last
  model output; test output after the final turn is not trained on or even stored.

### 3.6 Budgets and truncation on the session path

| Knob | Enforced by | Effect |
|---|---|---|
| `--rollout-max-response-len` | forwarded as `max_tokens` on every chat request | Per-turn cap. `length` finish → that turn `TRUNCATED` → trajectory ends there for training |
| `--max-seq-len` | `truncate_samples_by_total_tokens` at collect; also forwarded to the agent in `metadata["max_seq_len"]` | Whole-trajectory cap, applied after the fact at a turn boundary. The episode keeps running unless the harness honors the forwarded value |
| SGLang context length (`--sglang-context-length`) | engine | Over-long prompt → 400 passed to the agent, unrecorded |
| Max turns / steps | the agent or harness | Miles has no turn limit on this path (`OPENENV_MAX_TURNS`, `HARBOR_AGENT_MAX_ITERATIONS`, ...). On the Harbor agent server `HARBOR_AGENT_MAX_ITERATIONS` reaches only terminus-2 and claude-code (`harbor/agent_server/trial_runner.py:80-82,93-94`); mini-swe-agent uses its own packaged config |
| Observation size | the agent | e.g. OpenEnv keeps 4000 chars (`openenv_agent_function.py:80`) |

Truncated samples are **trained normally** with whatever reward they carry.
`Sample.Status.TRUNCATED` only feeds the `truncated` training field and the
`truncated_ratio` metric (`train_data_conversion.py:95`, `ray/rollout/metrics.py:116`);
no built-in path masks or zeroes them. To exclude them from the loss, set
`s.remove_sample = True` in a `--rollout-sample-filter-path` hook: the loss mask
goes to zero (`train_data_conversion.py:112`) but the sample still counts in the
group's reward normalization.

### 3.7 Abort at the end of a rollout step

Sync rollout over-samples and then aborts stragglers (`abort`,
`inference_rollout_train.py:27`): `POST /abort_request {abort_all: true}` on every
engine, then `call_agent_abort_hook` (`miles/utils/misc.py:35`), which calls an
`abort(args)` function living next to your agent function if you define one. The
hook runs in the rollout process, so under `subproc` it shares no memory with the
agent processes and has to reach them from outside: TMax bumps an abort counter in a
file and removes the rollout's containers (`examples/tmax/tmax_agent_function.py:165`).
The Harbor agent-server example posts `/flush` for each session-server instance
(`examples/swe-agent-harbor-docker/swe_agent_function.py:147`). The server cancels
those `/run` tasks and SIGKILLs their trial worker processes
(`harbor/agent_server/trial_runner.py:458-466`), which stops them calling SGLang.
On the Docker backend it does **not** free their containers; section 4.2 explains
why.
Without a hook, harnesses keep running until their own limits, getting 503s
(`UpstreamGenerationAbortedError`) or 404s from deleted sessions. Without
`--partial-rollout` (which the session path rejects) the aborted groups are
discarded. `--fully-async` never aborts this way and never calls the hook.

## 4. Sandboxes

Miles core has no sandbox abstraction. `miles/rollout/agentic/` holds the
credential registry (`PROVIDER_CREDENTIALS`, `credentials.py:29`),
`openai_session_url` (`session.py:4`, appends `/v1` to the session URL), the
agent-process runner behind `--custom-agent-function-mode` (`agent_function.py:57`),
and `rollout_limits`. Containers are created by the connector, inside the agent
function. Off-cluster agents get the session server's external address in
`base_url`: `--session-server-external-host` (one host for every instance, which
pins the session servers to the head node) or a per-node `MILES_NODE_EXTERNAL_IP`,
else the cluster address (`_compute_external_addr`,
`miles/ray/rollout/router_manager.py:117`). The
launcher refuses a leftover `MILES_ROUTER_EXTERNAL_HOST`, which is no longer read.

### 4.1 Where the container lives

| Recipe | Who creates the sandbox | Where the agent loop runs | Who calls the model |
|---|---|---|---|
| `examples/swe-agent-harbor-docker` | separate Harbor `miles_agent_server.py` (harbor `harbor-miles-v0.20.0` branch) on a Docker host, one subprocess per trial | installed agents (mini-swe-agent, claude-code, opencode) inside the task container; terminus-2 in the trial process on the host, driving tmux in the container | the harness, over the network, to the session URL |
| `examples/experimental/harbor` (in-process) | `Trial.run()` in the agent process, cloud provider via `HARBOR_ENV_TYPE` | Harbor harness in the sandbox | the harness from inside the sandbox |
| `examples/experimental/openenv` (shared) | one tbench2 env server (`OPENENV_ENV_URL`) with its own `MAX_CONCURRENT_ENVS` | the Miles agent function (`multi_turn`) | the agent function, via `AsyncOpenAI` |
| `examples/experimental/openenv` (per-episode) | `SandboxBackend.episode_env` on Daytona / E2B / Modal | the Miles agent function | the agent function |
| `examples/experimental/nemo-gym` | NeMo Gym's own sandbox API | NeMo Gym agent server | NeMo Gym's harness |
| `examples/tmax` | `TmaxSandbox` in the agent process, through docker-py on a Podman (or Docker) API socket | the Miles agent function (`tmax_agent_function.run`) | the agent function, via `AsyncOpenAI` |

**The Harbor agent-server contract.** The server is `harbor/miles_agent_server.py`
plus the `harbor/agent_server/` package. The in-process module ports its harness
bindings (`harbor_agent_function.py:105-117`).

- **Endpoints.** `POST /run` runs one trial (`miles_agent_server.py:115`).
  `GET /health` returns `{"status": "ok"}` (`:256`). `POST /flush` cancels the
  trials of one session-server instance (`:268`). `GET /clients` lists the
  tracked session servers (`:299`). `POST /flush_all` cancels every trial and runs
  `docker compose down --volumes` on the compose projects the server has
  registered (`:314`). The last three check `Authorization: Bearer
  $HARBOR_ADMIN_SECRET` only when that variable is set (`:261-265`).
- **Request** (`RunRequest`, `agent_server/models.py:8-19`): `base_url`, `model`,
  `sampling_params`, `api_key` (default `"dummy"`), `instance_id`, `agent_name`
  (default `mini-swe-agent`), `max_seq_len`, `session_server_id`,
  `session_server_instance_id`. Extra keys are allowed, which is why
  `swe_agent_function.run` can spread the whole sample `metadata` into the body
  (`swe_agent_function.py:104-109`).
- **Response** (`RunResponse`, `models.py:22-26`): exactly `reward` (float,
  default 0.0), `exit_status`, `agent_metrics`, `eval_report`. The worker also
  reports `trial_dir` and `compose_project`, but `RunResponse` has no field for
  them and pydantic drops them. **No token ids, logprobs, or messages come back.**
  Trial failures come back as HTTP 200 with `reward: 0.0` and a named
  `exit_status` (`_error_response`, `agent_server/results.py:119-125`), never as
  an HTTP error. The only non-200 responses are FastAPI's 422 for a malformed body
  and a server crash.
- **How the harness reaches the session server.** `base_url` is
  `http://<session-server>/sessions/<id>/v1`: the instance's external address
  (section 4) plus `/v1` from `openai_session_url`. Harbor passes it through
  unchanged. mini-swe-agent gets it as `OPENAI_API_BASE` with model
  `openai/model` (`trial_runner.py:136-140`). terminus-2 gets it as `api_base`
  (`:70`). claude-code gets it as `ANTHROPIC_BASE_URL` and hits the session
  server's `/v1/messages` route (`:96-98`). opencode gets it as an
  `openai-compatible` provider `baseURL` (`:110-123`). The token-level trajectory
  is never sent back by Harbor. Miles rebuilds it from the session server's own
  records at collect time (section 3.5). mini-swe-agent is an installed agent
  that runs **inside the task container**, so model calls leave the sandbox. That
  is why the launcher's `--session-server-external-host` (or each node's
  `MILES_NODE_EXTERNAL_IP`) must route from the Docker host's containers.
- **Session liveness.** `session_server_id` must have the same port as
  `base_url` and a host that resolves to the same address, or `/run` returns
  `exit_status: "InvalidSessionServerId"` (`agent_server/net.py:98`,
  `miles_agent_server.py:124-135`). `swe_agent_function` sends the `host:port`
  that `base_url` names (`swe_agent_function.py:115-117`), so the two agree. The
  server then polls `http://<session_server_id>/health`
  every `HEARTBEAT_INTERVAL_SEC` (120 s). After `HEARTBEAT_FAILURE_THRESHOLD` (3)
  consecutive misses it flushes that instance's trials. A reply whose
  `session_server_instance_id` does not match also counts as a miss (`agent_server/background.py:29-31,57-118`). Miles' session server
  reports its instance id on `/health` (`miles/rollout/session/core.py:236-240`).
- **What Harbor drops.** For mini-swe-agent (and any agent without a binding),
  `_agent_connection_config` returns no kwargs, so **`sampling_params` never
  reaches the harness** (`trial_runner.py:136-140`). Temperature, `top_p`, and
  `top_k` still apply, because the session server fills them from the session's
  defaults (`apply_session_sampling_defaults`, `request_args.py:119`). `max_tokens`
  (`--rollout-max-response-len`) is not sent by Harbor at all, and the session
  server adds no default, so on this path the per-turn cap is whatever
  mini-swe-agent's own config sets. The `model_info` block the server adds for
  `openai/...` models (`AGENT_MAX_INPUT_TOKENS`/`AGENT_MAX_OUTPUT_TOKENS`,
  `trial_runner.py:354-360`) is swallowed by mini-swe-agent's `**kwargs`.
  terminus-2 does receive `sampling_params` as `llm_call_kwargs` (`:69`).

### 4.2 Lifecycle

- **Harbor agent server** (`harbor/agent_server/trial_runner.py:304-481`). Each
  `/run` waits on the server semaphore. It then runs **one trial in a fresh Python
  subprocess** (`python -m agent_server.trial_runner`, `:434-457`), which gives
  the trial its own event loop and environment. The child resolves
  `HARBOR_TASKS_DIR/<instance_id>`. A bad id returns `InvalidInstanceId`, and a
  missing directory returns `TaskNotFound` (`:329-349`). The child then builds a
  `TrialConfig` and runs `Trial.create` + `trial.run()` (`:383-409`).
  `Trial.run` (`harbor/src/harbor/trial/trial.py:346-374`) proceeds in this order:
  1. **Environment start.** `docker compose build` runs, unless the task names a
     prebuilt `docker_image`. A prebuilt image is pulled by `compose up`. Then
     `down --remove-orphans` clears stale containers, and `up --detach --wait`
     starts the main service, which is `sleep infinity`
     (`harbor/src/harbor/environments/docker/docker.py:912-968`,
     `docker-compose-build.yaml`). All of this runs under
     `asyncio.wait_for(build_timeout_sec × multiplier)` (`trial.py:1174-1185`).
     A per-image build lock exists (`docker.py:122,937-941`), but it is a
     class-level `asyncio.Lock` inside each worker process. It therefore does
     **not** serialize the concurrent builds of one task's GRPO siblings, which
     run in separate processes.
  2. **Agent setup.** Installed agents such as mini-swe-agent install themselves
     into the container on every trial: `apt-get`, `uv`, then
     `uv tool install mini-swe-agent`, plus two patches, one capping command
     output at 1 MiB head+tail and one adding step timing
     (`harbor/src/harbor/agents/installed/mini_swe_agent.py:769-811`). Setup is
     bounded by 360 s × multiplier (`trial.py:81,1187-1209`) and needs network
     access from the sandbox.
  3. **Agent run** under `asyncio.wait_for(agent timeout)` (`trial.py:421-481`).
     For mini-swe-agent this is one long
     `docker compose exec ... mini-swe-agent --yolo --exit-immediately`
     (`mini_swe_agent.py:1003-1014`). Its commands run inside the container
     through mini-swe-agent's own local environment, so the per-command timeout
     and step limit come from mini-swe-agent's packaged config, not from Harbor.
     A host-side poller reads the trajectory file every 5 s. It kills the run
     with `MaxSeqLenExceededError` when the last call's
     `prompt_tokens + completion_tokens` exceeds `max_seq_len`, or when the file
     exceeds 500 MB (`mini_swe_agent.py:1016-1089`).
  4. **Verifier.** It runs even after an agent timeout or a non-zero agent exit
     (`harbor/src/harbor/trial/single_step.py:75-86`). By default it runs in the
     agent's container (`SHARED`). It uploads `tests/` only now, runs the test
     script, and parses the reward file (section 4.5).
  5. **Teardown.** `stop(delete=...)` is shielded from cancellation
     (`trial.py:1213-1234`). On Docker, `HARBOR_DELETE_CONTAINERS` defaults to
     **false** (`trial_runner.py:235-242`). The trial then ends with plain
     `docker compose down`, which removes the containers and the network but
     keeps named volumes and the built image. With `true` it runs
     `down --rmi local --volumes --remove-orphans` (`docker.py:986-1019`). A
     background loop runs `docker container prune`, `image prune -a`, and
     `buildx prune --max-used-space 100GB` every 30 min
     (`agent_server/background.py:123-154`). The server refuses to start without
     a Docker Hub entry in `~/.docker/config.json` (`agent_server/docker_ops.py:89`).
  The only other backend is `HARBOR_ENV_TYPE=daytona`, which sets delete-on-stop
  plus 540-minute auto-stop and 1440-minute auto-delete orphan backstops
  (`trial_runner.py:211-234`). E2B, Modal, and the rest are reachable only
  through the in-process path.
- **Cancellation and flush on the agent server.** `/flush` (the Miles abort
  hook), `/flush_all`, and the heartbeat all cancel the `/run` task. The parent
  then `proc.kill()`s the trial worker (`trial_runner.py:458-466`) and answers
  `exit_status: "Flushed"` (`miles_agent_server.py:215-222`). SIGKILL means the
  child's step 5 never runs. The compose project name is only registered in the
  parent from the child's final RESULT line (`trial_runner.py:467-476`), so a
  killed trial's project is unknown to `/flush_all` too. On Docker, a
  flushed trial's container (`sleep infinity`) and any mini-swe-agent process
  inside it keep running. Only a manual `docker compose down` reclaims them. On
  Daytona the auto-stop and auto-delete timers do. A client that disconnects
  (for example on `AGENT_TRIAL_TIMEOUT`) cancels nothing: `run_instance` does not
  watch for disconnects, so the trial runs to completion and its result is
  discarded.
- **Harbor in-process** (`harbor_agent_function.py:445`): `build_trial_config`
  resolves `HARBOR_TASKS_DIR/<instance_id>` and the harness binding (terminus-2,
  mini-swe-agent, claude-code, opencode; `HARNESS_BINDINGS`, `:256`), then
  `Trial.create` + `Trial.run()` under `asyncio.wait_for(AGENT_TRIAL_TIMEOUT)`.
  Harbor builds the environment, runs the agent, runs `test.sh`, and deletes the
  environment (`EnvironmentConfig(delete=True)`, `:338`). On Daytona it also sets
  auto-stop and auto-delete timers so a killed worker's sandbox is eventually
  reclaimed (`:308-320`). `trial_result_to_metadata` (`:395`) maps the verdict
  to `{reward, exit_status, eval_report, agent_metrics}`.
- **OpenEnv per-episode** (`openenv_sandbox_common.py`): `start_task_sandbox`
  throttles creates with a per-rollout `rollout_semaphore` shared by the rollout's
  agent processes (`OPENENV_<P>_CREATE_CONCURRENCY`, default 4) and retries only
  provider-classified throttle errors with jittered exponential backoff (default 8
  retries). `create_once` hands a sandbox that
  finishes creating after its episode was cancelled to a reaper, so wall-clock
  cancels do not leak sandboxes. `episode_env` closes the sandbox in `finally`.
- **OpenEnv episode** (`openenv_agent_function.py:234`): `reset(task_id)` →
  up to `OPENENV_MAX_TURNS` (30) rounds of (chat completion → strip a fenced bash
  block → `exec` → append output as a `user` message) → `evaluate` (the task's own
  `tests/test.sh`). With `TB2_WITHHOLD_TESTS=1` the server reads `tests/` into
  memory at reset, deletes it before the agent's first action, and restages it only
  for the verify window (`tb2_sandbox_recipe.py:26-36`).
- **TMax** (`examples/tmax/tmax_sandbox.py`): one container per episode from the
  task's prebuilt image, started with up to 5 attempts and removed when the episode
  exits; `tests/` is uploaded only on submit.
  [TMax Terminal-RL](/examples/tmax) has the full episode.

### 4.3 Concurrency, from outer to inner

1. Submission pacing: about `over_sampling_batch_size × n_samples_per_prompt`
   episodes in flight in sync mode; `--async-max-concurrent-samples` in fully-async
   mode (`fully_async_rollout.py`, `_max_in_flight_groups`).
2. `generate_fn_semaphore`: `sglang_server_concurrency × num_engines` (512 per engine
   by default), held per episode.
3. The connector's own cap. For the Harbor agent server this is `--max-concurrent`
   (default 8). It becomes an `asyncio.Semaphore` held around the whole trial
   subprocess (`harbor/miles_agent_server.py:79-82,213-214,417`). The README
   says to set it to at least `rollout_batch_size × n_samples_per_prompt`. Excess
   requests **queue without limit**; there is no 429 or 503 backpressure. Queue
   time counts against the client's `AGENT_TRIAL_TIMEOUT`. The client side adds no
   cap: `swe_agent_function`'s `httpx` pool is unbounded unless
   `AGENT_SERVER_MAX_CONNECTIONS` is set (`swe_agent_function.py:37-54`). Other
   connectors: OpenEnv `MAX_CONCURRENT_ENVS` with a 30-minute capacity wait
   (`_CAPACITY_MAX_WAIT_S`, `openenv_agent_function.py:73`), provider create
   semaphores, TMax's per-node start, exec, and live-sandbox limits, Daytona disk
   quota. Under `subproc` a module-level `asyncio.Semaphore` in an agent module
   limits nothing, since every call has its own process; these caps are `flock`
   slots shared by a rollout process and its agent processes, mostly through
   `rollout_semaphore` / `rollout_lock` (`miles/rollout/agentic/rollout_limits.py:31,43`).

There is no Miles-side pool of warm containers comparable to open-instruct's
`EnvironmentPool`; every episode creates and tears down its own sandbox.

### 4.4 Timeouts

| Timeout | Default | Where |
|---|---|---|
| Harbor `--agent-timeout` (agent server) | task's `agent.timeout_sec`; unbounded if neither is set | **agent phase only**, and **multiplied by `HARBOR_TIMEOUT_MULTIPLIER` (default 2.0)**: `--agent-timeout 5400` gives a 10800 s cap (`harbor/agent_server/trial_runner.py:392,400`, `harbor/src/harbor/trial/trial.py:409-419,1062-1073`). On expiry the verifier still runs |
| Harbor environment start | task `build_timeout_sec` 600 s × multiplier | build + pull + `compose up` (`trial.py:1174-1185`); scale with `HARBOR_ENV_BUILD_TIMEOUT_MULTIPLIER` |
| Harbor agent setup | 360 s × multiplier | in-container agent install (`trial.py:81,1085-1094`); `--agent-setup-timeout` overrides the base |
| Harbor verifier | task `verifier.timeout_sec` 600 s × multiplier | `HARBOR_VERIFIER_TIMEOUT_SEC` replaces the base and is **still multiplied** (`trial_runner.py:370-375`, `trial.py:1075-1083`) |
| Harbor per command | mini-swe-agent's own config | not set by Harbor; terminus-2 uses the model's `duration_sec` and returns a timeout message as the observation (`harbor/src/harbor/agents/terminus_2/terminus_2.py:1273-1288`) |
| Harbor heartbeat | 120 s × 3 misses | flushes a dead session server's trials (`agent_server/background.py:29-31`) |
| `AGENT_TRIAL_TIMEOUT` (client) | 7200 s | `swe_agent_function.py:27`, `harbor_agent_function.py:81`. It covers semaphore queue time + environment start + setup + agent + verifier, so it must exceed their sum, not just `--agent-timeout` |
| `OPENENV_MAX_ROLLOUT_TIME_SECONDS` | 3600 s | whole-episode cap, `openenv_agent_function.py:111` |
| `OPENENV_MESSAGE_TIMEOUT_S` | 1200 s | per env op (reset, exec, evaluate) |
| `TB2_COMMAND_TIMEOUT_S` | recipe | per exec inside the sandbox |
| Session collect / delete | 120 s | `openai_endpoint_utils.py:24` |
| Session-server → router | `--miles-router-timeout` | `session/config.py` |
| Agent process cleanup (`subproc`) | 120 s for a cancelled agent's own cleanup; 600 s for leftover threads once the call is over | then the agent process exits; after a cancel the rollout side also SIGKILLs it at their sum (`agent_function.py:33-35,95-98`) |

A synchronous step waits for its slowest episode, so the episode caps are also your
worst-case step time. At the README's `--agent-timeout 5400` with default
multipliers, one agent-server trial can run
1200 + 720 + 10800 + 1200 s, about 3.9 h, plus queue time. The client gives up at
7200 s, so any episode longer than about two hours is ended by the client
timeout, not the server's.

### 4.5 What infra failure does to a sample

This is the part that differs most from what the comments in the examples say.
Tracing each case through the code:

**What the Harbor agent server returns.** Nearly every failure comes back as
HTTP 200 with `reward: 0.0` and a named `exit_status`. The exceptions are the
trials where the verifier still ran. `exit_status` comes from
`_extract_exit_status` (`harbor/agent_server/results.py:23-35`):
`AgentTimeoutError`, `VerifierTimeoutError`, and `EnvironmentStartTimeoutError`
map to `TimeLimitExceeded`; the two max-seq-len errors map to
`SequenceLengthLimitExceeded`; any other recorded exception maps to `AgentError`;
a verdict with no exception maps to `Submitted`. The reward is the verifier's,
whenever a verifier result exists, **regardless of `exit_status`**
(`results.py:68-79`). Miles never reads `exit_status`: no built-in reward
function or filter looks at it.

| Harbor-side failure | `exit_status` | `reward` | Verifier ran | Model call recorded? | Miles outcome |
|---|---|---|---|---|---|
| Bad or missing `instance_id` / task dir (`trial_runner.py:329-349`) | `InvalidInstanceId` / `TaskNotFound` | 0.0 | no | no | `ABORTED` → group dropped |
| Bad `session_server_id` (`miles_agent_server.py:124-163`) | `InvalidSessionServerId` | 0.0 | no | no | group dropped |
| Image pull or build fails, `compose up` fails (`docker.py:704-711` raises `RuntimeError`) | `AgentError` | 0.0 | no | no | group dropped |
| Environment start exceeds its timeout | `TimeLimitExceeded` | 0.0 | no | no | group dropped |
| Agent install fails or exceeds setup timeout (`AgentSetupTimeoutError` is not in the timeout set) | `AgentError` | 0.0 | no | no | group dropped |
| Exception before `trial.run()` (config error, Harbor import error) (`trial_runner.py:318-322,426-428`) | `Error: <Type>` / `ImportError` | 0.0 | no | no | group dropped |
| Agent exceeds the agent timeout (`single_step.py:83-84`) | `TimeLimitExceeded` | **the verdict** (can be 1.0) | **yes** | yes | trained with the verdict |
| Agent exits non-zero (crash, OOM, or API errors such as a 400 context-too-long or a 404 from a deleted session) (`single_step.py:83-84`) | `AgentError` | **the verdict** | **yes** | usually | trained with the verdict |
| mini-swe-agent `max_seq_len` poll trips, terminus-2 `Chat` `max_seq_len`, or terminus-2 single-turn length abort | `SequenceLengthLimitExceeded` | 0.0 | **no** (not caught in `_run_agent`) | yes | trained as a 0 |
| Verifier exceeds its timeout (`single_step.py:109-112`) | `TimeLimitExceeded` | 0.0 | started, no verdict | yes | trained as a 0 |
| Verifier crashes: tests upload fails, no `reward.txt` or `reward.json`, empty or unparsable file (`harbor/src/harbor/verifier/verifier.py:66-94,227-236`) | `AgentError` | 0.0 | started, no verdict | yes | trained as a 0 |
| Trial worker dies without printing a result (`trial_runner.py:477-481`) | `Error: NoResult` | 0.0 | unknown | maybe | 0 if records, else dropped |
| `/flush`, `/flush_all`, or heartbeat flush (`miles_agent_server.py:215-222`) | `Flushed` | 0.0 | no | maybe | abort-hook flushes hit groups that are discarded anyway; a heartbeat flush means the session server is gone, so collect usually fails → `ABORTED` |
| Server overloaded (`--max-concurrent` full) | none; the request waits on the semaphore | none | n/a | n/a | only the client timeout ends the wait |

**What the Miles side then does with it** (`swe_agent_function.run` →
`agentic_tool_call.generate`):

| Failure | Agent function returns | Sample outcome | Group outcome |
|---|---|---|---|
| Any Harbor-side failure above | the four keys, with the `reward` above (`swe_agent_function.py:139-144`) | reward as returned | dropped if no model call was recorded, otherwise trained with that reward (0 or the verdict) |
| Harbor agent server unreachable / HTTP error / client `AGENT_TRIAL_TIMEOUT` (`swe_agent_function.py:124-137`) | `None` | records exist → status from last turn, `metadata` has no `reward` → `reward_func` returns `0.0` (`generate.py:39`) | **trained as a 0**. The server keeps running the trial against a now-deleted session. If the timeout fired while the request was still queued on the semaphore or the sandbox was still building, there are no records → `ABORTED` → **group dropped** |
| `CancelledError` inside `run` (`swe_agent_function.py:132-134`) | `inline`: `None` (the cancellation is swallowed). `subproc`: the rollout side re-raises it and SIGTERMs the agent process (`agent_function.py:74-76`) | collect still runs; under `inline` → as above, under `subproc` the cancellation then propagates | the server-side trial keeps running and holds its slot |
| Harbor trial raises, or exceeds the cap (in-process, `harbor_agent_function.py:472-478`) | `{"reward": 0.0, "exit_status": ...}` | reward 0 | trained as a 0. A Harbor-internal agent timeout is not this row: `Trial.run` returns the verdict (`trial_result_to_metadata`, `:395`), exactly as on the agent server |
| Harbor trial fails before the first model call (sandbox never started, agent never installed, task missing) | either of the above | no records → `ABORTED` (`agentic_tool_call.py:126-135`) | **whole group dropped** by `apply_aborted_filter` (`common_filters.py:82`) |
| OpenEnv: no canonical verdict, or episode exception (`openenv_agent_function.py:442-454`) | `None` ("dropping sample") | `reward_func` default → `0.0` (`openenv_generate.py:15`) | **trained as a 0**, not dropped |
| OpenEnv: episode wall-clock cap | `{"reward": 0.0}` | reward 0 | trained as a 0 (intended) |
| Agent function raises, or its process dies (any connector) | nothing; caught at `agentic_tool_call.py:98` | records → reward via RM (usually 0); no records → `ABORTED` | 0 or dropped |
| Session collect times out / transport error | n/a | `ABORTED` | dropped |
| Session collect 422 (TITO assembly assertion) | n/a | `RuntimeError` out of generate | sync: group logged and dropped (`inference_rollout_train.py:142-144`); **fully-async: the rollout worker dies and the run fails** (`fully_async_rollout.py:187-189,202-206`) |
| `multi_turn` `execute_tool` raises | n/a | exception | siblings cancelled (`inference_rollout_common.py:176-178`); same sync/async split as above |

Rules that fall out of this:

- **Dropping is group-granular.** A GRPO group must have exactly
  `n_samples_per_prompt` samples (`inference_rollout_train.py:156`), so any
  `ABORTED` sample, or any sample whose reward is `None`
  (`apply_missing_reward_filter`, `common_filters.py:89`), discards all its
  siblings. There is no "drop one sample, renormalize over the rest".
- **Where infra failures land, on the Harbor agent-server path.** Sandbox-side
  failures (image pull or build, container start, agent install, missing task)
  happen before the first model call, so they **drop the group**; they do not
  train as 0. Failures after the first model call train as 0: verifier timeout or
  crash, client timeout, session-server or transport errors. Agent-side failures
  (agent timeout, non-zero agent exit) are **graded normally**, because Harbor
  still runs the verifier.
- **To actually drop an infra failure,** make your reward function return `None`
  for it. The missing-reward filter then discards the group. When the agent
  function returned `None` (client-side failures),
  `sample.metadata.get("reward")` without a default does this, as
  `examples/tmax/generate.py:21` does. It does **not**
  catch Harbor-side failures: the server already turned them into
  `reward: 0.0`. For those, key on the metadata the server returns. An empty
  `eval_report` with an `exit_status` other than `SequenceLengthLimitExceeded`
  means no verdict for a non-policy reason. The shipped reward functions default
  to `0.0`, which turns every infra failure after the first model call into a
  false negative.
- **In sync mode, discarded prompts are gone for the epoch.** Nothing is retried;
  the loop just pulls more prompts until `rollout_batch_size` groups pass the
  filters. In fully-async mode, `--async-unused-samples-handler retry` resets
  aborted and stale groups and puts their prompts back (`_recycle`,
  `fully_async_rollout.py:258`); missing-reward and dynamic-filter rejections are
  discarded either way.
- **A systematic failure spins instead of stopping.** If every episode aborts
  before any model call, sync rollout keeps submitting and dropping groups with
  warnings and never completes the step. Examples: a bad `HARBOR_TASKS_DIR` (the
  agent server answers `TaskNotFound`; the in-process path raises), a dead
  registry, or a sandbox that cannot reach the network to install the agent.

## 5. Reward and filtering

### 5.1 Where rewards come from

`async_rm` (`rm_hub/__init__.py:42`) resolves, in order, a per-sample
`reward_spec`, then `metadata["rm_type"]`, then `--custom-rm-path` / `--rm-type`
(`_resolve_reward_config`, `:31`). Built-ins are rule-based (`math`, `dapo`,
`deepscaler`, `gemma_math`, `f1`, `gpqa`, `ifbench`, `random`,
`deterministic_random`, `boxed_*`) plus `remote_rm`, which POSTs
`{prompt, response, label}` to `--rm-url` (`:18`).

For environments, the environment grades the episode and the reward rides in
metadata: the agent function returns `{"reward": r, ...}`, v1 merges it into
`sample.metadata`, and a two-line `--custom-rm-path` reads it back
(`examples/swe-agent-harbor-docker/generate.py:32`). The same function has to accept
a list, because `batched_async_rm` calls the custom RM with the whole list
(`rm_hub/__init__.py:103-105`). Under v2, a non-`None` `reward` in the agent dict is
assigned server-side to every kept leaf as `Sample.reward`, so the RM step skips
those samples; a `None` one leaves the RM to run.

**How Harbor grades.** Harbor's `Verifier.verify` uploads the task's `tests/` into
the sandbox and runs the test script, writing its stdout to
`/logs/verifier/`. It then reads `/logs/verifier/reward.json` if present (a flat
JSON object), else `reward.txt` (one float)
(`harbor/src/harbor/verifier/verifier.py:138-238`). **The script's exit code is
ignored.** The scalar is `rewards["reward"]` when that key exists, else the first
value (`harbor/agent_server/results.py:68-79`), so a Harbor reward is
**whatever float the task writes**. It is binary only if the task's script makes
it so. Harbor's own `adapters/terminal_tasks/template/test.sh:11-38`, for example,
writes a weighted sum of passing tests. The whole dict comes back as
`eval_report`. A missing,
empty, or unparsable reward file is a verifier exception, which scores 0.

When rewards are computed:

- default: per sample, right after its generate returns, inside the semaphore
  window (`inference_rollout_common.py:135`), so sandboxed episodes and reward
  calls overlap;
- `--group-rm`: once per group after all siblings finish (`:186`), for rewards that
  need the whole group (pairwise judges, ranking). Not supported with session v2;
- a generate function that sets `sample.reward` itself skips the RM
  (`inference_rollout_common.py:128,135`); `batched_async_rm` asserts it never
  overwrites a set reward (`rm_hub/__init__.py:97`).

Reward values can be dicts; `--reward-key` picks the scalar
(`Sample.get_reward_value`).

### 5.2 Filters, in order

Per group, as soon as it finishes (`apply_preput_filters`, `common_filters.py:70`):

1. `apply_aborted_filter`: any `ABORTED` sample → drop (`group_has_aborted`).
2. `apply_missing_reward_filter`: any `None` reward → drop.
3. `--dynamic-sampling-filter-path`: DAPO-style. The stock
   `apply_reward_nonzero_std_filter` (`common_filters.py:95`) drops zero-variance
   groups. `check_no_aborted` and `check_reward_nonzero_std` in
   `dynamic_sampling_filters.py` are deprecated aliases; note that the Harbor
   launcher passes `check_no_aborted` (`run.py:199`), which duplicates step 1 and
   does **not** filter zero-variance groups. Terminal-RL groups are often all-0 or
   all-1, so pass `apply_reward_nonzero_std_filter` if you want active sampling.

After the batch is full: `--rollout-sample-filter-path` (in place, sets
`remove_sample`; `inference_rollout_train.py:195`), then
`--rollout-all-samples-process-path` over everything including dropped groups,
then optional prefill logprob recompute (`--recompute-logprobs-via-prefill`,
`:204`).

### 5.3 From rewards to advantages

`_post_process_rewards` (`ray/rollout/train_data_conversion.py:344`) keeps
`raw_reward` and, for `grpo`/`gspo`/`reinforce_plus_plus_baseline` (unless
`--disable-rewards-normalization`), calls `_normalize_rewards_by_rollout` (`:294`): within
each prompt group (by `group_index`), samples that share a `rollout_id` are one
rollout and must share one reward (`:323`), the group mean is subtracted, and GRPO/GSPO
divide by the std unless `--disable-grpo-std-normalization` is passed. This is how v2 leaves and
compaction siblings (several samples, one episode) get one advantage instead of
being counted as extra rollouts. `--custom-reward-post-process-path` replaces all of
it.

### 5.4 Fully async

`--fully-async` swaps in `FullyAsyncRolloutFn` (`fully_async_rollout.py:63`): a
persistent producer keeps groups in flight (per-sample backfill by default), and a
`DefaultDataBuffer` applies the same three pre-put filters on arrival
(`fully_async_data_buffer.py:190`) and drops groups older than
`--max-weight-staleness` versions on consumption (`:244`), using per-token
`weight_versions`. A single trajectory can span weight updates; its spans record
which version produced which tokens.

## 6. Walkthrough: one Terminal-Bench episode in `swe-agent-harbor-docker`

The launcher (`examples/swe-agent-harbor-docker/run.py`) at the documented shape:
GLM-4.7-Flash, 1 node × 8 GPUs colocated, `--rollout-batch-size 4
--n-samples-per-prompt 8` (32 trajectories per step), `--rollout-max-response-len
8192` (per turn), `--max-seq-len 65536` (per trajectory), `--sglang-context-length
73728` (their sum, `run.py:111-114`), `--tito-model glm47`,
`--sglang-tool-call-parser glm47 --sglang-reasoning-parser glm45`, 32 session
servers (v1) on ports 30000-30031, router on 31000, agent server with
`--max-concurrent 32`.

**0. Data.** A JSONL row is `{"prompt": [...messages...], "metadata":
{"instance_id": "<tb2-task>", "agent_name": "mini-swe-agent", ...}}`
(`download_and_process_data.py`). `--input-key prompt --metadata-key metadata`; no
`--apply-chat-template`.

**1. Rollout step starts.** `RolloutFn._call_train` (`generate.py:96`, a thin
`InferenceRolloutFn` subclass that adds `agent/*` metrics) calls
`generate_rollout_async`, which pulls 4 groups of 8 identical samples and starts 32
`generate_and_rm` tasks.

**2. Session.** Each task's `agentic_tool_call.generate` opens a session on one of
the 32 servers and calls `swe_agent_function.run` (`swe_agent_function.py:82`), in
an agent process of its own, with `base_url=http://<external host>:300xx/sessions/<id>`.

**3. Hand-off to Harbor.** `run` appends `/v1` (`openai_session_url`) and POSTs
`{**metadata, base_url: <session>/v1, model: "openai/model",
sampling_params: {temperature: 0.8, max_tokens: 8192, ...}, max_seq_len: 65536}` to
`AGENT_SERVER_URL/run` (`swe_agent_function.py:126`) with a 7200 s ceiling and TCP
keepalive. It also sends `session_server_id` (the `host:port` of `base_url`) and
`session_server_instance_id`, which the agent server uses to heartbeat that
session server and to scope `/flush`.

**4. Sandbox and agent (in Harbor).** The request waits for a `--max-concurrent`
slot. A per-trial subprocess then builds (or pulls) and starts the task's compose
project from `HARBOR_TASKS_DIR/<instance_id>`. It installs mini-swe-agent
**inside** the container and runs it there with `OPENAI_API_BASE` set to the
session URL and model `openai/model` (section 4.2). The `sampling_params` from
step 3 are dropped for mini-swe-agent (`harbor/agent_server/trial_runner.py:136-140`).
Temperature still comes from the session's defaults, but no `max_tokens` is sent
unless mini-swe-agent's own config sets one. Each agent step is an ordinary
`/v1/chat/completions` call from the container, carrying the full message history
and, for native tool calling, `tools`.

**5. Turn 1 in the session server.** No checkpoint yet, so the GLM-4.7 template
renders system + user + tools into `input_ids`. The server forces
`logprobs`/`return_meta_info`, forwards to the router, and SGLang generates. The
8192-token per-turn cap from `--rollout-max-response-len` applies only if the
harness sends `max_tokens`. terminus-2 does, through `llm_call_kwargs`;
mini-swe-agent, per step 4, does not. The `glm45`/`glm47` parsers split
`reasoning_content` and `tool_calls`, and the server commits
`input_ids + output_ids` as checkpoint 0 and appends record 0. The agent sees a
normal OpenAI response (no `meta_info`, and no `logprobs` unless it asked).

**6. Tool call.** mini-swe-agent runs the bash command in the container through its
local environment. Harbor's install-time patch caps each command's output at 1 MiB
head+tail (`harbor/src/harbor/agents/installed/mini_swe_agent.py:58-89`).
mini-swe-agent then sends turn 2:
`[..., assistant (echoed verbatim), tool or user (command output)]`.

**7. Turn 2.** The matcher confirms the stored three messages prefix the request;
`GLM47TITOTokenizer.merge_tokens` drops the trailing `<|observation|>` stop token
from checkpoint 0 (the template renders it as the next message opener) and appends
the tokens of the new message plus the `<|assistant|>` opener. No model token is
re-tokenized.

**8. The episode ends** in one of four ways:

- The agent submits or hits its own step limit. The verifier runs.
- The agent exceeds the agent timeout (`--agent-timeout` × 2.0) or exits
  non-zero. The verifier **still** runs and its verdict is the reward.
- The Harbor-side `max_seq_len` poll trips, when the last call's
  prompt + completion tokens exceed 65536. The trial ends with
  `SequenceLengthLimitExceeded` and **no verifier run**, so the reward is 0.
- The client's 7200 s ceiling fires first (section 4.5).

The verifier uploads `tests/`, runs the test script, and reads
`/logs/verifier/reward.txt` or `reward.json`. The container is then removed with
`docker compose down`. The server returns exactly
`{reward, exit_status, eval_report, agent_metrics}` (`RunResponse`,
`harbor/agent_server/models.py:22-26`), and `run` returns those four keys. No
tokens come back from Harbor; the training tokens are the session server's
records.

**9. Collect.** The session server builds one sample per turn, trims each
intermediate turn's `<|observation|>` (`max_trim_tokens=1`), truncates at 65536
tokens on a turn boundary, and merges: `tokens` is the whole conversation, the first
turn's `input_ids` are the prompt, model tokens carry SGLang's logprobs and
`loss_mask=1`, every tool output and turn opener is `loss_mask=0`, logprob 0.0. The
session is deleted.

**10. Reward and filtering.** v1 merges the agent's dict into `sample.metadata`;
`generate.reward_func` returns `metadata["reward"]`. When all 8 siblings are done,
the group passes `apply_aborted_filter` and the missing-reward filter (always, since
the RM defaults to 0.0), then `check_no_aborted` again. `exit_status` is not
consulted, so a `SequenceLengthLimitExceeded` or `TimeLimitExceeded` trial trains
like any other. Once four groups are kept, the remaining in-flight work is
aborted: the engines, plus `/flush` on the agent server. That kills the
over-sampled trials' workers, but on Docker it leaves their containers running
(section 4.2).

**11. Training.** Rewards are normalized within each group of 8, R3 routes (if
enabled) and rollout logprobs travel with the samples, and Megatron trains with
GRPO + low-var KL as configured in `run.py`.

## 7. Concept map: open-instruct → Miles

| open-instruct | Miles | Notes |
|---|---|---|
| `vllm_utils.process_request` (token loop) | `generate_hub/multi_turn.py` for in-process tools; session server + agent function for sandboxed agents | Miles' main agentic path is message-level with server-side token recording |
| `rlvr_tokenize_v1` (`apply_chat_template` once, with tools) | `compute_prompt_ids_from_sample` (multi_turn) or TITO first-turn render (session) | Both happen at rollout time, not preprocessing |
| `tool_parser.get_tool_calls` (vLLM parsers) | SGLang `FunctionCallParser` (`--generate-tool-call-parser`) or engine-side `--sglang-tool-call-parser` | Session path: the agent gets structured `tool_calls` |
| `format_tool_outputs` role templates (`VLLM_PARSERS`) | template suffix diff (`_tokenize_postfix_messages`, `TITOTokenizer.tokenize_additional_messages`) | Derived from the chat template; per-family boundary fixes live in `tito_tokenizer.py` |
| `--mask_tool_use` | always on | Injected tokens are always `loss_mask=0`; there is no flag to train on them |
| `response_masks` / `response_logprobs` (0.0 on tools) | `Sample.loss_mask` / `Sample.rollout_log_probs` (0.0 on injected) | Same convention |
| `--max_steps` | `--generate-max-turns` (multi_turn); harness-owned on the session path | No Miles-level step cap for agents |
| `--per_turn_max_tokens` | `--rollout-max-response-len` | In Miles this is always per turn on multi-turn paths |
| `--response_length` | `--max-seq-len` (session, total incl. prompt) or `--rollout-max-context-len` (multi_turn) | Session truncation happens after the episode, at a turn boundary |
| observation truncation to fit context | agent's job | Over-long prompts return 400 to the agent, unrecorded |
| `tool_call_timeout` | agent's job | |
| `tool_call_format_error_feedback` | agent's job | |
| `EnvironmentPool` + `SWERLVanilluxSandboxEnv` actors | connector-owned sandboxes (Harbor, OpenEnv backends, `examples/tmax`) | No warm pool; per-episode create/teardown |
| podman / docker backends, host rotation | `examples/tmax`: Podman (or Docker) API sockets, one picked per episode, failing daemons benched; Harbor agent server: `docker` or `daytona` only; in-process Harbor: E2B / Daytona / Modal / ... via `HARBOR_ENV_TYPE`; OpenEnv backends | No podman backend in Miles core or Harbor; see gap list |
| `SWERL_RESET_FAILURE_ZERO_REWARD=1` | the default behavior once a model call happened | Sandbox create or agent-install failure happens before any model call, so it drops the group instead. Agent timeout and agent crash are graded by the verifier. TMax keeps a never-started episode in its group by making a one-token model call first |
| `rollout_state["rewards"]` + `LastRewardAggregator` | scalar `metadata["reward"]` read by `--custom-rm-path` | One reward per episode; no per-step rewards |
| `DataPreparationActor` filtering / active sampling | `apply_preput_filters` + `--dynamic-sampling-filter-path` + `--over-sampling-batch-size` | |
| truncated = `finish_reason != stop or len >= response_length` | `Sample.Status.TRUNCATED` (any turn hit `length`, or collect truncation) | Not masked by default in either |
| `--async_steps` | `--fully-async` + `--max-weight-staleness` | Staleness is per token span |
| `cache_salt=base_request_id` (prefix cache per group) | `X-SMG-Routing-Key` = session id (per episode) | Siblings are not pinned together |
| `--use_vllm_logprobs` | SGLang `output_token_logprobs` in `rollout_log_probs`; `--use-tis`, `--recompute-logprobs-via-prefill` | |
| `pool.release` in `finally` | connector `finally` (Harbor `Trial.run` shielded stop; `episode_env`) + agent `abort` hook | The agent server's Docker stop is `compose down`, which keeps volumes and the image unless `HARBOR_DELETE_CONTAINERS=true`; a flushed trial is SIGKILLed and never stops its container |

## 8. Porting Terminal RL (bash tool, per-task container, Harbor tasks)

### 8.1 Recommended shape

Keep open-instruct's control over the loop, but let the session server own the
tokens. `examples/tmax` is this port; [TMax Terminal-RL](/examples/tmax) describes
its episode, failure handling, and launch. Its agent function, condensed from
`examples/tmax/tmax_agent_function.py:127-254`:

```python
async def run(base_url, prompt, request_kwargs, metadata, **kwargs):
    policy = AsyncOpenAI(base_url=openai_session_url(base_url), api_key="EMPTY")
    messages = list(prompt)
    async with TmaxSandbox(image=metadata["image"], task_dir=...) as box:     # fresh container, removed on exit
        while tool_calls < MAX_STEPS:
            resp = await policy.chat.completions.create(
                model="model", messages=messages, tools=[BASH_TOOL],
                max_tokens=min(per_turn_cap, max_seq_len - last_total_tokens), extra_body=request_kwargs)
            choice = resp.choices[0]
            messages.append(choice.message.model_dump(exclude_none=True))     # echo verbatim
            if choice.finish_reason == "length" or not bash_calls(choice):
                return {"reward": 0.0}
            for tc in bash_calls(choice):
                out, submitted = await box.bash(command(tc))   # 120 s; infra errors become the observation
                if submitted:
                    return {"reward": await box.run_tests()}   # upload tests/ only now; read reward.txt
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    # a sandbox that never starts scores 0 through a one-token turn; an aborted episode returns reward None
```

The launcher (`examples/tmax/run_tmax_qwen3_5.py:307-328,373-379`) wires it with
`--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
--custom-agent-function-path tmax_agent_function.run --use-session-server v2
--tito-model qwen35 --custom-rm-path generate.reward_func
--dynamic-sampling-filter-path
miles.rollout.filter_hub.common_filters.apply_reward_nonzero_std_filter
--max-seq-len 34816 --rollout-max-response-len 8192 --sglang-reasoning-parser qwen3
--sglang-tool-call-parser qwen3_coder`. The parser flags are required: `--tito-model`
does not set them (section 3.2). Each episode runs in its own agent process, so
limits shared across episodes are `rollout_limits` locks, and the module's
`abort(args)` reaches in-flight episodes through a file and by removing the
rollout's containers (section 3.7); without it, over-sampled episodes keep running.

Alternatively, run Harbor tasks directly: `examples/swe-agent-harbor-docker` for a
local Docker daemon, or the in-process recipe for a cloud provider. Both give you
Harbor's harnesses (terminus-2, mini-swe-agent) instead of your own loop.

### 8.2 Gaps

1. **No podman backend in Miles core or in Harbor.** The in-tree one is an
   example: `examples/tmax/tmax_sandbox.py` drives Podman through docker-py on its
   Docker-compatible API socket. For Harbor tasks, the only local option is the
   separate Harbor agent server. Its backends are `docker`
   and `daytona` only (`harbor/agent_server/trial_runner.py:192-242`). The Docker
   backend shells out to the `docker` CLI: `docker compose` for
   build/up/exec/down, and `docker buildx prune` in the prune loop
   (`harbor/src/harbor/environments/docker/docker.py:655-714`,
   `harbor/agent_server/background.py:132-143`). Startup exits unless
   `~/.docker/config.json` has a Docker Hub `auths` entry
   (`harbor/agent_server/docker_ops.py:89`). A podman host therefore needs a
   Docker-compatible CLI with compose and buildx, plus that login entry. Nothing
   in either repo tests that setup. In-process Harbor expects a remote provider.
2. **No warm container pool.** Per-episode create/teardown only,
   with a fresh in-container agent install on every Harbor trial. Harbor tries
   environment start once (`harbor/src/harbor/trial/trial.py:1174-1185`) and the
   agent server never retries a trial. Retries exist only for provider throttling
   in the OpenEnv backends and for container start in TMax (5 attempts). The
   eight siblings of a group all start the same task at once, and Harbor's
   per-image build lock does not span its per-trial processes, so a non-prebuilt
   task image is built concurrently by every sibling.
3. **No first-class infra-drop.** The agent contract has no "discard" return; the
   shipped reward functions turn infra failures into 0-reward samples. Return
   `None` from your RM for infra failures, and expect the whole group to go. On
   the agent-server path, Harbor-side failures already arrive as
   `reward: 0.0`, so detect them from `exit_status` and an empty `eval_report`
   (section 4.5). A per-sample drop without losing siblings does not exist.
4. **No Miles-level step or observation budget on the session path.** Max steps,
   per-command timeout, output truncation, and "context almost full" handling are
   all agent code; `--max-seq-len` only truncates the recorded sample afterwards.
5. **A length-truncated turn ends the training sample** while the episode (and its
   reward) continues. Stop the episode on `finish_reason == "length"`, as OpenEnv,
   TMax, and terminus-2 (abort policy) do. With Harbor, stopping this way, or tripping
   `max_seq_len`, skips the verifier and scores 0. Harbor's comment that this is
   "filtered by dynamic sampling" does not hold in Miles, where no filter reads
   `exit_status`.
6. **Per-step rewards are not supported.** One scalar per episode (or per v2 leaf).
7. **`multi_turn.generate` cannot drive per-episode sandboxes**, because
   `execute_tool(name, params)` gets no sample context; forking it is the
   alternative to the agent-function route.
8. **Fully-async is less forgiving:** any exception escaping a generate function
   (for example a 422 from TITO assembly) kills the producer and the run.
9. **Template families are an allowlist.** The named families in
   `TITOTokenizerType` (`tito_tokenizer.py:1072`; Qwen3/3.5/3.6/Next, GLM-4.7/5.3,
   Nemotron-3, Kimi, MiniMax, DeepSeek, ...) are registered; anything else runs as
   `default` (best-effort) until it passes
   `scripts/tools/verify_chat_template.py` and
   `scripts/tools/verify_session_tito_tokenizer.py`.
10. **The Harbor agent server leaks Docker containers on flush.** `/flush` (the
    Miles abort hook, after every over-sampled sync step) and the heartbeat
    SIGKILL the trial worker. The trial's `compose down` never runs, and the
    project was never registered for `/flush_all` (section 4.2). Plan a periodic
    sweep of stale `docker compose ls` projects, or run with Daytona.
11. **mini-swe-agent through Harbor gets no per-turn token cap from Miles.**
    `sampling_params` is dropped for it
    (`harbor/agent_server/trial_runner.py:136-140`), so
    `--rollout-max-response-len` does not reach the engine on this path. Set
    `max_tokens` in the mini-swe-agent config, or use terminus-2, which forwards
    `llm_call_kwargs`.
12. **Harbor timeouts are multiplied.** `HARBOR_TIMEOUT_MULTIPLIER` (default 2.0)
    scales `--agent-timeout`, the setup and build timeouts, and
    `HARBOR_VERIFIER_TIMEOUT_SEC`. The client's `AGENT_TRIAL_TIMEOUT` has to
    cover queue time plus all of them, or long episodes end as client timeouts
    (reward 0) while their sandboxes keep running.
13. **No backpressure from the agent server.** Past `--max-concurrent`, `/run`
    waits on the semaphore indefinitely. The first sign of an undersized server
    is client timeouts, and groups dropped because their samples never made a
    model call.

## 9. Corrections to existing docs and comments

- `examples/swe-agent-harbor-docker/README.md` (and its generated
  `docs/examples/swe-agent-harbor-docker.md`) says a client-side
  `AGENT_TRIAL_TIMEOUT` makes "the aborted sample take its whole GRPO group down".
  In the code the timeout returns `None` (`swe_agent_function.py:129-131`), the
  recorded turns are collected normally, and `reward_func` scores the sample 0.0.
  The group is dropped only if no model call was recorded. That happens when the
  timeout fires while the request is still queued on the agent server's
  semaphore, or while the sandbox is still building or installing the agent. The
  README's other half is right: the agent server does not notice the disconnect
  and keeps the trial and its slot until the trial ends on its own.
- The same README calls `--agent-timeout` "the authoritative one" and says
  `AGENT_TRIAL_TIMEOUT` only has to stay above it. Harbor multiplies
  `--agent-timeout` by `HARBOR_TIMEOUT_MULTIPLIER` (default 2.0;
  `harbor/agent_server/trial_runner.py:400`, `harbor/src/harbor/trial/trial.py:409-419`).
  The client ceiling also covers queue, build, setup, and verifier time. The
  README's own example, `--agent-timeout 5400` with the default
  `AGENT_TRIAL_TIMEOUT=7200`, is therefore inverted: the effective agent cap is
  10800 s, so the client times out first.
- `examples/swe-agent-harbor-docker/swe_agent_function.py:148-155` says the abort
  hook's flush makes the server "release their containers". The server kills the
  trial worker, which skips Harbor's teardown, so on Docker the containers stay up
  (section 4.2).
- In Harbor itself: the terminus-2 abort path says the truncated trajectory is
  "filtered by dynamic sampling" (`harbor/src/harbor/agents/terminus_2/terminus_2.py`,
  the `policy in ("abort", "regenerate")` branch). In Miles it is trained as a
  0-reward sample. `trial_runner.py:370` calls `HARBOR_VERIFIER_TIMEOUT_SEC`
  "absolute", but it is multiplied like the others. `agent_server/state.py`
  says a finished trial with `HARBOR_DELETE_CONTAINERS=false` "leaves its
  containers behind". It runs `docker compose down`, which removes the containers
  and keeps volumes and images.
- `examples/experimental/openenv/openenv_agent_function.py:448-454` says a missing
  canonical verdict "drops the sample". Returning `None` does not drop anything:
  `openenv_generate.reward_func` defaults to 0.0, which is the false negative the
  comment warns against. Removing the default (`metadata.get("reward")`) would route
  it through the missing-reward filter and drop the group.
- `examples/experimental/harbor/harbor_agent_function.py:58-59` says configuration
  errors "raise instead ... a loud stop". `agentic_tool_call.generate` swallows
  every agent exception (`agentic_tool_call.py:98`); with no model calls recorded
  the sample is `ABORTED`, and a run whose every episode fails this way loops
  without finishing a step.
- `examples/swe-agent-harbor-docker/generate.py:11` and `run.py:199` use the
  deprecated `check_no_aborted` as the dynamic filter; that repeats the built-in
  aborted check and does no zero-std filtering.
- [Verifiers](/user-guide/verifiers) says Miles "does not currently preserve a
  trace's rollout-group boundary when it flattens multiple training samples".
  `_normalize_rewards_by_rollout` can: it groups by `group_index` and `rollout_id`
  (`train_data_conversion.py:294`). The Verifiers adapter is what still lacks it: it
  sets only `group_index` and raises on a trace with more than one branch
  (`examples/experimental/verifiers/verifiers_rollout.py:487-491`), so the
  statement holds for that adapter.
