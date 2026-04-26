# TITO Session Server Design

This document describes the multi-turn TITO design in branch `fix/tito-non-tool-append` at commit `f66555b2`.

## Scope

The design covered here is the code path formed by:

- `miles/rollout/generate_hub/agentic_tool_call.py`
- `miles/rollout/generate_utils/openai_endpoint_utils.py`
- `miles/rollout/session/session_server.py`
- `miles/rollout/session/sessions.py`
- `miles/rollout/session/linear_trajectory.py`
- `miles/utils/chat_template_utils/tito_tokenizer.py`
- `miles/utils/chat_template_utils/token_seq_comparator.py`

It also references the tests that currently define the contract:

- `tests/fast/router/test_linear_trajectory.py`
- `tests/fast/router/test_session_pretokenized_e2e.py`
- `tests/fast/utils/chat_template_utils/test_tito_tokenizer.py`
- `tests/fast/utils/chat_template_utils/test_tito_tokenizer_model_matrix.py`
- `tests/fast/utils/chat_template_utils/test_token_seq_comparator.py`
- `tests/e2e/sglang/test_session_server_tool_call.py`

## Problem Statement

In multi-turn tool-calling, the agent repeatedly sends:

1. the full message history so far
2. a new batch of non-assistant messages after the previous assistant turn
3. a request for the next assistant continuation

Naively, every turn re-renders and re-tokenizes the entire conversation. That is expensive, and it also throws away the exact token sequence the model actually decoded in previous turns.

TITO ("token-in token-out") solves this by:

- storing the exact token IDs returned by SGLang for each assistant checkpoint
- reusing those token IDs as the next turn's prompt prefix
- incrementally tokenizing only the newly appended non-assistant suffix
- validating that the merged prompt still agrees with the canonical chat-template rendering where it matters

The `fix/tito-non-tool-append` branch extends the tokenizer side from "tool-only append" toward "non-assistant append", meaning it can incrementally tokenize appended `tool`, `system`, and opt-in `user` segments instead of assuming that every appended turn is a tool response.

## Design Goals

- Reuse exact model-produced token prefixes across turns.
- Keep the session/TITO logic outside the router so it can proxy any compatible backend.
- Make correctness failures explicit at token level instead of silently falling back.
- Support append-only non-assistant growth after an assistant checkpoint.
- Isolate model-specific boundary quirks in one place.
- Preserve per-turn decode logprobs for training sample construction.

## Non-Goals

- Arbitrary message-history editing. The supported mutation is append-only growth plus at most one assistant-step rollback.
- Proving that assistant text re-tokenizes identically to the canonical template. That is intentionally not required.
- Universal support for every chat template. TITO still depends on append-only template behavior and on synthetic segment renders matching full renders.

## High-Level Architecture

The production path is:

1. `agentic_tool_call.generate` creates an `OpenAIEndpointTracer`.
2. The tracer creates a session on the standalone session server.
3. The custom agent sends requests to `/sessions/{session_id}/v1/chat/completions`.
4. The session server:
   - validates the message history against the stored session state
   - optionally builds `input_ids` by merging the stored prefix with incremental suffix tokens
   - proxies the request to the backend
   - updates the session checkpoint from the backend's returned prompt/output token IDs
5. After the agent finishes, `OpenAIEndpointTracer.collect_records` fetches all per-turn records plus session metadata.
6. `compute_samples_from_openai_records` reconstructs training samples from those records, trimming model-specific trailing stop/boundary tokens when needed.

The session server is therefore both:

- a stateful proxy for inference requests
- the authoritative owner of per-session token checkpoints

## Real Request Path

### 1. Generate Function Entry

`miles/rollout/generate_hub/agentic_tool_call.generate` is the outer entry point for agentic rollout.

Its job is intentionally narrow:

- assert that `session_server_ip/session_server_port` exist
- create a session-scoped tracer
- run the user agent against the session-scoped base URL
- fetch session records at the end
- convert records into `Sample` objects
- optionally merge multi-turn samples into one sample

The session/TITO logic itself is not implemented here. It lives behind the HTTP session API.

### 2. Session Creation and Teardown

`OpenAIEndpointTracer.create`:

- probes `/health` to learn `session_server_instance_id`
- creates a session via `POST /sessions`
- returns a tracer bound to `/sessions/{session_id}`

`collect_records`:

- fetches `GET /sessions/{session_id}`
- parses records and metadata
- deletes the session with `DELETE /sessions/{session_id}`

This keeps the trainer side stateless while the session server owns the mutable trajectory.

### 3. Chat Completion Request

The hot path is `POST /sessions/{session_id}/v1/chat/completions` in `miles/rollout/session/sessions.py`.

It has three phases:

1. `prepare request` under `session.lock`
2. `proxy to backend` without holding the lock
3. `update state` under `session.lock`

This lock split is intentional. The server serializes state mutation, but it does not block deletion or unrelated waiting on a long backend inference.

Before proxying, the route hardcodes the SGLang flags required by TITO:

- `logprobs=True`
- `return_prompt_token_ids=True`
- `return_meta_info=True`
- `no_stop_trim=False`

These are not left to agent-side defaults because TITO state reconstruction depends on them.

### 4. Response Finalization

After the proxy returns, the route requires:

- `choice.meta_info.output_token_logprobs`
- `choice.prompt_token_ids`
- non-`None` assistant content
- `len(output_token_logprobs) == completion_tokens`

If any of these are wrong, the route raises `UpstreamResponseError` instead of guessing.

If the backend returns a non-200 response at all, the session route simply proxies that error back to the caller and does not append a session record or checkpoint. Failed turns therefore do not mutate session state.

## Session State Model

Per session, `LinearTrajectory` stores:

- `messages`: full accepted message history up to the latest assistant checkpoint
- `records`: one `SessionRecord` per successful assistant turn
- `trajectory_token_ids`: one accumulated token sequence per assistant checkpoint
- `num_assistant`: number of accepted assistant checkpoints
- `lock`: per-session mutation lock
- `closing`: tombstone flag used by delete/race handling

The important detail is that `trajectory_token_ids` stores full accumulated token sequences, not just deltas. `token_ids` always means "latest assistant checkpoint".

When a turn succeeds, `update_pretokenized_state` appends:

`prompt_token_ids + completion_token_ids`

as the next checkpoint. That means the stored prefix is exactly what the backend used and produced, not a later re-rendered approximation.

## State Machine and Invariants

### Invariant 1: First Turn Has No Pretokenized Prefix

If `session.token_ids` is empty, `prepare_pretokenized` returns `None`. The first request is forwarded without `input_ids`.

### Invariant 2: New Requests Must Be Append-Only After Validation/Rollback

Before token merging, the route calls:

- `_try_detect_and_rollback_to_assistant_checkpoint`
- `assert_messages_append_only_with_allowed_role`

This means the accepted request must be one of:

- a strict append-only extension of stored history
- a retry that diverges after some earlier assistant checkpoint and can be rolled back to that checkpoint

### Invariant 3: Rollback Is Limited to One Assistant Step

Rollback is not arbitrary history rewriting.

The current implementation:

- finds the longest prefix of stored/request messages that still matches by template-relevant keys
- finds the last assistant inside that matched prefix
- rolls back to the checkpoint ending at that assistant
- rejects the request if this would discard more than `MAX_ASSISTANT_ROLLBACK_STEPS` assistant turns

Today `MAX_ASSISTANT_ROLLBACK_STEPS = 1`, so only one assistant-step rollback is legal.

### Invariant 4: The Stored Prefix Must Still Match the New Backend Result

After the backend returns, `update_pretokenized_state` checks that the previously stored checkpoint is a prefix of the newly returned:

`prompt_token_ids + completion_token_ids`

except for up to `max_trim_tokens` trailing tokens.

This is the online correctness gate for TITO. If the merged prompt was wrong, the new backend result will usually fail this prefix check and the request becomes a `TokenizationError`.

### Invariant 5: Only Configured Roles May Be Appended

`assert_messages_append_only_with_allowed_role` enforces the append-role contract.

Runtime defaults:

- `--tito-allowed-append-roles` defaults to `tool`
- legal choices are `tool`, `user`, `system`

The branch expands tokenizer support to non-tool appends, but user append is still opt-in and warned about because some templates retroactively change earlier rendering when a new user message appears.

## Why We Need a Dedicated `TITOTokenizer`

The tokenizer in `miles/utils/chat_template_utils/tito_tokenizer.py` is not redundant with the HF tokenizer.

It exists because the session server needs a different contract from ordinary full-chat tokenization:

1. It must work incrementally.
   The backend already produced an exact assistant prefix. We only want tokens for the newly appended non-assistant suffix plus the next assistant opener.

2. It must preserve model-produced assistant tokens.
   We are not allowed to re-tokenize the old assistant text and replace the stored prefix, because that would destroy decode-time token identity and break later logprob/sample reconstruction.

3. It must own model-specific boundary repair.
   The hard part is not just "tokenize suffix". It is joining:

   - old exact backend tokens
   - newly rendered suffix tokens

   at a boundary where some models emit or omit structural tokens differently.

4. It must expose the validation policy used by the comparator.
   `create_comparator()` is part of the same abstraction because the same model-specific boundary rules must also define how correctness is judged.

Without a dedicated `TITOTokenizer`, the logic for:

- suffix segmentation
- dummy synthetic contexts
- model-specific prefix surgery
- trailing token trimming
- comparator configuration

would be spread across the session server and tests instead of living in one tokenizer-specific layer.

## `TITOTokenizer` Algorithm

### Base Contract

`tokenize_additional_non_assistant(old_messages, new_messages, tools)` assumes:

- `new_messages` extends `old_messages`
- appended messages are all non-assistant roles allowed by configuration

It then tokenizes only the appended suffix and returns the incremental token IDs that should follow the stored prefix.

### Why the Branch Switched to Role-Segmented Synthetic Prefixes

Before this branch, the tokenizer used one dummy conversation shape for all appended messages.

That breaks down once the appended suffix can contain more than just tool responses, because:

- tool-response rendering depends on a preceding assistant tool-call shape
- user-message rendering depends on a preceding user/assistant boundary
- system-message rendering can have its own wrapper rules
- some templates treat contiguous tool responses as one logical block

The branch therefore splits the appended suffix into segments by role:

- contiguous `tool` runs are grouped together
- `user` messages are single-message segments
- `system` messages are single-message segments

Grouping contiguous tools is deliberate: many templates wrap a tool-response run as one block, so tokenizing each tool independently would invent boundaries that do not exist in the full render.

### Segment Tokenization Strategy

For each segment, TITO renders a minimal synthetic context and takes a suffix diff at the text level:

- tool segment: `[_DUMMY_SYSTEM, _build_dummy_assistant(tool_responses)]`
- user segment: `[_DUMMY_SYSTEM, _DUMMY_USER]`
- system segment: `[_DUMMY_SYSTEM]`

Then it computes:

1. `text_without = render(base_messages, add_generation_prompt=False)`
2. `text_with = render(base_messages + appended_segment, ...)`
3. assert `text_with.startswith(text_without)`
4. encode only the suffix `text_with[len(text_without):]`

This text-diff approach is important because it lets the tokenizer preserve the exact special-token/string structure emitted by the template without re-tokenizing the full conversation every turn.

### Why Tool Segments Need a Dummy Assistant

A tool response is not standalone in OpenAI-format chat templates. It is normally rendered relative to the assistant tool call that caused it.

`_build_dummy_assistant` therefore synthesizes:

- `tool_calls` with matching ids/names
- empty assistant content
- `reasoning_content = " "`

The `reasoning_content` placeholder is not cosmetic. The Qwen3 thinking-template test verifies that this preserves the expected tool-call rendering shape for reasoning-capable templates.

### Why the Generation Prompt Is Appended Only Once

After all non-assistant segments are tokenized left-to-right, the tokenizer appends the next assistant opener exactly once by rendering:

`_tokenize_rendered_suffix(new_messages, [], add_generation_prompt=True)`

This avoids accidentally inserting multiple assistant openers between appended segments. The branch adds an explicit test for this.

## `merge_tokens` Design

`merge_tokens` is where the stored backend prefix and the incremental suffix are joined.

### Base Behavior

The default implementation is simple concatenation:

`pretokenized_token_ids + incremental`

This is correct when the model's generated boundary tokens already line up with the template-rendered next-turn boundary.

### Qwen3 Behavior

`Qwen3TITOTokenizer.merge_tokens` inserts a newline when the stored prefix ends with `<|im_end|>`.

Reason:

- Qwen3 templates render `<|im_end|>\n` after a message
- the model commonly stops at `<|im_end|>` without generating the trailing newline
- the next template-rendered suffix assumes that newline exists

So the merged prefix becomes:

- unchanged if the last stored token is not `<|im_end|>`
- `prefix + ["\n"]` if the last stored token is `<|im_end|>`

The same newline is also registered as a trailing token for comparison trimming.

### GLM 4.7 Behavior

`GLM47TITOTokenizer.merge_tokens` strips the final stored token when it is one of:

- `<|observation|>`
- `<|user|>`

Reason:

- these tokens are both valid assistant stop tokens and valid next-turn openers
- the model may emit one of them at the end of a decode
- the template may emit the same or the other one again as the next-turn boundary

So the safe rule is: if the stored prefix ends with an ambiguous GLM boundary token, drop it before concatenating the incremental suffix.

This tokenizer also sets `max_trim_tokens = 1`, because downstream sample reconstruction is allowed to trim exactly one such trailing token from non-final turns.

### Why Boundary Repair Lives Here

`merge_tokens` is the right layer for Qwen/GLM quirks because the session server itself should not know model-specific token ids. It only asks the selected tokenizer to:

- compute incremental suffix tokens
- define how prefix/suffix should be joined
- define how much trailing mismatch is tolerable later

## Correctness Validation Strategy

The system does not rely on one check. It uses three layers.

### Layer 1: Online Prefix Check After Real Inference

`LinearTrajectory.update_pretokenized_state` is the first hard gate.

After a real backend response, it asserts that the previously stored checkpoint is still a prefix of the newly returned total token sequence, allowing at most `max_trim_tokens` trailing-token differences.

This catches incorrect TITO merges immediately during the live request path.

### Layer 2: Canonical Sequence Comparison via `TokenSeqComparator`

`SessionRegistry.compute_session_mismatch` renders the canonical full token sequence with:

`apply_chat_template(session.messages, add_generation_prompt=False, tokenize=True)`

and compares it against `session.token_ids`.

This comparison is diagnostic metadata. It is exposed as:

- `tito_session_mismatch`
- `accumulated_token_ids`
- `max_trim_tokens`

and later propagated into rollout samples.

### Layer 3: Live Session E2E

The active end-to-end contract is the role-aware session-server test:

- it runs the real Miles rollout path against SGLang
- it validates the session metadata exposed by the server
- it fails if the configured append roles are not actually exercised

Numerical re-prefill logprob comparison is no longer treated as the TITO correctness oracle. TITO correctness is defined by token-prefix integrity, mismatch taxonomy, role coverage, and the fast rollback regressions.

## Why `TokenSeqComparator` Matches Our Intent

The comparator is deliberately not a raw "token-by-token equality" check.

Our intent is:

- structural markers must be correct
- non-assistant content must be exact
- assistant content may differ because we intentionally preserve model-produced assistant tokens instead of canonical re-tokenized assistant text

The comparator encodes exactly that intent.

### Step 1: Collect Special-Token Boundaries

It builds the special-token set from:

- `tokenizer.all_special_ids`
- tokenizer added tokens marked `special=True`

This lets it split a flat token sequence into alternating:

- special-token segments
- content segments

### Step 2: Trim Known Trailing Boundary Tokens

Before comparing, it can strip model-specific trailing ids from both sequence tails.

Examples:

- Qwen3 newline after `<|im_end|>`
- GLM `<|user|>` / `<|observation|>`

This avoids false structural mismatches caused by end-of-turn stop tokens that get consumed by the next turn's template rendering.

### Step 3: Reject Structural Drift First

If the two segmented sequences differ in:

- segment count
- special/content pattern

the comparator returns a single `SPECIAL_TOKEN_COUNT` mismatch immediately.

This is the strongest failure mode: the turn-boundary structure is wrong, so per-segment alignment is no longer meaningful.

### Step 4: Compare Aligned Segments

For aligned segments:

- special-token mismatch -> `SPECIAL_TOKEN_TYPE`
- content mismatch in assistant region -> `ASSISTANT_TEXT`
- content mismatch elsewhere -> `NON_ASSISTANT_TEXT`

Assistant-region detection is model-specific and uses `assistant_start_str`, for example:

- Qwen3: `<|im_start|>assistant`
- GLM47: `<|assistant|>`

### Why This Is the Right Severity Split

This severity split matches the actual TITO contract:

- `SPECIAL_TOKEN_COUNT`, `SPECIAL_TOKEN_TYPE`, and `NON_ASSISTANT_TEXT` indicate real bugs in TITO or the template
- `ASSISTANT_TEXT` is expected because assistant tokens come from previous decode output, not from canonical full-history re-tokenization

That same policy is used in rollout metrics:

- CI requires zero rate for structural and non-assistant mismatches
- assistant-text mismatch is treated as non-critical

## Comparator Limits

The comparator is intentionally scoped. It does not prove everything.

It does prove:

- special-token structure still lines up
- non-assistant text still lines up
- model-specific trailing boundaries are being normalized consistently

It does not prove:

- that assistant text would re-tokenize identically under a full render
- that a template family is globally append-only for every possible conversation
- that live backend logprobs are equivalent by itself

Those are covered, respectively, by:

- allowing `ASSISTANT_TEXT`
- chat-template verification/autofix tooling
- the re-prefill logprob-equivalence e2e test

## Boundary Cases and How They Are Handled

### First Turn

No stored checkpoint exists, so there is no `input_ids` injection.

### Tool-Only Append

This is the default and most stable path. It is enabled by default and is the historical TITO path.

### System Append

Supported by the new segmented tokenizer when enabled through `--tito-allowed-append-roles`.

This is useful for retry injections like:

- "please retry with a tool"
- intermediate system guidance between tool turns

### User Append

Supported by the tokenizer, but still opt-in and warned about in argument validation.

Why warned:

- some templates use context-sensitive logic for the "latest user query"
- adding a new user message can retroactively change how earlier turns render
- incremental tokenization assumes earlier rendered text does not change

So the branch supports user append as a capability, but it does not claim it is universally safe across all template families.

### Contiguous Multi-Tool Responses

Grouped as one segment, because some templates wrap an entire tool-response run rather than each tool independently.

### Retry After Failed Tool Parsing

Supported in two ways by the e2e agent:

- rollback to the previous assistant checkpoint
- keep the assistant message and append a retry tool message

The session server handles the former via rollback detection and the latter via normal append validation.

### Divergence Before Any Assistant Checkpoint

Rejected. If rollback cannot find an assistant inside the matched prefix, `MessageValidationError` is raised.

### More Than One Assistant-Step Rollback

Rejected. The code explicitly enforces `MAX_ASSISTANT_ROLLBACK_STEPS = 1`.

### Backend Returns Invalid Metadata

Rejected as `UpstreamResponseError`.

### Prefix Mismatch After Real Decode

Rejected as `TokenizationError`.

### Session Closed During Inference

The route detects `session.closing` and skips state update instead of mutating a deleted session.

### State Changed While Proxy Was In Flight

The route compares `expected_num_assistant` against current `session.num_assistant`. If they differ, it skips state update, preventing stale writes.

## Model Coverage

There are two different coverage notions.

### Runtime Tokenizer Types

The runtime session server exposes these tokenizer types via `--tito-model`:

- `default`
- `qwen3`
- `qwen35`
- `qwennext`
- `glm47`

This is the actual production selection surface today.

### Default Fast Test Matrix

The default fast model-matrix test is a repo-maintained tokenizer regression. It keeps one representative per maintained tokenizer family and only runs the default `tool` append surface:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen3-0.6B`
- `Qwen/Qwen3.5-0.8B`
- `Qwen/Qwen3-Next-80B-A3B-Thinking`
- `zai-org/GLM-4.7-Flash`

Important details:

- Qwen3/Qwen3.5/QwenNext representatives use their declared tokenizer types.
- GLM-4.7 resolves to the `glm47` tokenizer type.
- Qwen2.5 uses the `default` tokenizer path in tests.
- For `default`, the tests pass a model-specific `assistant_start_str` so the comparator can still distinguish assistant vs non-assistant mismatches.
- Broad exploratory models and non-default role sets are not part of fast CI.

### Known Exclusions

`deepseek-ai/DeepSeek-V3` is explicitly excluded from the branch's TITO matrix because segment-wise tool tokenization emits extra tool-output wrappers compared with full-conversation rendering.

This is a real design boundary, not a missing assertion.

### Partial Coverage

System/user append cases and broad exploratory model families are not part of the default fast model matrix. The fast contract only claims the repo-maintained `tool` append surface.

## Dependency on Append-Only Chat Templates

TITO only works when the chat template is append-only after the assistant checkpoint:

- rendering a longer history must preserve the earlier rendered prefix
- synthetic segment renders must preserve the intended local boundary shape

This is why the e2e session tests use:

- `--chat-template-path autofix`

and why the repository ships `docs/en/agentic/chat_template_verification.md`.

If the template itself is not append-only, no tokenizer-side merge algorithm can make the session server fully correct.

## Sample Reconstruction and Trailing-Token Trimming

`compute_samples_from_openai_records` reconstructs training samples from per-turn session records using the final `accumulated_token_ids`.

Its cursor-based algorithm:

1. positions the cursor at the end of each turn's prompt
2. greedily matches that turn's output ids against the accumulated sequence
3. trims unmatched trailing output tokens on non-final turns, up to `max_trim_tokens`
4. requires the full accumulated sequence to be consumed by the end

This is downstream of `merge_tokens`, but it is part of the same design. The model-specific tokenizer defines `max_trim_tokens`, and sample reconstruction consumes that contract.

For GLM47, this is essential because a trailing boundary token may belong logically to the next turn's template render instead of the previous turn's sampled output.

## Error Surface

The session server uses typed errors:

- `SessionNotFoundError` -> `404`
- `MessageValidationError` -> `400`
- `TokenizationError` -> `500`
- `UpstreamResponseError` -> `502`

This mapping matters operationally:

- `400` means the agent/request violated the session contract
- `500` means TITO invariants failed inside Miles
- `502` means the upstream backend returned an incompatible response

## Observability

The main TITO observability signals are:

- `tito_session_mismatch`
- `accumulated_token_ids`
- `max_trim_tokens`
- rollout metrics:
  - `tito_session_mismatch_rate`
  - `tito_session_mismatch_rate/special_token_count`
  - `tito_session_mismatch_rate/special_token_type`
  - `tito_session_mismatch_rate/non_assistant_text`
  - `tito_session_mismatch_rate/assistant_text`

In CI, the first three mismatch types must be exactly zero.

## What This Branch Changes

Relative to the previous tool-only tokenizer design, `fix/tito-non-tool-append` changes the tokenizer contract in three important ways:

1. appended non-assistant content is segmented by role instead of handled by one dummy prefix
2. `user` append is now tokenizable in the tokenizer layer, although still opt-in and warned about
3. a new model-matrix test suite checks that the merged result preserves non-assistant structure/content across multiple model families

The session server state machine, error mapping, and online prefix check are not fundamentally changed by this branch. The major change is the tokenizer's ability to compute correct incremental suffixes for more non-assistant append patterns.

## Known Limits and Open Follow-Ups

- `default` runtime tokenizer does not auto-infer `assistant_start_str`; only the `qwen3` and `glm47` specializations set it automatically. In tests, broader families pass it manually to improve mismatch classification.
- `_assert_no_user_after_assistant` exists in `linear_trajectory.py` but is not on the active request path today. The live validation path is append-only validation plus rollback.
- Multi-step rollback is intentionally unsupported.
- Some template families still require exclusion or partial coverage because segment-local rendering does not match full-conversation rendering.

## Summary

The session server owns the mutable session checkpoint.

`TITOTokenizer` owns incremental suffix tokenization plus model-specific merge behavior.

`TokenSeqComparator` owns the "what counts as correct" judgment:

- structure must match
- non-assistant content must match
- assistant content may differ

That split is what lets Miles reuse exact decoded prefixes across turns without pretending that full re-tokenization is the ground truth for assistant tokens, while still keeping hard failure modes for the parts of the sequence that actually affect next-turn semantics.
