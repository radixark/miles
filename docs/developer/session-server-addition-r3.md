---
title: Incremental R3 for the Session Server
description: Design for returning and assembling only additional routed-expert rows in in-place weight-update mode.
---

> Status: Implemented. This document records the design and the behavior contract of the landed implementation.

## Motivation and decision

TITO sessions send the full accumulated token sequence on every chat turn. When routing replay is enabled, SGLang currently returns routed-expert replay data (R3) for the full prompt and response again, so a long multi-turn session repeatedly base64-encodes, transfers, stores, and decodes the same prefix rows. The cumulative R3 payload grows approximately with the sum of all turn lengths instead of the final trajectory length.

The decision is to let the session server request only the additional R3 rows when weight updates use `in_place`. A session-server-internal boolean named `use_addition_r3` owns this behavior. It is derived automatically from `args.pause_generation_mode == "in_place"`; it is not a new CLI flag and cannot be configured independently.

The implementation is successful when the final training `Sample.rollout_routed_experts` is byte-for-byte identical to a full-R3 reference produced under the same `in_place` lifecycle while each turn transfers only the suffix that has not already been retained by the session. This equivalence comparison changes only whether SGLang slices the returned R3; it does not compare `in_place` with `retract`, whose weight-update lifecycle can legitimately produce different data. The existing full-R3 behavior remains unchanged in `retract` mode.

## Constraints and non-goals

- `--use-session-server` and `--pause-generation-mode=abort` are incompatible and must fail during argument validation before any session server or rollout engine starts.
- `retract` must preserve the current request, response, record, decode, and merge behavior: SGLang returns complete prompt-plus-response R3 because the KV cache is cleared and the prefix is prefilled again.
- Additional R3 is enabled only by `in_place`, where the active request and its KV-backed prefix remain in place across a weight update.
- `use_addition_r3` is an internal session-server parameter. It is not added to `argparse`, environment variables, HTTP APIs, or persisted user configuration.
- The mode mapping is static and explicit. Session code does not inspect updater implementations, cache metrics, retraction counters, or response metadata to rediscover the configured weight-update mode.
- Downstream sample serialization, training-data conversion, and routing replay keep their existing full-tensor contract: `rollout_routed_experts.shape[0] == len(tokens) - 1`.
- This proposal changes routed-expert replay only. It does not add an incremental protocol for `indexer_topk` or alter non-session `/generate` and chat-completion paths.
- The expected savings are in R3 collection output, base64/JSON serialization, HTTP transfer, session retention, decoding, and repeated full-prefix copying. This proposal does not claim to reduce the model's MoE routing computation.

## Current behavior

The current session path establishes the following facts:

- `SessionCore.chat_completions` in `miles/rollout/session/core.py` replaces messages with full TITO `input_ids` and sets `return_routed_experts=True` whenever routing replay is enabled.
- The unmodified upstream response is retained in `SessionRecord`, including its base64 `choice.meta_info.routed_experts` value.
- `get_routed_experts_from_response` in `miles/rollout/generate_utils/generate_endpoint_utils.py` decodes every response as a full `(len(tokens) - 1, num_layers, topk)` tensor.
- `compute_samples_from_openai_records` in `miles/rollout/session/samples/merge.py` builds one `Sample` per record, and generic `merge_samples` keeps the later sample's full R3 tensor.
- `examples/infra_features/random_async/random_async_rollout.py` already demonstrates the SGLang wire contract: a response with `routed_experts_start_len=s` contains `total_tokens - 1 - s` rows.

For turn `i`, let `N_i` be the number of prompt and completion tokens represented by the response. The current raw int32 payload contains `(N_i - 1) * num_layers * topk` values. Across many turns, the repeated-prefix cost is proportional to `sum(N_i - 1)`. Additional R3 makes the transferred row count proportional to the final trajectory length, except for intentional overlap after a rollback or rewritten suffix.

## Alternatives and choice

| Alternative | Result | Decision |
|---|---|---|
| Always return full R3 | Keeps the current implementation but retains the repeated-prefix overhead. | Rejected because it does not solve the motivating cost. |
| Add a user-facing `--use-addition-r3` flag | Allows combinations such as `retract` plus additional R3, which are incorrect, and makes users coordinate two descriptions of the same lifecycle. | Rejected. |
| Check `pause_generation_mode` throughout request and assembly code | Produces the right branch but couples R3 data handling directly to weight-update policy in multiple modules. | Rejected. |
| Derive one internal `use_addition_r3` parameter at session-server startup | Makes the legal mode mapping explicit once and lets all session R3 logic depend on one capability. | Chosen. |

## Design

### Configuration and ownership

`miles_validate_args` in `miles/utils/arguments.py` adds the configuration invariant:

```python
assert not (
    args.use_session_server and args.pause_generation_mode == "abort"
), "--use-session-server is incompatible with --pause-generation-mode=abort"
```

The session-server bootstrap derives the internal capability without adding an attribute intended for CLI configuration; `SessionServer.__init__` is the single derivation point, so the mapping cannot be configured independently of the mode:

```python
self.use_addition_r3 = getattr(args, "pause_generation_mode", None) == "in_place"
setup_session_routes(self.app, self, args, use_addition_r3=self.use_addition_r3)
```

`SessionServer` passes the boolean through `setup_session_routes` to both cores: the linear `SessionCore` and the tree-serving `SessionCoreV2`. The cores and the session-specific sample assembler read `use_addition_r3`; they do not read `pause_generation_mode` to choose R3 behavior.

| Session server | Weight-update mode | `use_addition_r3` | R3 behavior |
|---|---|---:|---|
| Disabled | Any mode | Not constructed | No session behavior change. |
| Enabled | `abort` | N/A | Startup assertion fails. |
| Enabled | `retract` | `False` | Omit `routed_experts_start_len` and retain full R3 behavior. |
| Enabled | `in_place` | `True` | Request and assemble additional R3. |

If routing replay itself is disabled, `use_addition_r3=True` is dormant: the request contains neither `return_routed_experts` nor `routed_experts_start_len`.

```mermaid
flowchart LR
    A["pause_generation_mode"] --> B{"Mode"}
    B -->|"abort + session server"| X["arguments.py hard assert"]
    B -->|"retract"| F["use_addition_r3 = False"]
    B -->|"in_place"| I["use_addition_r3 = True"]
    F --> R["Full R3 response"]
    I --> S["Request routed_experts_start_len"]
    S --> D["Store additional R3 patch"]
    D --> M["Materialize one full training tensor"]
```

### Request offset

The offset is computed in phase 1 of `SessionCore.chat_completions`, under the session lock, after `prepare_pretokenized` has applied any retry rollback and returned the new prompt token IDs. No independent monotonically increasing R3 counter is added to session state.

```text
previous_rows = max(0, len(checkpoint_token_ids) - 1)
stable_prefix_tokens = LCP(checkpoint_token_ids, prompt_token_ids)
routed_experts_start_len = min(previous_rows, stable_prefix_tokens)
```

The `LCP` term proves that every skipped row belongs to an unchanged causal token prefix. The `previous_rows` bound reflects that a checkpoint containing `N` tokens has only `N - 1` reusable R3 rows. A first turn or a rollback to the empty checkpoint therefore starts at `0`; a rollback to an earlier assistant checkpoint automatically moves the offset backward.

The v2 tree server computes the same offset against the positioned attach node's token snapshot (`active_token_ids()` after `position_for_request`): a request that branches off an ancestor re-requests the rows past that ancestor, exactly like a v1 rollback retry.

When `use_addition_r3` and routing replay are both enabled, the session server sends:

```json
{
  "return_routed_experts": true,
  "routed_experts_start_len": 123
}
```

The field is inserted before `proxy_body` is serialized. Because the full upstream request is later stored in `SessionRecord.request`, the exact offset used by a successful turn is persisted with its response without adding another record field. In `retract` mode the field is omitted, preserving the original wire request and SGLang's default full response.

### Response and patch contract

For a stored record, define:

```text
end = len(request.input_ids) + len(output_token_logprobs) - 1
start = request.routed_experts_start_len
delta_rows = end - start
```

The decoded additional tensor must have exactly `delta_rows * args.num_layers * topk` int32 values. To preserve the current decoder contract, `topk` is inferred from the first non-empty patch after divisibility checks rather than being newly required to equal `args.moe_router_topk`; every later non-empty patch must infer the same value because the patches share one final tensor. An empty patch contributes no new top-k evidence and reuses the value established by an earlier patch. If the trajectory has no non-empty R3 payload at all, the assembler preserves the existing empty-buffer behavior instead of introducing a new top-k source. The assembler rejects negative lengths, missing payloads, wrong byte counts, layer/top-k mismatches, and any patch sequence that leaves a gap.

The logical fold is:

```text
R_i = R_(i-1)[:start_i] + delta_i
```

This is replacement from `start_i`, not unconditional concatenation. It supports an offset moving backward after rollback or suffix rewriting: retained rows before `start_i` stay valid, overlapping old suffix rows are discarded, and the response supplies their replacement.

### Session-specific assembly

Additional R3 reconstruction belongs to `miles/rollout/session/samples/merge.py`, because only the session path persists ordered records and their offsets. The generic decoder and `merge_samples` continue to serve callers whose responses contain full tensors and must not gain delta semantics.

The additional path performs these operations:

1. Build the per-turn token, log-probability, loss-mask, status, replay-presence, and lifecycle data using the existing TITO alignment rules, without pretending that an R3 patch is a complete per-turn tensor.
2. Apply the existing turn-level trailing-token trim and `max_seq_len` selection.
3. Determine the terminal record with the same stop rules as `merge_samples`: a non-`COMPLETED` accumulated turn cannot be extended, and a later turn that introduces a required replay gap is not consumed. The selection loop must expose the terminal record index; it must not be inferred afterward from the original record count.
4. Decode and logically fold R3 patches only through that terminal record. Patch bookkeeping may retain slices/chunks, but it must not materialize a new accumulated full tensor for every turn.
5. Allocate one final int32 tensor with `len(merged.tokens) - 1` rows, copy the selected logical patches into it, verify complete coverage, and attach it to the merged `Sample`.
6. Pass the resulting full tensor through the existing safetensors codec, driver decode, training-data conversion, and routing-replay consumers unchanged.

The `retract` branch keeps the existing per-record full decoder and later-snapshot merge behavior. This explicit split makes “retract remains unchanged” testable rather than treating `start_len=0` as an incidental approximation of the old path.

The v2 tree server reuses the same assembler per leaf inside `build_leaf_material`: each root-to-leaf path is exactly the linear record chain its offsets were computed on, so every leaf folds its own patches into one full tensor.

### Session lifecycle and concurrency

- The offset is bound to the phase-1 checkpoint snapshot while the session lock is held.
- Only a phase-3 response whose `expected_num_assistant` still matches may update the TITO checkpoint and append its record. A stale concurrent response therefore cannot introduce an orphan R3 patch.
- Retry rollback truncates token checkpoints and records together. Recomputing the offset from the selected checkpoint keeps the R3 patch stream aligned without another rollback mechanism.
- Session deletion, client response stripping, and fake streaming remain unchanged. Agent-facing chat responses still do not expose replay payloads.

### Compatibility and failure behavior

- The deployment's SGLang revision must support `routed_experts_start_len` on `/v1/chat/completions`. There is no silent fallback from additional to full R3, because silently changing the payload contract can create an undetected gap or duplicate prefix.
- If SGLang rejects an additional-R3 request with a non-200 response, the session server retains the existing proxy behavior and does not record the turn.
- Malformed stored R3 is reported by the existing `/sessions/{id}/samples` assembly error boundary rather than changing the successful agent-facing chat response format.
- In-memory sessions require no migration. Records without `routed_experts_start_len` remain full-R3 records and continue down the existing branch.
- Configured `in_place` is the source of truth for this design. Whether every updater correctly preserves active KV under that mode is a backend correctness requirement, not a capability-discovery responsibility of the session server.

## Implementation and verification

The work can land in four reviewable cuts:

1. Add the `abort` startup assertion and wire the derived `use_addition_r3` boolean through `SessionServer`, `setup_session_routes`, and `SessionCore`.
2. Compute and persist `routed_experts_start_len` for in-place session requests while leaving retract requests byte-for-byte unchanged with respect to R3 fields.
3. Add session-specific patch decoding and one-time full-tensor materialization without changing the generic R3 decoder, `Sample` contract, codec, or trainer.
4. Add correctness tests and extend the existing session overhead benchmark to compare actual full and additional payloads.

The minimum CPU verification command is:

```bash
pytest -q tests/fast/utils/test_arguments.py tests/fast/router/test_sessions.py tests/fast/rollout/session/test_samples.py tests/fast/router/test_session_samples_op.py
```

Tests must cover:

- `session + abort` fails validation, `session + retract` derives `False`, and `session + in_place` derives `True`.
- Retract requests omit `routed_experts_start_len` and preserve the existing full-R3 expected response and final tensor.
- In-place first turn starts at `0`; later turns use the stable checkpoint prefix; retry rollback moves the offset backward.
- Two or more additional patches reconstruct exactly the same final int32 tensor as full-R3 responses for the same tokens and the same `in_place` weight lifecycle.
- TITO trailing-token rewrite, a terminal `max_seq_len` cut, an empty delta, and malformed/gapped patch data preserve or fail the stated contracts.
- An intermediate `finish_reason=length` followed by another stored record folds only through the intermediate terminal record; a later replay gap obeys the same boundary.
- Routing replay disabled remains unaffected even when `use_addition_r3=True`.

An SGLang integration check must additionally prove that, for each response, decoded rows equal `len(input_ids) + completion_tokens - 1 - routed_experts_start_len`. The performance comparison should report upstream R3 bytes and `/samples` assembly time for identical multi-turn `in_place` trajectories using full-R3 and addition-R3 response forms; correctness passes only if final tokens, loss mask, log probabilities, and full R3 are identical between those response forms.

## Risks and open questions

- Miles currently does not pin the rolling SGLang server source in all build paths. The integration test must run against the deployed revision so a field that is accepted but ignored cannot masquerade as successful additional R3.
- The additional protocol removes repeated payload work, but SGLang may still capture or stage full-prefix routing state internally. Any claim about GPU-side savings requires a separate SGLang measurement.
- The precise performance acceptance threshold is not yet fixed. Correctness and asymptotic payload reduction are blocking; a numerical latency target can be set after the existing manual benchmark measures representative turn lengths, layer counts, and top-k values.
