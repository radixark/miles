# Harbor agents on the multi-LoRA Tinker gateway

Agentic RL on the gateway from [`examples/multi_lora`](../README.md) without changing the Tinker wire format.
Training is the unmodified tinker-cookbook `rl/train.py` loop. Agents are Harbor harnesses (terminus-2 first)
running their tasks on the internal AgentENV sandbox; they send plain OpenAI chat messages to a **native tinker
session server** on the gateway, a small token-trajectory collector that renders with the base model's chat
template, samples through the gateway's own token path (adapter `M@V`), and records every turn's `input_ids`,
`output_ids` and `logprobs`. With `--tinker-tito-model` every turn's prompt inherits the previous turn's tokens
(TITO), so the whole trajectory is one token stream. The client turns the recorded turns into cookbook `Transition`s;
`trajectory_to_data` merges chained turns into one Datum and starts a new one where a chain breaks.

## Pieces

| Where | What |
|---|---|
| gateway `miles/tinker/core/tinker_session_server.py` | `TrajectoryCollector`: recorded sessions, one per trajectory; `tito_render_prompt` / `on_turn_committed` inherit tokens turn to turn through an injected miles `TITOTokenizer` (full re-render when off or when a chain breaks) |
| gateway `miles/tinker/server/oai_routes.py` | `POST /oai/sessions/{sid}` bind (`sampling_session_id`, optional `max_datum_tokens`), `POST /oai/sessions/{sid}/v1/chat/completions`, `GET /oai/sessions/{sid}` turns, `DELETE` |
| gateway `serve_tinker.py`, `miles/tinker/arguments.py` | mounts the routes on the served app; `--tinker-session-ttl-s` (default 3600), `--tinker-tito-model` (a `TITOTokenizerType`, its fixed template replaces `--chat-template-path`), renders with `--apply-chat-template-kwargs` |
| client `harbor_env.py` | cookbook plug-ins: `HarborDatasetBuilder`, `HarborGroup` (rewards from Harbor verdicts), `SessionRolloutStrategy` (bind → Harbor trial → export → delete → `Trajectory`) |
| client `run_harbor_tinker.py` | `HarborTinkerConfig` → cookbook `train.Config` → `train.main`, with a sandbox preflight |

Status codes on the session routes: 400 bad input or missing key, 403 another tenant's session, 404 unknown session,
429 per-tenant session cap or per-session turn cap, 502 engine failure (nothing recorded).

## Run

1. **Gateway** (serving node): start it as in [`examples/multi_lora`](../README.md). The cookbook creates its model with
   the SDK default `train_unembed=True`, so keep `--tinker-train-unembed` on (the launcher's default). The gateway refuses a
   Datum longer than `min(model max_position_embeddings, --max-tokens-per-gpu)` and closes the model, and agent trajectories
   run to 10–30k tokens, so pass e.g. `--extra-args "--max-tokens-per-gpu 32768 --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --tinker-session-ttl-s 7200"`
   and set the client's `max_datum_tokens` to the same cap.
2. **Client host** (on the tailnet that reaches the gateway and AgentENV):

   ```bash
   pip install "tinker==0.26.2" tinker-cookbook "harbor[e2b] @ git+https://github.com/harbor-framework/harbor@harbor-miles-v0.20.0"
   mkdir -p ~/.config/e2b && echo <key> > ~/.config/e2b/api_key && chmod 600 ~/.config/e2b/api_key
   git clone https://github.com/laude-institute/terminal-bench-2 ~/.cache/terminal-bench-2   # one task dir per task.toml
   ```

3. **Train**:

   ```bash
   HARBOR_ENV_TYPE=e2b E2B_API_URL=https://<agentenv> TINKER_API_KEY=tml-<key> \
   python examples/multi_lora/harbor_tinker/run_harbor_tinker.py \
       gateway=http://<gateway>:10613 model_name=Qwen/Qwen3-30B-A3B tasks_dir=~/.cache/terminal-bench-2 \
       groups_per_batch=4 group_size=4 max_tokens=8192 lora_rank=16
   ```

   Every step runs `groups_per_batch × group_size` trials, each in its own sandbox. The preflight fails fast on a missing
   `HARBOR_ENV_TYPE` / `HARBOR_TASKS_DIR`, a missing provider key, an old e2b SDK, or an unreachable `E2B_API_URL`.

## What a step looks like

```
cookbook train.main ── Tinker SDK ──▶ gateway: create_model / forward_backward / optim_step / save_weights_for_sampler
 │ per trial: POST /oai/sessions/{sid} {sampling_session_id}   (the policy's Tinker sampler → tinker://M/sampler_weights/V)
 │           harbor_agent_function.run(base_url=…/oai/sessions/{sid}) ── e2b ──▶ AgentENV task container, verifier → reward
 │              terminus-2 ── OpenAI messages ──▶ collector: apply_chat_template → submit_sample(M@V) → Turn recorded
 │           GET /oai/sessions/{sid} → turns → Transition per turn → Trajectory; DELETE
 └ trajectory_to_data → Datums → forward_backward
```

## TITO (token inheritance across turns)

Without `--tinker-tito-model` the collector renders the full history every turn. On Qwen3 the native template strips
earlier thinking and the no-think generation prompt carries an empty `<think>` block the history omits, so a re-render
never equals the previous turn's tokens: consecutive turns do not chain and each turn is its own Datum, exact
on-policy but O(T²) tokens per trajectory.

With `--tinker-tito-model qwen3` (any `TITOTokenizerType`) the gateway reuses miles' full-model TITO machinery:

- `resolve_fixed_chat_template` installs the family's fixed template (`qwen3_fixed.jinja`, `clear_thinking=false`) as
  the chat template and merges its kwargs into `--apply-chat-template-kwargs`; the first turn and every fallback still
  go through the same HF `apply_chat_template`.
- `tito_render_prompt` (mirrors `LinearTrajectory.prepare_pretokenized`) asks `TITOTokenizer.merge_tokens` for the
  previous turn's `input_ids + output_ids` plus only the tokens of the appended messages (append-only history check,
  allowed roles, the missing `\n` after `<|im_end|>`); the sandbox's new text is tokenized once and never again.
- `on_turn_committed` (mirrors `update_pretokenized_state`) keeps the history plus reply and the token checkpoint on
  the session; the export marks each turn `inherits`.
- A chain breaks (full re-render, new Datum) when the harness edits, reorders or summarizes history, appends a role
  the family does not allow, or the prompt plus `max_tokens` would exceed the datum budget: the gateway cap lowered to
  the client's `max_datum_tokens` sent at bind.

Sessions live per trajectory under their tenant: freed by the client's `DELETE`, by the idle TTL, or as soon as the
tenant's Tinker lease is gone (`sweep` checks `service.sessions`), so token state never outlives its LoRA.

## Gates

- Local loop (no GPU): serve the gateway app with the fast suite's `FakeBackend` under uvicorn, point the real cookbook
  `train.main` at it with `SessionRolloutStrategy(run_trial=<fake trial that chats through the session route>)`; one step
  records 8 turns → 8 Datums → `forward_backward` → checkpoints.
- GPU: start the gateway, `curl` one session for three turns and read `GET /oai/sessions/{sid}`; run Harbor's `oracle`
  agent through AgentENV; one terminus-2 `fix-git` trial; one training step with a moving loss.
