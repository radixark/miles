---
title: "Harbor agents on the multi-LoRA Tinker gateway"
description: "Harbor agents (terminus-2 on the AgentENV sandbox) trained through the multi-LoRA Tinker gateway with the unmodified tinker-cookbook RL loop."
# Generated from examples/multi_lora/harbor_tinker/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
Agentic RL on the gateway from [`examples/multi_lora`](/examples/multi-lora) without changing the Tinker wire format.
Training is the unmodified tinker-cookbook `rl/train.py` loop. Agents are Harbor harnesses (terminus-2 first)
running their tasks on the internal AgentENV sandbox; they send plain OpenAI chat messages to a **native tinker
session server** on the gateway, a small token-trajectory collector that renders with the base model's chat
template, samples through the gateway's own token path (adapter `M@V`), and records every turn's `input_ids`,
`output_ids` and `logprobs`. The client turns those turns into cookbook `Transition`s; `trajectory_to_data`
decides whether a trajectory becomes one Datum (turns chain) or one Datum per turn (they do not).

## Pieces

| Where | What |
|---|---|
| gateway `miles/tinker/core/tinker_session_server.py` | `TrajectoryCollector`: recorded sessions, one per trajectory; TITO hooks empty (full re-render every turn) |
| gateway `miles/tinker/server/oai_routes.py` | `POST /oai/sessions/{sid}` bind, `POST /oai/sessions/{sid}/v1/chat/completions`, `GET /oai/sessions/{sid}` turns, `DELETE` |
| gateway `serve_tinker.py`, `miles/tinker/arguments.py` | mounts the routes on the served app; `--tinker-session-ttl-s` (default 3600), renders with `--apply-chat-template-kwargs` / `--chat-template-path` |
| client `harbor_env.py` | cookbook plug-ins: `HarborDatasetBuilder`, `HarborGroup` (rewards from Harbor verdicts), `SessionRolloutStrategy` (bind → Harbor trial → export → delete → `Trajectory`) |
| client `run_harbor_tinker.py` | `HarborTinkerConfig` → cookbook `train.Config` → `train.main`, with a sandbox preflight |

Status codes on the session routes: 400 bad input or missing key, 403 another tenant's session, 404 unknown session,
429 per-tenant session cap or per-session turn cap, 502 engine failure (nothing recorded).

## Run

1. **Gateway** (serving node): start it as in [`examples/multi_lora`](/examples/multi-lora). The cookbook creates its model with
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

Stage 1 renders the full history every turn. On Qwen3 with thinking on and terminal output fed back as user
messages (what terminus-2 does), the template strips earlier thinking, so consecutive turns do not chain and each
turn is its own Datum: exact on-policy, but O(T²) tokens per trajectory. `--apply-chat-template-kwargs '{"enable_thinking": false}'`
keeps turns chaining; the `tito_render_prompt` hook is where token inheritance goes later.

## Gates

- Local loop (no GPU): serve the gateway app with the fast suite's `FakeBackend` under uvicorn, point the real cookbook
  `train.main` at it with `SessionRolloutStrategy(run_trial=<fake trial that chats through the session route>)`; one step
  records 8 turns → 8 Datums → `forward_backward` → checkpoints.
- GPU: start the gateway, `curl` one session for three turns and read `GET /oai/sessions/{sid}`; run Harbor's `oracle`
  agent through AgentENV; one terminus-2 `fix-git` trial; one training step with a moving loss.
