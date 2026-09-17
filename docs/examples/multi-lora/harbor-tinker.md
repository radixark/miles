---
title: "Harbor agents on the multi-LoRA Tinker gateway"
description: "Harbor agents (terminus-2 on the AgentENV sandbox) trained through the multi-LoRA Tinker gateway with the unmodified tinker-cookbook RL loop."
# Generated from examples/multi_lora/harbor_tinker/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
Agentic RL on the gateway from [`examples/multi_lora`](/examples/multi-lora) without changing the Tinker wire format. The
unmodified tinker-cookbook `rl/train.py` loop trains; Harbor harnesses (terminus-2) run their tasks on the AgentENV
sandbox and chat with a recorded session on the gateway, which samples through the gateway's own token path
(adapter `M@V`) and records every turn's `input_ids`, `output_ids` and `logprobs`. With `--tinker-tito-model` each
turn's prompt inherits the previous turn's tokens (TITO), so a trajectory trains as one Datum.

## Pieces

| Where | What |
|---|---|
| gateway `miles/tinker/core/tinker_session_server.py` | `TrajectoryCollector`: recorded sessions, one per trajectory; `tito_render_prompt` / `on_turn_committed` inherit tokens turn to turn through an injected miles `TITOTokenizer` (full re-render when off or when a chain breaks) |
| gateway `miles/tinker/server/oai_routes.py`, `oai_shapes.py` | `POST /oai/sessions/{sid}` bind (`sampling_session_id`, optional `max_datum_tokens`), `POST /oai/sessions/{sid}/v1/chat/completions`, `GET /oai/sessions/{sid}` turns, `DELETE`; `oai_shapes.py` maps the OpenAI body to a `TurnRequest` and the recorded `TurnResult` back to ChatCompletion JSON |
| gateway `serve_tinker.py`, `miles/tinker/arguments.py` | `--tinker-session-server` (default off) mounts the routes on the served app; `--tinker-session-ttl-s` (default 3600), `--tinker-session-max-body-bytes` (default 16 MiB), `--tinker-tito-model` (a `TITOTokenizerType`, its fixed template replaces `--chat-template-path`), renders with `--apply-chat-template-kwargs` |
| client `harbor_env.py` | cookbook plug-ins: `HarborDatasetBuilder`, `HarborGroup` (rewards from Harbor verdicts), `SessionRolloutStrategy` (bind → Harbor trial → export → delete → `Trajectory`) |
| client `run_harbor_tinker.py` | `HarborTinkerConfig` → cookbook `train.Config` → `train.main`, with a sandbox preflight |

Status codes on the session routes: 400 bad input or missing key, 403 another tenant's session, 404 unknown session,
429 per-tenant session cap or per-session turn cap, 502 engine failure (nothing recorded).

## Run

1. **Gateway** (a node of the Ray cluster; see [`examples/multi_lora`](/examples/multi-lora) for the launcher):

   ```bash
   python3 examples/multi_lora/serve_qwen3_30b_a3b_tinker.py serve \
       --model-dir /models --save-dir <ckpt-dir> --output-dir <out-dir> --n-adapters 8 --lora-rank 16 --lora-alpha 32 \
       --extra-args "--use-miles-router --tinker-base-model Qwen/Qwen3-30B-A3B \
                     --tinker-session-server --tinker-tito-model qwen3 --apply-chat-template-kwargs '{\"enable_thinking\": false}' \
                     --max-tokens-per-gpu 32768 --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
                     --tinker-session-ttl-s 7200"
   ```

   - `--tinker-session-server` mounts the `/oai/sessions/*` routes (off by default: without it `serve_tinker.py` is the
     plain Tinker gateway, no tokenizer load, no extra routes, no sweep task).
   - `--tinker-tito-model qwen3` selects the Qwen3 fixed chat template; leave `--chat-template-path` unset with it. Drop the
     flag to re-render the full history every turn (one Datum per turn).
   - Keep `--tinker-train-unembed` on (the launcher's default): the cookbook creates its model with the SDK default
     `train_unembed=True`.
   - The gateway refuses a Datum longer than `min(model max_position_embeddings, --max-tokens-per-gpu)` and closes the
     model; agent trajectories run to 10–30k tokens, so pass `--max-tokens-per-gpu 32768` (with recompute) and give the
     client the same cap as `max_datum_tokens`.

2. **Client host** (on the tailnet that reaches the gateway and AgentENV):

   ```bash
   pip install "tinker==0.26.2" tinker-cookbook "harbor[e2b] @ git+https://github.com/harbor-framework/harbor@harbor-miles-v0.20.0"
   mkdir -p ~/.config/e2b && echo <key> > ~/.config/e2b/api_key && chmod 600 ~/.config/e2b/api_key
   git clone https://github.com/laude-institute/terminal-bench-2 ~/.cache/terminal-bench-2   # one task dir per task.toml
   ```

3. **Train**:

   ```bash
   HARBOR_ENV_TYPE=e2b E2B_API_URL=https://<agentenv> E2B_API_KEY_FILE=~/.config/e2b/api_key \
   HARBOR_TASKS_DIR=~/.cache/terminal-bench-2 TINKER_API_KEY=tml-<key> \
   python examples/multi_lora/harbor_tinker/run_harbor_tinker.py \
       gateway=http://<gateway>:10613 model_name=Qwen/Qwen3-30B-A3B tasks_dir=~/.cache/terminal-bench-2 \
       groups_per_batch=4 group_size=4 max_tokens=1536 max_seq_len=16384 max_datum_tokens=32768 lora_rank=16
   ```

   Every step runs `groups_per_batch × group_size` trials, each in its own sandbox. The preflight fails fast on a missing
   `HARBOR_ENV_TYPE` / `HARBOR_TASKS_DIR`, a missing provider key, an old e2b SDK, or an unreachable `E2B_API_URL`.
