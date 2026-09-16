# exp: multi-LoRA Tinker gateway × Harbor agent RL (2026-09-16)

End-to-end agentic RL through PR #3286: tinker-cookbook `train.main` (unmodified) on a laptop, the multi-LoRA Tinker gateway
(#2846 + this PR) on one H200 node, Harbor terminus-2 trials in AgentENV sandboxes, every model call recorded as exact tokens
by the gateway's session collector. Nothing in the Tinker wire format changed.

## Setup

| Piece | Value |
|---|---|
| Gateway | `examples/multi_lora/serve_qwen3_30b_a3b_tinker.py serve` as a Ray job on the shared 4-node cluster (driver on 10.220.51.54, all 8 GPU bundles on 10.220.51.9); Qwen3-30B-A3B, 4 train GPUs (TP2 / EP4) + 4 sampling GPUs (2 SGLang engines × TP2), `--multi-lora-n-adapters 2`, rank 16, alpha 32, target modules incl. `output_layer` |
| Gateway extras | `--use-miles-router` (the Rust router drops `lora_backfill_paths`), patched SGLang tree on `PYTHONPATH` (`/personal/miles-pressure-test-20260910/src/sglang/python`, has `lora_backfill_paths`), `--tinker-base-model Qwen/Qwen3-30B-A3B`, `--chat-template-path` = Qwen3 template with `enable_thinking=false` baked in, `--tinker-session-ttl-s 7200` |
| Client | laptop (tailnet) → `ssh -L 10613:10.220.51.54:10613` → gateway; `run_harbor_tinker.py` with tinker 0.26.2, tinker-cookbook 0.5.7, harbor 0.20.0 (`harbor-miles-v0.20.0`), e2b 2.49.1 |
| Sandbox | AgentENV (internal E2B-compatible), `HARBOR_ENV_TYPE=e2b`, `E2B_API_URL=https://sandbox-service-control-plane.tail134ba0.ts.net`, key file `~/.config/e2b/api_key` |
| Tasks | Terminal-Bench-2 `difficulty = "easy"`: `cobol-modernization`, `fix-git`, `overfull-hbox`, `prove-plus-comm` (real directory copies, not symlinks) |
| Agent | terminus-2 (host process), `HARBOR_AGENT_MAX_ITERATIONS=12`, `AGENT_TIMEOUT=600`, `AGENT_TRIAL_TIMEOUT=900`, api_key `dummy` on a pre-bound session |

## Gates

- Sandbox control plane from the laptop: `GET /sandboxes` → HTTP 200 (25 running). Harbor oracle smoke (`scripts/sandbox_smoke/run.py --connector harbor --backend e2b`): reward 1.0 in 26.6 s (env setup 7.1 s, verifier 13.5 s).
- Gateway healthy 120 s after submission. S0 gate over the real gateway (`s0_gate.py`): `create_model` 1.8 s, `save_weights_for_sampler` 6.4 s, bind by `tinker://` path, three chat turns through adapter `M@V` (41/66/94 prompt tokens, 4/7/6 output tokens, first turn 5.2 s incl. adapter load, then 0.7 s / 0.2 s), turns exported, session deleted. 14.8 s total.
- Observation from the S0 turns: with the no-thinking template the re-rendered history does **not** extend the previous prompt+output (Qwen3's template emits `<think>\n\n</think>` only in the generation prompt, never for history), so consecutive turns do not chain and `trajectory_to_data` yields one Datum per turn. Same outcome as thinking-on for terminus-2 (history thinking stripped). Exact on-policy either way; the `tito_render_prompt` hook is where inheritance would go.

## Run 2 — 6 steps, 2 tasks × 4 trials per step (8 trials/step)

`groups_per_batch=2 group_size=4 epochs=3 max_steps=6 max_tokens=1024 max_seq_len=16384 lora_rank=16 learning_rate=1e-4 loss_fn=ppo temperature=1.0`

Totals: 47 trajectories recorded (one cobol trial errored out), 417 chat completions served by the gateway with 0 non-200 responses, 12 `forward_backward`, 6 `optim_step`, 11 sampler exports. Exit statuses: 33 Submitted, 11 SequenceLengthLimitExceeded (a reply hit the 1024-token turn cap → terminus-2 aborts the trajectory), 3 AgentError (AgentENV `snapshot … is not ready` while the `overfull-hbox` image template was still being built; gone by step 3).

## Per step (sampler version the trials sampled from)

| step | sampler | trials | tasks | reward mean | per-task reward | turns mean (max) | output tok / turn | final seq len mean (max) | aborted / failed |
|---|---|---|---|---|---|---|---|---|---|
| 0 | `1` | 8 | cobol-modernization, fix-git | 0.00 | cobol-modernization 0.00, fix-git 0.00 | 9.2 (max 12.0) | 323 (max 1024) | 11276 (max 24748) | 2 |
| 1 | `2` | 8 | overfull-hbox, prove-plus-comm | 0.25 | overfull-hbox 0.00, prove-plus-comm 0.50 | 9.4 (max 12.0) | 274 (max 419) | 5622 (max 10106) | 3 |
| 2 | `000002` | 8 | cobol-modernization, fix-git | 0.00 | cobol-modernization 0.00, fix-git 0.00 | 7.2 (max 12.0) | 302 (max 1024) | 5966 (max 9890) | 4 |
| 3 | `3` | 8 | overfull-hbox, prove-plus-comm | 0.00 | overfull-hbox 0.00, prove-plus-comm 0.00 | 10.8 (max 12.0) | 306 (max 1024) | 7348 (max 13863) | 2 |
| 4 | `000004` | 7 | cobol-modernization, fix-git | 0.00 | cobol-modernization 0.00, fix-git 0.00 | 8.7 (max 12.0) | 330 (max 1024) | 6796 (max 10922) | 3 |
| 5 | `4` | 8 | overfull-hbox, prove-plus-comm | 0.25 | overfull-hbox 0.00, prove-plus-comm 0.50 | 10.2 (max 12.0) | 322 (max 615) | 10684 (max 18465) | 0 |

## Cookbook step metrics

| batch | reward/total | turns/episode | ob tok/turn | ac tok/turn | KL sample-train (v1) | entropy | rollout s |
|---|---|---|---|---|---|---|---|
| 0 | 0.00 | 9.25 | 6997 | 323 | 0.0037 | 0.318 | 69 |
| 1 | 0.40 | 9.4 | 3381 | 274 | 0.0028 | 0.230 | 107 |
| 2 | 0.00 | 7.25 | 3938 | 302 | 0.0038 | 0.312 | 47 |
| 3 | 0.00 | 10.75 | 4162 | 306 | 0.0031 | 0.299 | 82 |
| 4 | 0.00 | 8.714285714285714 | 4074 | 330 | 0.0030 | 0.364 | 52 |
| 5 | 0.25 | 10.25 | 6168 | 322 | 0.0036 | 0.324 | 93 |

## Per-task reward by visit

| task | visit 0 | visit 1 | visit 2 | visit 3 | visit 4 | visit 5 |
|---|---|---|---|---|---|---|
| cobol-modernization | 0.00 (4) |  | 0.00 (4) |  | 0.00 (3) |  |
| fix-git | 0.00 (4) |  | 0.00 (4) |  | 0.00 (4) |  |
| overfull-hbox |  | 0.00 (4) |  | 0.00 (4) |  | 0.00 (4) |
| prove-plus-comm |  | 0.50 (4) |  | 0.00 (4) |  | 0.50 (4) |


### Sequence lengths

- Output per turn: mean 312 tokens, median 280; 11 of 408 turns hit the 1024 cap.
- Final sequence (last prompt + last output): mean 8,133 tokens, median 5,844, max 24,748 (fix-git, 12 turns).
- Prompt growth per turn: prompt_{k+1} − (prompt_k + output_k) median 322 tokens = the terminal observation plus template overhead. Example fix-git trajectory prompts: 861 → 2,325 → 3,353 → 4,128 → 5,102 → 8,687 → 9,568.
- Cookbook view: 7.3–10.8 turns per episode, observation 3.4k–7.0k tokens per turn (full re-render each turn), action 274–330 tokens per turn.

### Reward

- Flat at this scale: `prove-plus-comm` 0.50 / 0.00 / 0.50 on its three visits, the other three tasks 0.00 throughout. With 4 trials per task and mostly all-zero groups, GRPO has a non-zero advantage only in the `prove-plus-comm` groups with mixed outcomes; six steps are too few to show growth. Run 3 below raises trials per step, the turn cap and the learning rate.
- Sample-vs-train logprob KL 0.0028–0.0038 per step (the recorded logprobs match the trainer's recomputation of the same tokens: TITO is exact).

### Timing

- Rollout per step 47–107 s wall (8 concurrent trials, sandbox create ≈ 7 s each, verifier ≈ 15 s); train step 14–55 s (forward_backward on 24–47 per-turn Datums + optim + sampler export).

## Bugs found and fixed during the run (PR files only)

1. Task directory with **symlinks** → Harbor's `_task_path` resolves them outside `HARBOR_TASKS_DIR` and raises `instance_id … escapes HARBOR_TASKS_DIR`; every trial failed fast. Fix: real directory copies; documented on `HarborDatasetBuilder`.
2. The cookbook loop runs exactly `len(dataset)` steps: 4 tasks at 2 groups per batch = 2 steps regardless of `max_steps`. Fix: `epochs` on `HarborDataset` / `HarborDatasetBuilder` / `HarborTinkerConfig` (steps = ceil(tasks × epochs / groups_per_batch)).
3. Per-trajectory experiment log: `SessionRolloutStrategy(record_path=…)` writes one JSON line per trial (task, turns, per-turn prompt/output token counts, final length, reward, exit_status, sampler version); exposed as `record_path=` on the entry point. This file's tables come from it plus the cookbook's `metrics.jsonl`.

Not bugs in the PR, but worth knowing: (a) launching the example launcher from a checkout that is not first on `PYTHONPATH` makes it import the image's `/root/miles` and point the Ray job at `/root/miles/serve_tinker.py`; put the checkout first. (b) A client that exits keeps its adapter slot for the 300 s session lease; with `--multi-lora-n-adapters 2`, two quick test clients in a row make the third `create_model` fail with `no free adapter slots` until the sweeper frees them. (c) `overfull-hbox`'s first trials hit AgentENV `snapshot … is not ready` (template build in progress); later trials ran.

## Run 3 — 12 steps × 16 trials, crashed at the first training step (bug found and fixed)

`groups_per_batch=2 group_size=8 epochs=6 max_steps=12 max_tokens=2048 max_seq_len=24576 learning_rate=3e-4 HARBOR_AGENT_MAX_ITERATIONS=15`

Batch 0 sampled fine (16 trajectories: fix-git 8 × reward 0, cobol-modernization 8 × reward 0; 14 turns per episode, final
sequence mean 14,598 tokens, max 33,669; 1 of 224 turns hit the 2048 cap; 641 recorded chats in total on the gateway, still 0
non-200). The first `forward_backward` was then refused by the gateway, `model training failed (datum 4: 33669 tokens exceeds
32768)`: one turn's prompt+output exceeded the gateway's per-datum cap (`--tinker-max-tokens-per-datum`, 32768 by default),
the gateway closes the model on a failed training request, and the cookbook run died. Harbor's own `max_seq_len` (24,576 here)
did not stop the agent first because it counts tokens approximately.

Fix (PR files): `truncate_turns` in `harbor_env.py` keeps only the leading turns whose prompt+output fit `max_datum_tokens`
(default 32768, the gateway default) before building the `Trajectory`; the drop is logged and recorded per trajectory
(`dropped_turns`). Exposed as `max_datum_tokens=` on the entry point. Run 4 re-runs the configuration with it.

## Run 4 — 12 steps × 16 trials with the datum cap (completed after one gateway restart)

`groups_per_batch=2 group_size=8 epochs=6 max_steps=12 max_tokens=1536 max_seq_len=16384 max_datum_tokens=32768 learning_rate=3e-4 HARBOR_AGENT_MAX_ITERATIONS=12`

Steps 0–6 trained (126 trajectories, 1,927 recorded chats, 0 non-200, 54 `forward_backward`, 15 `optim_step`). At step 7's
`forward_backward` the **trainer went CUDA-OOM** (`MultiLoRATrainRayActor.forward_backward`: tried to allocate 3.8 GiB with 3.75 GiB
free, 136 GiB in use on one of the four TP2/EP4 training GPUs) on a batch whose longest per-turn Datum was 26,865 tokens
(`--max-tokens-per-gpu 8192` cannot split a sequence, and the example launcher runs without activation recompute). The Ray job
died with the actor, taking the gateway down with it; the four SGLang schedulers survived the job and had to be killed by hand
(they held ~104 GB each on node .9). Base-side finding, not PR code: a trainer OOM is fatal for every tenant of the gateway.

Recovery: gateway relaunched with `--recompute-granularity full --recompute-method uniform --recompute-num-layers 1` and the same
`--run-id` (same checkpoint root), run4 resumed by the cookbook from `tinker://…/weights/000006` with `max_datum_tokens=20480`
(turns longer than that are left out of training; the agent still runs them).

### Run 4 results (both halves)

Steps 0–7 are the first gateway process (sampler versions `1`…`6`/`000006`), steps 8–13 the resumed run on the relaunched gateway
(new model id, sampler versions restart at `1`; the cookbook re-sampled batch 6 after resuming, hence 13 metric rows for 12
batches). 222 trajectories, 217 Submitted / 5 SequenceLengthLimitExceeded, 25 turns dropped by the 20,480-token datum cap after
the resume, 2,761 recorded chats across both gateway processes with 0 non-200, 72 `forward_backward`, 21 `optim_step`.

## Per step (sampler version the trials sampled from)

| step | sampler | trials | tasks | reward mean | per-task reward | turns mean (max) | output tok / turn | final seq len mean (max) | aborted / failed |
|---|---|---|---|---|---|---|---|---|---|
| 0 | `1` | 16 | cobol-modernization, fix-git | 0.00 | cobol-modernization 0.00, fix-git 0.00 | 11.6 (max 12.0) | 321 (max 1207) | 10949 (max 20372) | 0 |
| 1 | `2` | 16 | overfull-hbox, prove-plus-comm | 0.19 | overfull-hbox 0.00, prove-plus-comm 0.38 | 9.7 (max 12.0) | 312 (max 1536) | 9249 (max 20071) | 1 |
| 2 | `3` | 15 | cobol-modernization, fix-git | 0.00 | cobol-modernization 0.00, fix-git 0.00 | 9.9 (max 12.0) | 313 (max 1536) | 9339 (max 22194) | 1 |
| 3 | `000003` | 15 | overfull-hbox, prove-plus-comm | 0.13 | overfull-hbox 0.00, prove-plus-comm 0.29 | 9.3 (max 12.0) | 280 (max 1092) | 9766 (max 21118) | 0 |
| 4 | `4` | 16 | cobol-modernization, fix-git | 0.00 | cobol-modernization 0.00, fix-git 0.00 | 11.4 (max 12.0) | 278 (max 1536) | 8942 (max 15178) | 1 |
| 5 | `5` | 16 | overfull-hbox, prove-plus-comm | 0.06 | overfull-hbox 0.00, prove-plus-comm 0.12 | 8.1 (max 12.0) | 279 (max 668) | 7499 (max 19990) | 0 |
| 6 | `000006` | 16 | cobol-modernization, fix-git | 0.06 | cobol-modernization 0.00, fix-git 0.12 | 10.3 (max 12.0) | 286 (max 1038) | 8716 (max 22069) | 0 |
| 7 | `6` | 16 | overfull-hbox, prove-plus-comm | 0.00 | overfull-hbox 0.00, prove-plus-comm 0.00 | 10.6 (max 13.0) | 313 (max 1176) | 12459 (max 26865) | 0 |
| 8 | `1` | 16 | cobol-modernization, fix-git | 0.00 | cobol-modernization 0.00, fix-git 0.00 | 11.7 (max 12.0) | 280 (max 1106) | 8057 (max 15348) | 0 |
| 9 | `2` | 16 | overfull-hbox, prove-plus-comm | 0.12 | overfull-hbox 0.00, prove-plus-comm 0.25 | 10.2 (max 12.0) | 289 (max 666) | 11122 (max 31739) | 0 |
| 10 | `3` | 16 | cobol-modernization, fix-git | 0.00 | cobol-modernization 0.00, fix-git 0.00 | 11.1 (max 12.0) | 279 (max 505) | 8475 (max 24154) | 0 |
| 11 | `000009` | 16 | overfull-hbox, prove-plus-comm | 0.00 | overfull-hbox 0.00, prove-plus-comm 0.00 | 8.4 (max 12.0) | 369 (max 1536) | 10858 (max 33078) | 1 |
| 12 | `4` | 16 | cobol-modernization, fix-git | 0.00 | cobol-modernization 0.00, fix-git 0.00 | 7.4 (max 12.0) | 310 (max 904) | 5196 (max 8538) | 0 |
| 13 | `5` | 16 | overfull-hbox, prove-plus-comm | 0.00 | overfull-hbox 0.00, prove-plus-comm 0.00 | 3.2 (max 9.0) | 384 (max 1536) | 4572 (max 22382) | 1 |

## Cookbook step metrics

| batch | reward/total | turns/episode | ob tok/turn | ac tok/turn | KL sample-train (v1) | entropy | rollout s |
|---|---|---|---|---|---|---|---|
| 0 | 0.00 | 11.625 | 5739 | 321 | 0.0029 | 0.287 | 340 |
| 1 | 0.19 | 9.6875 | 5422 | 312 | 0.0033 | 0.267 | 104 |
| 2 | 0.00 | 9.933333333333334 | 5288 | 313 | 0.0029 | 0.297 | 91 |
| 3 | 0.13 | 9.333333333333334 | 6438 | 280 | 0.0025 | 0.280 | 100 |
| 4 | 0.00 | 11.4375 | 5070 | 278 | 0.0027 | 0.229 | 93 |
| 5 | 0.06 | 8.0625 | 5257 | 279 | 0.0034 | 0.108 | 100 |
| 6 | 0.06 | 10.3125 | 4729 | 286 | 0.0030 | 0.112 | 70 |
| 6 | 0.00 | 11.6875 | 4388 | 280 | 0.0021 | 0.127 | 87 |
| 7 | 0.12 | 9.75 | 5791 | 282 | 0.0026 | 0.099 | 126 |
| 8 | 0.00 | 10.875 | 4583 | 279 | 0.0029 | 0.198 | 74 |
| 9 | 0.00 | 7.625 | 5174 | 362 | 0.0023 | 0.196 | 124 |
| 10 | 0.00 | 7.4375 | 3126 | 310 | 0.0025 | 0.279 | 65 |
| 11 | 0.00 | 3.1875 | 3811 | 381 | 0.0021 | 0.294 | 85 |

## Per-task reward by visit

| task | visit 0 | visit 1 | visit 2 | visit 3 | visit 4 | visit 5 | visit 6 | visit 7 | visit 8 | visit 9 | visit 10 | visit 11 | visit 12 | visit 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cobol-modernization | 0.00 (8) |  | 0.00 (7) |  | 0.00 (8) |  | 0.00 (8) |  | 0.00 (8) |  | 0.00 (8) |  | 0.00 (8) |  |
| fix-git | 0.00 (8) |  | 0.00 (8) |  | 0.00 (8) |  | 0.12 (8) |  | 0.00 (8) |  | 0.00 (8) |  | 0.00 (8) |  |
| overfull-hbox |  | 0.00 (8) |  | 0.00 (8) |  | 0.00 (8) |  | 0.00 (8) |  | 0.00 (8) |  | 0.00 (8) |  | 0.00 (8) |
| prove-plus-comm |  | 0.38 (8) |  | 0.29 (7) |  | 0.12 (8) |  | 0.00 (8) |  | 0.25 (8) |  | 0.00 (8) |  | 0.00 (8) |


### Sequence lengths

- Output per turn: mean 302 tokens, median 261; 5 of 2,111 turns hit the 1,536 cap (vs 11 of 408 at 1,024 in run 2).
- Final sequence: mean 8,937 tokens, median 6,938, max 33,078 — the tail is what overran the trainer before recompute was on.
- Turns per episode fell from ~10–11 to 7.4 and 3.2 in the last two steps: the agent started declaring the task done early.

### Reward

- No growth. `prove-plus-comm` 0.38 → 0.29 → 0.12 → 0.00 → 0.25 → 0.00 → 0.00 across its seven visits; `fix-git` one success in 56
  trials (visit 6); `cobol-modernization` and `overfull-hbox` 0 in every trial. Entropy fell 0.29 → 0.10 over the first seven steps
  (the policy sharpening on a near-constant reward) and recovered to ~0.2–0.29 after the resume from the step-6 checkpoint.
- Reading: with one solvable task out of four, 8 trials per task per step and lr 3e-4, GRPO gets a non-zero advantage only from the
  `prove-plus-comm` groups, and the signal it gets there (success ratio 0.12–0.38 on 8 samples) is noise-dominated; the later
  short episodes look like the start of collapse rather than learning. Sample-vs-train logprob KL stayed at 0.002–0.003 per step
  throughout, so the token path itself is exact; what is missing is reward signal, not fidelity.
- What a real reward curve needs (not run here): tasks the base model solves 20–60% of the time (so groups have variance), ≥16
  samples per task per step, lr ≤ 1e-4 for rank 16, and per-turn budgets that keep the longest datum under the trainer's memory
  (recompute on, or `max_datum_tokens` ≈ 16k).

### Timing

- Rollout per step 65–340 s wall with 16 concurrent trials (the 340 s first step includes template builds); train step 47–103 s
  (forward_backward over ~90–150 per-turn Datums totalling 0.6–1.3 M tokens, optim, sampler export).

## Reproduce

```bash
# gateway (submitting node on the cluster, checkout at /personal/thinker-session/miles)
export MILES_SCRIPT_EXTERNAL_RAY=1 RAY_ADDRESS=http://<head>:8265 NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0 \
       PYTHONPATH=/personal/thinker-session/miles:/personal/miles-pressure-test-20260910/src/sglang/python
python3 examples/multi_lora/serve_qwen3_30b_a3b_tinker.py serve --model-dir /cluster-storage/models \
    --save-dir /personal/thinker-session/ckpt --output-dir /personal/thinker-session/out --n-adapters 2 --lora-rank 16 --lora-alpha 32 \
    --extra-args "--use-miles-router --tinker-base-model Qwen/Qwen3-30B-A3B --chat-template-path /personal/thinker-session/qwen3_nothink.jinja --tinker-session-ttl-s 7200"

# client (laptop on the tailnet; ssh -N -L 10613:<driver>:10613 <devbox> in another shell)
HARBOR_ENV_TYPE=e2b E2B_API_URL=https://sandbox-service-control-plane.tail134ba0.ts.net E2B_SANDBOX_URL=http://sandbox-service-control-plane \
E2B_API_KEY_FILE=~/.config/e2b/api_key HARBOR_TASKS_DIR=<tb2-easy> HARBOR_AGENT_MAX_ITERATIONS=12 AGENT_TIMEOUT=600 AGENT_TRIAL_TIMEOUT=900 \
TINKER_API_KEY=tml-<key> python examples/multi_lora/harbor_tinker/run_harbor_tinker.py gateway=http://127.0.0.1:10613 \
    model_name=Qwen/Qwen3-30B-A3B tasks_dir=<tb2-easy> log_path=<log> groups_per_batch=2 group_size=4 epochs=3 max_steps=6 \
    max_tokens=1024 max_seq_len=16384 lora_rank=16 learning_rate=1e-4 record_path=<log>/trajectories.jsonl
```
