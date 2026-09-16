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

## Run 5 — one task with real base success, 16 samples per step, 16 steps: **reward rises 0.38 → ~0.8**

`tasks = prove-plus-comm only, groups_per_batch=1 group_size=16 epochs=16 max_steps=16 max_tokens=1536 max_seq_len=16384 max_datum_tokens=20480 learning_rate=1e-4 loss_fn=ppo HARBOR_AGENT_MAX_ITERATIONS=12`

Designed from the run-4 reading: pick the task the base policy solves ~30% of the time so every group has reward variance, give
GRPO 16 samples of it per step, and lower the learning rate. 255 trajectories (251 Submitted, 3 SequenceLengthLimitExceeded, 1
AgentError), 0 turns dropped by the datum cap, 16 training steps, all on the relaunched gateway (recompute on), no errors.

| step | reward (16 trials) | turns/episode | output tok/turn | entropy | KL sample-train | train step s |
|---|---|---|---|---|---|---|
| 0 | 0.38 | 10.1 | 268 | 0.255 | 0.0042 | 66 |
| 1 | 0.25 | 9.2 | 284 | 0.237 | 0.0026 | 58 |
| 2 | 0.50 | 9.8 | 263 | 0.253 | 0.0023 | 58 |
| 3 | 0.40 | 10.5 | 254 | 0.192 | 0.0027 | 62 |
| 4 | 0.44 | 9.7 | 275 | 0.143 | 0.0030 | 54 |
| 5 | 0.69 | 9.9 | 255 | 0.117 | 0.0036 | 56 |
| 6 | 0.44 | 10.8 | 276 | 0.122 | 0.0034 | 65 |
| 7 | 0.88 | 8.9 | 258 | 0.100 | 0.0046 | 46 |
| 8 | 0.81 | 10.1 | 310 | 0.091 | 0.0040 | 62 |
| 9 | 0.62 | 10.3 | 301 | 0.090 | 0.0044 | 62 |
| 10 | 0.69 | 10.1 | 315 | 0.085 | 0.0034 | 61 |
| 11 | 0.88 | 9.5 | 311 | 0.083 | 0.0044 | 59 |
| 12 | 0.88 | 8.4 | 321 | 0.085 | 0.0044 | 50 |
| 13 | 0.73 | 9.3 | 316 | 0.079 | 0.0043 | 52 |
| 14 | 0.81 | 9.4 | 316 | 0.073 | 0.0038 | 55 |
| 15 | 0.44 | 10.5 | 354 | 0.079 | 0.0027 | 75 |

- Mean reward over steps 0–3: 0.38; over steps 7–14: 0.79. Steps 7, 11 and 12 reached 0.88 (14/16). The last step dipped to
  0.44, single-step noise on 16 samples (the per-step standard error is ≈ 0.1).
- Entropy fell 0.26 → 0.07 as the policy sharpened; sample-vs-train logprob KL stayed 0.002–0.005 throughout (exact token path).
- Sequence lengths stayed short and stable: 9.7 turns per episode, 292 output tokens per turn, final sequence mean 5,468 tokens
  (median 5,311, max 11,079); no turn hit the datum cap. Train step 46–75 s, rollout (16 concurrent trials) about 2 minutes.
- This is one task learned in isolation, i.e. a demonstration that the whole loop (recorded turns → Datums → LoRA update → new
  sampler version → next rollouts) moves the policy in the right direction, not a claim about generalisation.

## Per step (sampler version the trials sampled from)

| step | sampler | trials | tasks | reward mean | per-task reward | turns mean (max) | output tok / turn | final seq len mean (max) | aborted / failed |
|---|---|---|---|---|---|---|---|---|---|
| 0 | `1` | 16 | prove-plus-comm | 0.38 | prove-plus-comm 0.38 | 10.1 (max 12.0) | 268 (max 506) | 5013 (max 6926) | 0 |
| 1 | `2` | 16 | prove-plus-comm | 0.25 | prove-plus-comm 0.25 | 9.2 (max 12.0) | 284 (max 533) | 5198 (max 10466) | 0 |
| 2 | `3` | 16 | prove-plus-comm | 0.50 | prove-plus-comm 0.50 | 9.8 (max 12.0) | 263 (max 476) | 5046 (max 7119) | 0 |
| 3 | `4` | 15 | prove-plus-comm | 0.40 | prove-plus-comm 0.40 | 10.5 (max 12.0) | 254 (max 565) | 5515 (max 9293) | 0 |
| 4 | `000004` | 16 | prove-plus-comm | 0.44 | prove-plus-comm 0.44 | 9.7 (max 12.0) | 275 (max 1536) | 5339 (max 7695) | 1 |
| 5 | `5` | 16 | prove-plus-comm | 0.69 | prove-plus-comm 0.69 | 9.9 (max 12.0) | 255 (max 409) | 5247 (max 7288) | 0 |
| 6 | `6` | 16 | prove-plus-comm | 0.44 | prove-plus-comm 0.44 | 10.8 (max 12.0) | 276 (max 642) | 5564 (max 7194) | 0 |
| 7 | `7` | 16 | prove-plus-comm | 0.88 | prove-plus-comm 0.88 | 8.9 (max 12.0) | 258 (max 451) | 4667 (max 6503) | 0 |
| 8 | `000008` | 16 | prove-plus-comm | 0.81 | prove-plus-comm 0.81 | 10.1 (max 12.0) | 310 (max 575) | 5879 (max 7703) | 0 |
| 9 | `8` | 16 | prove-plus-comm | 0.62 | prove-plus-comm 0.62 | 10.3 (max 12.0) | 301 (max 597) | 5765 (max 8496) | 0 |
| 10 | `9` | 16 | prove-plus-comm | 0.69 | prove-plus-comm 0.69 | 10.1 (max 12.0) | 315 (max 509) | 5815 (max 8101) | 0 |
| 11 | `10` | 16 | prove-plus-comm | 0.88 | prove-plus-comm 0.88 | 9.5 (max 12.0) | 311 (max 525) | 5614 (max 8506) | 0 |
| 12 | `000012` | 16 | prove-plus-comm | 0.88 | prove-plus-comm 0.88 | 8.4 (max 12.0) | 321 (max 1536) | 4970 (max 7199) | 1 |
| 13 | `11` | 16 | prove-plus-comm | 0.69 | prove-plus-comm 0.69 | 9.3 (max 12.0) | 316 (max 524) | 5583 (max 8028) | 1 |
| 14 | `12` | 16 | prove-plus-comm | 0.81 | prove-plus-comm 0.81 | 9.4 (max 12.0) | 316 (max 714) | 5695 (max 11079) | 0 |
| 15 | `13` | 16 | prove-plus-comm | 0.44 | prove-plus-comm 0.44 | 10.5 (max 12.0) | 354 (max 1536) | 6594 (max 9624) | 1 |

## Cookbook step metrics

| batch | reward/total | turns/episode | ob tok/turn | ac tok/turn | KL sample-train (v1) | entropy | rollout s |
|---|---|---|---|---|---|---|---|
| 0 | 0.38 | 10.125 | 3028 | 268 | 0.0042 | 0.255 | None |
| 1 | 0.25 | 9.25 | 3337 | 284 | 0.0026 | 0.237 | None |
| 2 | 0.50 | 9.75 | 3064 | 263 | 0.0023 | 0.253 | None |
| 3 | 0.40 | 10.533333333333333 | 3267 | 254 | 0.0027 | 0.192 | None |
| 4 | 0.44 | 9.6875 | 3058 | 275 | 0.0030 | 0.143 | None |
| 5 | 0.69 | 9.9375 | 2991 | 255 | 0.0036 | 0.117 | None |
| 6 | 0.44 | 10.75 | 3190 | 276 | 0.0034 | 0.122 | None |
| 7 | 0.88 | 8.875 | 2719 | 258 | 0.0046 | 0.100 | None |
| 8 | 0.81 | 10.125 | 3274 | 310 | 0.0040 | 0.091 | None |
| 9 | 0.62 | 10.3125 | 3201 | 301 | 0.0044 | 0.090 | None |
| 10 | 0.69 | 10.0625 | 3271 | 315 | 0.0034 | 0.085 | None |
| 11 | 0.88 | 9.5 | 3145 | 311 | 0.0044 | 0.083 | None |
| 12 | 0.88 | 8.375 | 2874 | 321 | 0.0044 | 0.085 | None |
| 13 | 0.73 | 9.333333333333334 | 3177 | 316 | 0.0043 | 0.079 | None |
| 14 | 0.81 | 9.4375 | 3116 | 316 | 0.0038 | 0.073 | None |
| 15 | 0.44 | 10.5 | 3614 | 354 | 0.0027 | 0.079 | None |

## Per-task reward by visit

| task | visit 0 | visit 1 | visit 2 | visit 3 | visit 4 | visit 5 | visit 6 | visit 7 | visit 8 | visit 9 | visit 10 | visit 11 | visit 12 | visit 13 | visit 14 | visit 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prove-plus-comm | 0.38 (16) | 0.25 (16) | 0.50 (16) | 0.40 (15) | 0.44 (16) | 0.69 (16) | 0.44 (16) | 0.88 (16) | 0.81 (16) | 0.62 (16) | 0.69 (16) | 0.88 (16) | 0.88 (16) | 0.69 (16) | 0.81 (16) | 0.44 (16) |

## Run 6 — 8 LoRAs at once: 8 tenants × (prove-plus-comm, 8 samples per step, 8 steps)

Gateway relaunched with `--n-adapters 8` (recompute on); Ray put all 8 GPU bundles on node .7 this time. Eight independent
cookbook clients, keys `tml-tenant-01…08`, each with its own adapter, its own sessions and its own trajectories, started 10 s
apart from the same laptop; per tenant `groups_per_batch=1 group_size=8 epochs=8 max_steps=8 max_tokens=1536 max_seq_len=16384
max_datum_tokens=20480 learning_rate=1e-4`, so 64 terminus-2 trials ran concurrently in AgentENV every step.

**Everything held.** 8/8 `create_model` accepted, 496 trajectories, all `Submitted`, 0 turns dropped, 4,842 recorded chats with 0
non-200, 0 × 429, 0 × 5xx, 92 `forward_backward`, 64 `optim_step`, 80 sampler exports, 8/8 clients finished all 8 steps; 42 minutes
wall clock first to last trajectory. No cross-tenant error of any kind: each session bound to its owner's sampler version, foreign
keys never seen (the harness uses the placeholder key on pre-bound sessions).

| tenant | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | mean first 2 | mean last 2 | trajectories |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t01 | 0.62 | 0.38 | 0.50 | 0.25 | 0.25 | 0.80 | 0.12 | 0.25 | 0.50 | 0.19 | 61 |
| t02 | 0.25 | 0.75 | 0.50 | 0.12 | 0.62 | 0.25 | 0.50 | 0.62 | 0.50 | 0.56 | 64 |
| t03 | 0.50 | 0.38 | 0.38 | 0.75 | 0.62 | 0.83 | 0.62 | 0.62 | 0.44 | 0.62 | 62 |
| t04 | 0.12 | 0.50 | 0.38 | 0.38 | 0.88 | 0.17 | 0.71 | 0.12 | 0.31 | 0.42 | 61 |
| t05 | 0.50 | 0.38 | 0.00 | 0.38 | 0.00 | 0.12 | 0.12 | 0.12 | 0.44 | 0.12 | 60 |
| t06 | 0.50 | 0.38 | 0.88 | 0.57 | 0.38 | 0.57 | 0.50 | 0.62 | 0.44 | 0.56 | 62 |
| t07 | 0.12 | 0.25 | 0.62 | 0.75 | 0.50 | 0.88 | 0.88 | 0.88 | 0.19 | 0.88 | 62 |
| t08 | 0.25 | 0.75 | 0.38 | 0.50 | 0.62 | 0.62 | 0.62 | 0.62 | 0.50 | 0.62 | 64 |
| **mean of tenants** | **0.36** | **0.47** | **0.45** | **0.46** | **0.48** | **0.53** | **0.51** | **0.48** | | | |

- Per-tenant curves are noisy (8 samples per step, standard error ≈ 0.17): t07 climbs 0.12/0.25 → 0.88 for its last three steps,
  t03 0.44 → 0.62, t08 → 0.62, while t05 falls to 0.12 and t01 to 0.19. The mean over tenants moves 0.36 → 0.53 (step 5) → 0.48.
  With 8 samples GRPO's per-step estimate is too noisy for a clean curve; run 5 (16 samples, one tenant) is the cleaner
  demonstration of learning. Entropy collapsed in two tenants (t02 0.24 → 0.06, t06 0.23 → 0.11) and stayed ~0.2–0.3 in the others.
- Sequence lengths: 9.4 turns per episode, 280 output tokens per turn, final sequence mean 5,121 tokens (median 5,004, max 16,441).
- GPU layout on the node (nvidia-smi during step 1): trainer GPUs 0–3 at ~68 GB / 78–80 % util each with 8 slots resident, engine
  GPUs 4–7 at 103 GB / 100 % util (two TP2 engines decoding 64 concurrent conversations).
- Timing: per tenant a step took 150–430 s wall; `train_step` 20–183 s. The spread is the trainer serialising eight tenants'
  forward_backward / optim / export behind one lock (a tenant that arrives while another trains waits), the rollouts overlap.
- Disk: the checkpoint root grew to 522 GB across all runs (sampler export + state per save, 8 tenants × 8 steps here); prune
  old versions before a longer multi-tenant run.

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
