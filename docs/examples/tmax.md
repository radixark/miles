---
title: "TMax Terminal-RL"
description: "Terminal-RL on the TMax tasks with per-task Podman sandboxes: sync and fully-async GRPO, plus on-policy distillation."
# Generated from examples/tmax/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
GRPO and on-policy distillation on the 14.6k TMax terminal tasks
([`allenai/tmax-15k-open-instruct`](https://huggingface.co/datasets/allenai/tmax-15k-open-instruct)),
the Miles port of open-instruct's Terminal-RL recipe with the `swerl_vanillux_sandbox` environment.
Every episode runs in a fresh container from the task's prebuilt image, the only tool is `bash`, and the
reward is the task's `tests/test.sh` verdict, uploaded only at submit time so the policy cannot read it.

Three modes are validated end to end on 4x B300 (Beaker `ai2/holmes`):

| Mode | Launcher flags | Validated with |
|---|---|---|
| Synchronous GRPO, colocated | (default) | Qwen3.5-9B, 3 steps |
| Fully-async GRPO | `--async-rollout-gpus 2` | Qwen3.5-9B, 4 steps |
| On-policy distillation (sync and async) | `--opd-teacher tmax-4b` | `allenai/tmax-4b` into `hamishivi/Qwen3.5-4B`, 3 + 4 steps |

## Files

| File | Role |
|---|---|
| `run_tmax_qwen3_5.py` | Launcher: `prepare` (download, extract tasks, prompt JSONL, torch_dist conversion) and `execute` |
| `tmax_agent_function.py` | Agent loop for `agentic_tool_call.generate`, plus the over-sampling `abort` hook |
| `tmax_sandbox.py` | Per-episode container on a Docker-API socket (Podman), ported from open-instruct |
| `generate.py` | Reward function (`None` drops the group) and `agent/*` metrics |
| `tmax_prompts.py` | open-instruct's system prompt |
| `beaker/` | Beaker spec and entrypoint that install Podman in the task and run any of the modes |

## How an episode runs

1. `agentic_tool_call.generate` opens a TITO session and calls `tmax_agent_function.run`.
2. `run` starts a container from `metadata.image` (`sleep infinity`, 4 GB memory limit), uploads the task's
   `environment/seeds`, and installs a wrapper that keeps `cwd` and exported variables across commands.
   A container that cannot start raises before the first model call, so the sample is `ABORTED` and the
   group is dropped instead of training on a fake 0.
3. Up to `TMAX_MAX_STEPS` (64) bash calls: a chat completion with the single `bash` tool through the session
   server, each tool call run with a 120 s timeout, output truncated head/tail at 10k characters and
   followed by `(exit_code=N)`.
4. The episode ends on the submit marker (`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`), which uploads
   `tests/`, runs `test.sh` (600 s) and reads `/logs/verifier/reward.txt`; or with reward 0 on no tool call,
   a length-truncated turn, a full context, the step limit, an OOM-killed sandbox, or `TMAX_EPISODE_TIMEOUT_S`.
5. Sandbox failures after the first model call return `reward=None`, which the missing-reward filter turns
   into a dropped group. open-instruct scores these 0 (`SWERL_RESET_FAILURE_ZERO_REWARD=1`).

Each finished episode logs one line: `episode done: reward=... end=... turns=... tool_calls=... tokens=...`.

## Run

Requirements:

- A Docker-compatible API socket, normally `podman system service --time=0 unix:///run/podman/podman.sock`,
  whose containers can pull `docker.io/hamishi740/swerl-tmax-v3:*` (~400 MB each). Log in, or pass
  `TMAX_REGISTRY_USERNAME` / `TMAX_REGISTRY_PASSWORD`; unauthenticated Docker Hub pulls rate-limit at scale.
- `pip install docker` in the training image; the SDK is not shipped.

```bash
# download, extract, write prompts, convert checkpoints (pass the same --model-name / --opd-teacher as execute)
python examples/tmax/run_tmax_qwen3_5.py prepare --num-gpus-per-node 4
# synchronous, training and rollout colocated on every GPU
python examples/tmax/run_tmax_qwen3_5.py execute --num-gpus-per-node 4
# fully async: 2 GPUs for SGLang, the rest for Megatron, groups at most 4 weight versions stale
python examples/tmax/run_tmax_qwen3_5.py execute --num-gpus-per-node 4 --async-rollout-gpus 2
# on-policy distillation from allenai/tmax-4b (combines with --async-rollout-gpus)
python examples/tmax/run_tmax_qwen3_5.py execute --num-gpus-per-node 4 \
    --model-name Qwen3.5-4B-tmax-template --opd-teacher tmax-4b --no-filter-zero-std
```

On 80 GB+ GPUs add `--tensor-model-parallel-size 1 --no-optimizer-cpu-offload`. For a smoke test add
`--num-prompts 64 --rollout-batch-size 4 --n-samples-per-prompt 4 --num-rollout 3 --max-seq-len 16384
--no-filter-zero-std` (see "Zero-variance filter" below for why the filter has to be off).

### Beaker

`beaker/tmax_smoke.sh` is the entrypoint for `docker.io/radixark/miles:latest`. It installs Podman from apt,
writes open-instruct's container config (host network, cgroups disabled, `/proc` bind), starts the API
socket, logs in to Docker Hub, and runs `prepare` + `execute` for `MODE=sync|async|opd|opd-async`.
Models, datasets and outputs are cached under `TMAX_CACHE_DIR`. Edit the paths, budget and secret names in
`beaker/tmax_smoke.yaml`, then:

```bash
beaker experiment create --workspace ai2/oe-agents-holmes examples/tmax/beaker/tmax_smoke.yaml
```

The task needs `BEAKER_ALLOW_SUBCONTAINERS=1`; the sandboxes are nested containers inside it.

## Mapping from open-instruct

| open-instruct (`qwen35_9b_base_tmax_10k_dppo_32k.sh`) | Here |
|---|---|
| `--use_vllm_logprobs true` | `--use-rollout-logprobs` |
| `--advantage_normalization_type centered` | `grpo` + `--disable-grpo-std-normalization` |
| `--loss_denominator token` | `--calculate-per-token-loss` |
| `--active_sampling` | `--filter-zero-std` (default on), i.e. `apply_reward_nonzero_std_filter` |
| `--per_turn_max_tokens 8192` | `--rollout-max-response-len 8192` |
| `--response_length 32768` | `--max-seq-len 34816` (Miles counts the prompt) |
| `--max_steps 64` | `--max-steps 64` (`TMAX_MAX_STEPS`) |
| `--async_steps 4`, `--inflight_updates true` | `--async-rollout-gpus N --max-weight-staleness 4`, `--pause-generation-mode in_place` |
| `--system_prompt_override_file swerl_vanillux_sandbox_system_prompt.txt` | `tmax_prompts.TMAX_SYSTEM_PROMPT` (byte-identical) |
| `--loss_fn dppo` | not available in Miles; PPO clip (0.2 / 0.28) around the rollout log-probs instead |
| `EnvironmentPool` of 512 warm actors, sharded Podman daemons | one container per episode on one socket; no warm pool |

## Results (4x B300, 64-prompt subset, 4 prompts x 4 samples per step, 16k tokens per episode)

Synchronous GRPO, Qwen3.5-9B:

| Step | Episodes submitted | Mean episode time | pg_loss | grad_norm | train/rollout KL |
|---|---|---|---|---|---|
| 0 | 62% | 373 s | 0.014 | 0.36 | 3e-4 |
| 1 | 56% | 250 s | 0.0068 | 0.21 | 3e-4 |
| 2 | 25% | 109 s | -0.0038 | 0.21 | 3e-4 |

Fully-async GRPO, Qwen3.5-9B, 2 training + 2 rollout GPUs: 4 steps, grad_norm 0.40 / 0.28 / 0.23 / 0.22,
train/rollout KL 3e-4 to 4e-4, at most 1 weight version stale, no stale or aborted groups.

On-policy distillation, `allenai/tmax-4b` into `hamishivi/Qwen3.5-4B` (`--opd-kl-coef 1.0`):

| | Step 0 | Step 1 | Step 2 | Step 3 |
|---|---|---|---|---|
| Sync: teacher reverse KL per token | 0.178 | 0.168 | 0.150 | |
| Sync: grad_norm | 6.0 | 5.2 | 4.4 | |
| Async: teacher reverse KL per token | 0.150 | 0.187 | 0.202 | 0.104 |
| Async: grad_norm | 5.3 | 5.5 | 4.4 | 2.8 |
| Async: max staleness (versions) | 0 | 1 | 2 | 3 |

The step-0 reverse KL agrees with an independent measurement: scoring the student's own trajectories with
both models through vLLM `prompt_logprobs` gave 0.175 nats per assistant token (p90 0.61, 6.6% of tokens
above 1 nat).

## Findings

### Required settings

- **`--tito-model qwen35` does not set SGLang's parsers.** `resolve_reasoning_and_tool_call_parser` is only
  called from test utilities, so without `--sglang-reasoning-parser qwen3 --sglang-tool-call-parser
  qwen3_coder` the engines return the model's `<tool_call>` XML as plain content, `tool_calls` is empty, and
  every episode ends after one turn with reward 0. The launcher passes both.
- **Zero-variance filter.** Qwen3.5-9B is very consistent per task: in a local check, four tasks were solved
  4/4, 4/4, 0/4 and 0/4. With 4 samples per prompt almost every group is all-pass or all-fail, the filter
  drops it, and a synchronous step never fills (the progress bar sits at 0/16 while the loop cycles through
  the prompt pool without any error). Use 8 samples per prompt and a large prompt pool for real runs,
  `--no-filter-zero-std` for smoke tests, and always with OPD, where uniform groups still carry signal.
- **Thinking.** The `allenai/tmax-*` chat template (also on `hamishivi/Qwen3.5-*`) defaults to thinking
  off; Miles' `qwen35` template defaults to on. The `_RECIPES` table renders TMax-template checkpoints with
  `enable_thinking=false`, as they were trained. SGLang's Qwen3.5 reasoning detector treats `<tool_call>`
  as an implicit end of reasoning, so thinking-off turns still parse. vLLM's `qwen3` reasoning parser
  does not: it files the whole turn as reasoning and drops every tool call, so serve TMax checkpoints in
  vLLM without `--reasoning-parser`, and with `--language-model-only` (they ship no image-processor config).

### Throughput

- A synchronous step waits for its slowest episode. Most episodes take 1-6 minutes; 40-turn long-thinking
  ones took about 20. About 80% of episode wall time is sandbox execution, not generation. Cap episodes with
  `TMAX_EPISODE_TIMEOUT_S` (default 3600) or use `--async-rollout-gpus`.
- The first training step JIT-compiles the TileLang and Triton kernels for the linear-attention layers
  (a few minutes); SGLang engines also compile at start-up and log health-check failures meanwhile.

### Fully async

- `--fully-async` rejects `--colocate` and `--rollout-function-path`, so `agent/*` metrics are only logged
  in synchronous mode; the per-episode log lines are written in both.
- The abort hook matters only in synchronous mode: after over-sampling finishes a step, Miles waits for all
  pending episodes. `abort` signals an event that every in-flight model call and bash command races
  against, then removes this process's containers; aborted episodes return within ~10 s. The episodes are
  not cancelled, because `as_completed_async` would re-raise the cancellation.
- With OPD the trainer is the bottleneck on a 2 + 2 split (a 455 s step, 241 s of it the teacher forward),
  so staleness climbs one version per step and would reach `--max-weight-staleness` on the next. Give the
  trainer more GPUs than the generator for async OPD.

### On-policy distillation

- Megatron mode (`--opd-type megatron`) keeps the teacher as a CPU weight backup of the student's model and
  swaps it in for one forward pass per step, so teacher and student must share an architecture; the launcher
  asserts this. It leaves the task reward and the sandbox loop untouched, while SGLang mode replaces the
  reward function with OPD's own. With `--use-rollout-logprobs` the student term of the reverse-KL estimate
  is the rollout log-prob, the behavior policy's.
- `allenai/tmax-4b` and `hamishivi/Qwen3.5-4B` have identical architecture, vocabulary and merges;
  `tmax-4b` only adds seven unused audio/TTS special tokens (ids 248070-248076).
- The OPD term is dense: at `--opd-kl-coef 1.0` grad_norm is about 20x a GRPO-only step (4-6 vs 0.2-0.4)
  and is clipped to 1.0, so the update is mostly distillation. Use 0.1-0.3 to keep the task reward in play.

### Sandboxes and environment

- Podman inside a Beaker task: `apt install podman` (4.9) works with open-instruct's `containers.conf`; a
  `storage.conf` must set `runroot` and `graphroot`. docker-py consults the client's
  `~/.docker/config.json` credential helper on pull and fails if it is broken; `tmax_sandbox` always sends
  explicit (possibly empty) credentials so Podman uses its own login.
- GPU containers cannot be nested inside a Beaker task: Podman there can neither mount a fresh procfs nor
  create cgroups, and the `/proc` bind it falls back to belongs to another PID namespace, which makes
  `cuInit` fail with CUDA error 304. Run the Miles image as the Beaker task itself and nest only the CPU
  sandboxes. `radixark/miles:latest` is CUDA 13; use `latest-cu12` on 570-series drivers.
- `resolve_session_url` already appends `/v1`; agent functions pass its result to the OpenAI client as is.
- On B300 with TP=1 and no optimizer offload, Qwen3.5-9B training uses ~58 GB allocated per GPU, and a
  weight sync to the engines takes ~2.6 s.

### Known gaps

- No DPPO in Miles; add it as a custom loss (`--loss-type custom_loss`) if the ablation matters.
- No warm container pool, sharded Podman daemons or registry mirror; every episode pulls (cached per
  daemon), creates and removes its own container.
- Group-granular dropping: one infra failure discards its whole group.
