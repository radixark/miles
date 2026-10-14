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
| `configs/` | Experiment configs: `prod_1node.yaml` (open-instruct's production recipe on one node) and 4-GPU smoke tests |
| `beaker/` | `submit.py` (config -> Beaker job, pinned to a commit), the job's entrypoint, and open-instruct's vendored podman bring-up |

## How an episode runs

The loop follows open-instruct's `vllm_utils.process_request` with the `swerl_vanillux_sandbox` env as run in its
production script (`--tool_call_format_error_feedback` off, `SWERL_RESET_FAILURE_ZERO_REWARD=1`):

1. `agentic_tool_call.generate` opens a TITO session and calls `tmax_agent_function.run` in a process of its own.
2. `run` starts a container from `metadata.image` (`sleep infinity`, 4 GB memory limit) on one of the podman
   daemons, uploads the task's `environment/seeds`, and installs a wrapper that keeps `cwd` and exported variables
   across commands. A failed start is retried 5 times (1-16 s backoff); a daemon that fails to connect is benched
   for 300-330 s. If it never starts, the episode scores 0 and stays in its group, through a one-token turn
   (`start_failure: drop` drops the group instead).
3. Up to `max_steps` (64) bash calls: a chat completion with the single `bash` tool through the session server,
   each command run with a 120 s timeout, output truncated head/tail at 10k characters and followed by
   `(exit_code=N)`. Calls to other tools are ignored; a turn without a bash call ends the episode. A command that
   fails for infra reasons is reported as `Step 'bash' failed: ...` and the episode goes on; a vanished container
   is recreated once.
4. The episode ends on the submit marker (`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`), which uploads `tests/`,
   runs `test.sh` (600 s) and reads `/logs/verifier/reward.txt`; or with reward 0 on no tool call, a
   length-truncated turn, a full context, the step limit, an OOM-killed sandbox, or `episode_timeout_s` (off).

Each finished episode logs `episode done: reward=... end=... turns=...`, and with `trace_dir` set appends a JSONL
record (messages, reward, metrics) to `<trace_dir>/<exp_name>/` (open-instruct's `--save_traces`).

## Run

Experiments are YAML configs of the launcher's fields, as open-instruct's `configs/train_configs/`: pass one to
`prepare` / `execute`, and command-line flags override it.

```bash
python examples/tmax/run_tmax_qwen3_5.py prepare examples/tmax/configs/smoke_sync.yaml   # download, convert
python examples/tmax/run_tmax_qwen3_5.py execute examples/tmax/configs/smoke_sync.yaml --num-rollout 1
```

The sandboxes need a Docker-compatible API socket (podman), whose containers can pull
`docker.io/hamishi740/swerl-tmax-v3:*`, and `pip install docker`. The launcher hands the sockets in
`TMAX_DOCKER_HOST` (comma-separated) to the rollout; `docker_host` overrides them.

### Beaker

```bash
python examples/tmax/beaker/submit.py examples/tmax/configs/prod_1node.yaml [--dry-run] [-- --num-rollout 10]
```

`submit.py` plays mason's role. From the config's `beaker:` section it writes the spec (cluster, GPUs, image,
weka mounts, secrets, and `preemptible` + `autoResume` + task retries) and pins the job to the current commit:
the task clones this repository at that commit, so retries and later edits to the checkout cannot change what it
runs. Commit the config before submitting. The task runs `beaker/tmax_beaker_entry.sh`, which installs podman
(apt) and runs open-instruct's own bring-up, vendored in `beaker/podman/`: `sandbox_shards` daemons with their
own storage, the docker.io mirror (`mirror_url`), the image janitor and Docker Hub credentials. It also keeps the
newest `keep_last_n_checkpoints` training checkpoints, then runs `prepare` + `execute`.

### Checkpoints, resume and logging

| | open-instruct production | Here |
|---|---|---|
| Training state | DeepSpeed state + dataloader + data-prep state every 5-10 steps in `--checkpoint_state_dir`, keep 3 | Megatron `torch_dist` (model, optimizer, RNG) + dataset position every `save_interval` steps in `<output_dir>/<exp_name>/checkpoints`, newest `keep_last_n_checkpoints` kept |
| Resume | mason `--auto_resume --max_retries 5`; reloads `--checkpoint_state_dir` | `autoResume` + `retries`; the same `exp_name` loads its checkpoint dir; in-flight rollouts are lost and regenerated |
| HF checkpoints | every 20 steps (`--save_freq`) | not exported during training; `--save-hf` would write one at every checkpoint |
| Rollouts | `--save_traces` JSONL with advantages and trainer logprobs | per-episode JSONL with messages, reward and metrics (no advantages or trainer logprobs) |
| W&B | `oe-general-agents`, one run per restart | `wandb_project`; with `exp_name`, all restarts log into one run |

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

On-policy distillation, `allenai/tmax-4b` into `hamishivi/Qwen3.5-4B` (`--opd-kl-coef 1.0`; these runs rendered
with `enable_thinking=false`, off the models' training distribution, so they validate the pipeline, not the numbers):

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
- **Thinking.** `allenai/tmax-4b`, `allenai/tmax-9b`, `hamishivi/Qwen3.5-4B`/`-9B` and stock Qwen3.5 all
  open `<think>` by default, as does Miles' `qwen35` template, and open-instruct trained with that default;
  only `allenai/tmax-2b` defaults to thinking off. The `_RECIPES` table keeps thinking on. (The OPD runs in
  the results below predate this and rendered with `enable_thinking=false`.) SGLang's Qwen3.5 reasoning
  detector treats `<tool_call>` as an implicit end of reasoning, so turns parse either way. vLLM's `qwen3`
  reasoning parser does not: for a model that emits no `</think>` it files the whole turn as reasoning and
  drops every tool call. Serve these checkpoints in vLLM with `--language-model-only` (they ship no
  image-processor config).

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
  pending episodes. Each episode runs in its own process (`--custom-agent-function-mode subproc`, the
  default), so `abort`, which runs in the rollout process, bumps a generation counter in a file scoped to
  that process and removes its containers; every in-flight model call and bash command races against the
  counter, and aborted episodes return within seconds. The episodes are not cancelled, because
  `as_completed_async` would re-raise the cancellation. Limits shared across episodes (start and exec
  concurrency, one pull per image) are `rollout_limits` file locks for the same reason.
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
- `openai_session_url` already appends `/v1`; agent functions pass its result to the OpenAI client as is.
- Qwen3.5 `torch_dist` checkpoints converted before the head-sharded linear-attention layout (upstream #3610)
  are rejected at load; re-convert them from HF into a new `--model-dir`.
- On B300 with TP=1 and no optimizer offload, Qwen3.5-9B training uses ~58 GB allocated per GPU, and a
  weight sync to the engines takes ~2.6 s.

### Known gaps

- No DPPO in Miles; add it as a custom loss (`--loss-type custom_loss`) if the ablation matters.
- No warm container pool: every episode pulls (cached per daemon), creates and removes its own container.
- Tool observations differ from open-instruct's `vllm_qwen3_xml` framing by two newline tokens per tool turn
  (Qwen3.5's chat template puts `\n` after the assistant's `<|im_end|>` and none before the tool's); the first-turn
  prompt is token-identical.
- Podman comes from Ubuntu's apt (4.9) at job start; open-instruct's image builds 5.6.2 and crun 1.14.3 from
  source. Bake them into a Miles image for production.
- Group-granular dropping: one infra failure discards its whole group.
