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

Validated end to end on B300s (Beaker `ai2/holmes`):

| Mode | Launcher flags | Validated with |
|---|---|---|
| Synchronous GRPO, colocated, 4 GPUs | (default) | Qwen3.5-9B, 3 steps |
| Fully-async GRPO, 4 GPUs | `--async-rollout-gpus 2` | Qwen3.5-9B, 4 steps |
| On-policy distillation (sync and async), 4 GPUs | `--opd-teacher tmax-4b` | `allenai/tmax-4b` into `hamishivi/Qwen3.5-4B`, 3 + 4 steps |
| Pure on-policy distillation, fully async, 4 H100s (Jupiter) | `configs/smoke_opd_4b_async_jupiter_fused.yaml` | `allenai/tmax-4b` into `hamishivi/Qwen3.5-4B`, 3 steps, 2026-10-23 |
| open-instruct's production recipe, fully async on 4 nodes (32x B300), 64k context | `configs/prod_4node.yaml` | Qwen3.5-9B, training on Beaker since 2026-10-16 |

| Read | For |
|---|---|
| This page | What the example does, how an episode runs, how to run it, results, findings |
| [beaker/README.md](/examples/tmax/beaker) | Running on Beaker: submissions, run names and resume, images, mirrors, watching a run, failure triage |
| [DEVELOPING.md](https://github.com/radixark/miles/blob/main/examples/tmax/DEVELOPING.md) | How the port plugs into Miles, local development and tests, the validation ladder, conventions, history |
| [ROADMAP.md](https://github.com/radixark/miles/blob/main/examples/tmax/ROADMAP.md) | Open work by priority, decisions taken |
| [INVESTIGATIONS.md](https://github.com/radixark/miles/blob/main/examples/tmax/INVESTIGATIONS.md) | Investigations from the production runs: evidence, options, open decisions |
| [Miles vs open-instruct](/deep-dive/open-instruct-comparison) | Every framework, harness and infrastructure difference, and the flag-by-flag mapping of the production recipe |

## Files

| File | Role |
|---|---|
| `run_tmax_qwen3_5.py` | Launcher: `prepare` (download, extract tasks, prompt JSONL, torch_dist conversion) and `execute` |
| `tmax_agent_function.py` | Agent loop for `agentic_tool_call.generate`, plus the over-sampling `abort` hook |
| `tmax_sandbox.py` | Per-episode container: the episode logic shared by both backends, and the podman backend (a Docker-API socket), ported from open-instruct |
| `tmax_gateway.py` | The gateway backend: containers on a LiteRegistry podman fleet through literegistry's own client, after open-instruct's `GatewayBackend` |
| `generate.py` | Reward functions (`None` drops the group; with an SGLang OPD teacher, also the teacher's log-probs), and pure OPD's reward post-process |
| `tmax_metrics.py` | Per-rollout W&B metrics open-instruct logs: episodes and tools, solve rates before the filter, trained-sample lengths and advantages |
| `tmax_prompts.py` | open-instruct's system prompt |
| `configs/` | Experiment configs: open-instruct's production recipe on one node (`prod_1node.yaml`) and on its 4 nodes (`prod_4node.yaml`), and 4-GPU smoke tests (`smoke_2node.yaml` spans 2 nodes, `smoke_gateway.yaml` uses the podman fleet, `smoke_opd*.yaml` distill) |
| `beaker/` | `submit.py` (config -> Beaker job, pinned to a commit), the job's entrypoint, open-instruct's vendored podman bring-up, and the registry mirror tooling |
| `tools/` | `watch_run.py` (one line per step, slow sync or failure of a running Beaker experiment) and `check_samples.py` (trained samples vs episode traces) |

## How an episode runs

The loop follows open-instruct's `vllm_utils.process_request` with the `swerl_vanillux_sandbox` env as run in its
production script (`--tool_call_format_error_feedback` off, `SWERL_RESET_FAILURE_ZERO_REWARD=1`):

1. `agentic_tool_call.generate` opens a TITO session and calls `tmax_agent_function.run` in a process of its own.
2. `run` starts a container from `metadata.image` (`sleep infinity`, a 4 GB memory limit requested) on one of the podman
   daemons, uploads the task's `environment/seeds`, and installs a wrapper that keeps `cwd` and exported variables
   across commands. A failed start is retried 5 times (1-16 s backoff); a daemon that fails to connect is benched
   for 300-330 s. If it never starts, the episode scores 0 and stays in its group, through a one-token turn
   (`start_failure: drop` drops the group instead).
3. Up to `max_steps` (64) bash calls: a chat completion with the single `bash` tool through the session server,
   each command run with a 120 s timeout, output truncated head/tail at 10k characters and followed by
   `(exit_code=N)`. Calls to other tools are ignored; a turn without a bash call ends the episode. A command that
   fails for infra reasons is reported as `Step 'bash' failed: ...` and the episode goes on; a vanished container
   is recreated once. Each step, a submit's test run included, is capped by `tool_call_timeout_s` (120 s, open-instruct's
   `tool_call_timeout`): an overrun is reported as `Step 'bash' timed out ...` and the episode goes on, which also
   bounds a hung podman call. As in open-instruct's production runs, a test run longer than 120 s therefore cannot
   earn the reward; raise `tool_call_timeout_s` to stop reproducing that.
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

Sandboxes come from one of two backends (`sandbox_backend`):

- **`podman`** (the default) runs containers on Podman daemons the job brings up itself. It needs a
  Docker-compatible API socket whose containers can pull `docker.io/hamishi740/swerl-tmax-v3:*`, and
  `pip install docker`. The launcher hands the sockets in `TMAX_DOCKER_HOST` (comma-separated) to the rollout;
  `docker_host` overrides them.
- **`gateway`** runs containers on a LiteRegistry podman fleet through its gateway (`gateway_url`, or
  `TMAX_GATEWAY_URL`), so no sandbox shares a node with training or SGLang. On Beaker the head starts the gateway
  itself ([Sandboxes on a podman fleet](/examples/tmax/beaker#sandboxes-on-a-podman-fleet-gateway-backend)).

### Beaker

```bash
python examples/tmax/beaker/submit.py examples/tmax/configs/prod_4node.yaml [--dry-run] [-- --num-rollout 10]
```

`submit.py` turns a config's `beaker:` section into a Beaker job pinned to the current commit; its entrypoint brings
up open-instruct's podman setup on every node and runs `prepare` + `execute`. Each submission is its own run
(`<exp_name>_YYYYMMDD_HHMMSS`), and its Beaker retries resume from that run's checkpoints. Everything else (the
`beaker:` keys, multi-node layout, images, pull credentials and mirrors, checkpoints, watching a run, failures) is in
[beaker/README.md](/examples/tmax/beaker).

## Mapping from open-instruct

Every flag of open-instruct's production script and its counterpart here is in the
[comparison's flag-by-flag table](/deep-dive/open-instruct-comparison#flag-by-flag-the-production-recipe).
The deliberate algorithmic difference is the loss: open-instruct trains with DPPO, which Miles does not have, so
the port uses PPO clip (0.2 / 0.28) around the rollout log-probs.

## Results

### Gateway backend with open-instruct parity, 4 nodes

`configs/prod_4node_gateway.yaml` on 4x 8 B300 (`ai2/holmes`), run `tmax_prod4node_gateway_qwen35_9b_20261020_092350`
(Beaker `01M5C9ZQAM9J6P7D5FE9ZY4E43`, commit `7422022127`): the 4-node recipe below with sandboxes on the
`podman-terminal-jupiter-head` LiteRegistry fleet, DPPO, Adam beta2 0.999 and seed 42. Training compared with
open-instruct's `9b_dppo_repro_jupiter` (W&B `ut64ii7s`, `gyw0z542`) at the same steps:

| Steps | Reward (pre-filter), open-instruct / ours | Response length, open-instruct / ours | Entropy | KL to base | Train/rollout gap | Lost sandboxes |
|---|---|---|---|---|---|---|
| 0-19 | 0.521 / 0.507 | 13.6k / 16.4k | 0.31 | 0.002 | 0.0072 | 0.7% |
| 20-39 | 0.557 / 0.563 | 18.3k / 19.9k | 0.32 | 0.003 | 0.0080 | 1.4% |
| 40-59 | 0.563 / 0.544 | 17.3k / 20.9k | 0.31 | 0.006 | 0.0076 | 0.5% |
| 60-79 | 0.606 / 0.666 | 16.5k / 18.9k | 0.29 | 0.009 | 0.0071 | 0.5% |
| 80-99 | 0.620 / 0.619 | 15.2k / 20.4k | 0.28 | 0.011 | 0.0072 | 0.4% |
| 100-119 | 0.631 / 0.643 | 14.2k / 20.5k | 0.27 | 0.018 | 0.0081 | 0.3% |
| 120-139 | 0.587 / 0.627 | 17.9k / 22.3k | 0.31 | 0.020 | 0.0089 | 1.1% |
| 140-159 | 0.630 / 0.662 | 24.8k / 22.2k | 0.32 | 0.022 | 0.0084 | 0.4% |
| 160-179 | 0.631 / 0.627 | 23.6k / 27.9k | 0.32 | 0.023 | 0.0082 | 0.9% |
| 180-199 | 0.623 / 0.680 | 25.8k / 25.9k | 0.34 | 0.028 | 0.0086 | 0.4% |
| 200-219 | 0.625 / 0.669 | 26.0k / 24.7k | 0.36 | 0.050 | 0.0093 | 0.3% |
| 220-239 | 0.656 / 0.644 | 30.9k / 29.6k | 0.37 | 0.040 | 0.0095 | 0.3% |
| 240-259 | 0.567 / 0.670 | 23.5k / 26.0k | 0.39 | 0.044 | 0.0099 | 2.5% |
| 260-279 | 0.624 / 0.650 | 35.9k / 24.6k | 0.40 | 0.045 | 0.0099 | 0.2% |
| 280-299 | 0.634 / 0.668 | 35.1k / 23.9k | 0.44 | 0.071 | 0.0114 | 0.7% |
| 300-311 | 0.665 / 0.653 | 31.2k / 26.7k | 0.49 | 0.047 | 0.0116 | 0.4% |

- **Sandboxes.** No episode failed to start in 312 steps. Lost sandboxes are mostly the policy's own memory and
  process-limit kills (scored 0); fleet losses drop their group.
- **Preemptions.** Six in two days: five by workspace-group fair share just after the 8 h minimum runtime, one by a
  node drained for reimaging. The head closed the 263-432 containers it held on the fleet each time, and the next
  attempt resumed from the latest checkpoint (`save_interval: 5`).
- **Speed.** About 7 minutes per step.
- **Rising entropy.** Entropy climbs from step 260, with the train/rollout gap.

Greedy evaluations of the kept HF snapshots (k=5, 2 GPUs, `max-model-len` 67584, tmax repo `c98902cc`).
Open-instruct's and the base model's numbers are sampled (T=0.7), so the comparison is approximate:

| Model | TBlite pass@1 / pass@5 | TB2.1 pass@1 / pass@5 |
|---|---|---|
| Ours, step 120 | 0.496 / 0.700 | 0.229 / 0.360 |
| Ours, step 160 | 0.484 / 0.710 | 0.222 / 0.382 |
| Ours, step 200 | 0.472 / 0.720 | pending |
| Ours, step 240 | 0.506 / 0.770 | pending |
| Ours, step 280 | 0.490 / 0.690 | pending |
| Ours, step 300 | 0.478 / 0.680 | pending |
| Open-instruct, step 120 / 240 (sampled) | 0.524 / 0.760, 0.574 / 0.790 | 0.229 / 0.360, 0.249 / 0.393 |
| Base `hamishivi/Qwen3.5-9B` (sampled) | 0.418 / 0.660 | 0.191 / 0.337 |

The TBlite evals of steps 200-300 lost 10 trials each (the same 8 tasks) to Docker Hub's anonymous pull limit;
without those tasks step 240 scores 0.522 / 0.793 and the others 0.487-0.496 pass@1.

### Production recipe, 4 nodes

`configs/prod_4node.yaml` on 4x 8 B300 (`ai2/holmes`), run `tmax_prod4node_qwen35_9b_20261016_001037` (commit
`edf2c7ae3e`, session server v2): 8 prompts x 32 samples per step, 64k tokens per episode, fully async with 16
SGLang engines and 512 episodes in flight.

| Step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| loss | 0.022 | 0.025 | 0.012 | 0.032 | 0.008 | 0.042 | 0.027 | 0.057 | 0.025 | 0.044 | 0.030 |
| grad_norm | 0.084 | 0.082 | 0.051 | 0.076 | 0.101 | 0.066 | 0.079 | 0.086 | 0.076 | 0.066 | 0.056 |
| `rollout/raw_reward_unfiltered` | 0.41 | 0.40 | 0.65 | 0.43 | 0.29 | 0.40 | 0.27 | 0.50 | 0.38 | 0.34 | |
| `rollout/fully_async/avg_staleness` | 0 | 1 | 2 | 3 | 4 | 3 | 1.6 | 1.9 | 2.1 | 2.1 | |

- **Speed.** Step 0 trained 41 minutes after the replicas started, then a step about every 5 minutes.
- **Weight syncs** took 1.5-8 s, with an occasional 30-50 s one.
- **Train/rollout agreement.** `train/train_rollout_logprob_abs_diff` stayed at 0.0066-0.0068.
- **Session-collect failures:** none.
- **Episodes.** Over the first 5,705:
  - mean reward 0.38, mean 38 turns;
  - ended by submitting 62%, at the 64-step limit 30%, with no tool call 7%, with a full context 1.3%, on a
    length-truncated turn 0.1%;
  - 0.19 failed or timed-out tool steps per episode;
  - median episode 4 minutes (p90 10), 48% of it in model calls and 19% in sandbox commands.

### Smoke tests (4x B300, 64-prompt subset, 4 prompts x 4 samples per step, 16k tokens per episode)

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

Pure on-policy distillation on today's code (2026-10-23, `smoke_opd_4b_async_jupiter_fused.yaml`, Beaker
`01M5K694GXJ85TX6RZ9A014QTH`): `allenai/tmax-4b` into `hamishivi/Qwen3.5-4B` with thinking on, `opd_pure`, DPPO, session
server v2, sandboxes on the podman fleet, fully async on 2 + 2 of Jupiter's H100s through the CUDA
forward-compatibility driver:

| | Step 0 | Step 1 | Step 2 |
|---|---|---|---|
| Teacher reverse KL per token (`train/opd_reverse_kl`) | 0.179 | 0.265 | 0.153 |
| Advantage (`rollout/advantages`) | -0.183 | -0.272 | -0.178 |
| Task reward, logged only (`rollout/raw_reward`) | 0.81 | 0.375 | 0.375 |
| grad_norm | 4.8 | 6.2 | 3.9 |
| train/rollout log-prob gap | 0.0075 | 0.0099 | 0.0087 |
| KL to the base model (k3) | 0 | 0.0004 | 0.0005 |
| Max staleness (versions) | 0 | 1 | 2 |

The trained reward is 0 at every step and the advantage is minus the batch's reverse KL, so the OPD term is all
that trains. 59 episodes, 0 sandbox failures; 28 of them filled the 16k context. The teacher forward took 1.5-2 s
of each step, weight syncs 0.2-0.3 s. The first submission of this smoke, on the default log-prob backend, trained
step 0 and ran out of memory at step 1 (`01M5K4WN5P1HCVA1N2AY8C8MTS`; see Findings).

For the 2B pair (`smoke_opd_2b.yaml`, not yet run on Beaker), scoring `hamishivi/Qwen3.5-2B`'s own trajectories
(8 tasks x 2 at 16k, thinking off) with `allenai/tmax-2b` through vLLM gave 0.48 nats per assistant token (p90 1.75,
15% of tokens above 1 nat). Each 2B model solved 1 of the 16 episodes.

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
- **Thinking.** `allenai/tmax-4b`, `allenai/tmax-9b`, `hamishivi/Qwen3.5-4B`/`-9B` and stock Qwen3.5-4B and up
  all open `<think>` by default, as does Miles' `qwen35` template, and open-instruct trained with that default.
  The 2B checkpoints default to thinking off: `allenai/tmax-2b` and `hamishivi/Qwen3.5-2B` ship stock
  Qwen3.5-2B's template byte for byte, whose generation prompt closes an empty think block. Each `_RECIPES` entry
  follows its checkpoint's default. (The 4B OPD runs in the results below predate this and rendered with
  `enable_thinking=false`.) SGLang's Qwen3.5 reasoning
  detector treats `<tool_call>` as an implicit end of reasoning, so turns parse either way. vLLM's `qwen3`
  reasoning parser does not: for a model that emits no `</think>` it files the whole turn as reasoning and
  drops every tool call. Serve these checkpoints in vLLM with `--language-model-only` (they ship no
  image-processor config).

### Throughput

- A synchronous step waits for its slowest episode. Most episodes take 1-6 minutes; 40-turn long-thinking
  ones took about 20. Cap episodes with `episode_timeout_s` (default 0, off; the smoke configs use 3600) or use
  `--async-rollout-gpus`.
- Where the time goes depends on load. In 4-GPU smoke tests about 80% of episode wall time was sandbox
  execution; in the 4-node production run, with 512 episodes sharing 16 engines, model calls take 48% and sandbox
  commands 19%.
- The first training step JIT-compiles the TileLang and Triton kernels for the linear-attention layers
  (a few minutes); SGLang engines also compile at start-up and log health-check failures meanwhile.

### Session server

- `session_server: v2` (the default) is required by upstream's `--namespaced-radix-cache`, which fully async turns
  on with in-place weight updates so prefix KV computed under old weights never serves a later call. v2 also needs
  an image whose SGLang has prefill weight versions ([Images](/examples/tmax/beaker#images)). `v1` still works; pair it
  with `--no-namespaced-radix-cache` in that mode.
- v2 builds the same training samples as v1: one sample per episode, tool tokens loss-masked, the same observation
  framing. Episodes that end with a full context lose their last, overflowing turn, or keep it cut at
  `max_seq_len`; both versions do both.

### Fully async

- `--fully-async` rejects `--colocate` and `--rollout-function-path`, so the episode metrics are logged by a
  `--custom-rollout-log-function-path` hook (`tmax_metrics.log_rollout_data`), which both modes call.
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
  `tmax-4b` only adds seven unused audio/TTS special tokens (ids 248070-248076). The same holds for
  `allenai/tmax-2b` and `hamishivi/Qwen3.5-2B` (both tokenizers have 248,077 entries; tied embeddings). Like
  `tmax-4b`, `tmax-2b` has no vision tower or MTP weights; neither matters to Megatron-mode OPD.
- **Pure OPD** (`opd_pure`, open-instruct's `--opd_pure`, which all of its terminal OPD runs use) trains on the
  OPD term alone. `generate.opd_pure_rewards`, a `--custom-reward-post-process-path`, hands GRPO a reward of 0 for
  every sample, so its advantage is 0, and keeps the task reward as `raw_reward` (`rollout/raw_reward`).
  `rollout/advantages` is then the mean OPD term, and `tmax_metrics` leaves out `rollout/trained/advantage_*`.
  The launcher requires `filter_zero_std: false` with it, since every group carries signal.
- **Teacher on an SGLang server** (`opd_teacher_mode: sglang`; `auto`, the default, picks it when the teacher's
  architecture differs from the student's, as for `allenai/tmax-9b` into a 4B). Before submitting the job the launcher
  starts `sglang.launch_server` on the node's last GPU, which Ray is started without, and waits up to 30 minutes for
  it to answer, or the launch fails. `generate.reward_func_with_teacher` then sends every finished sample with a task
  reward to it (Miles' own teacher request, `--rm-url`) and stores the response tokens' log-probs in
  `sample.teacher_log_probs`, which Miles trains on as the Megatron teacher's. The task reward stays the reward, so
  `opd_pure`, additive OPD, the filters and the metrics work as in Megatron mode; Miles' documented SGLang-mode reward
  function and post-process replace it with 0. A teacher request is retried after `opd_teacher_timeout_s` (1800) or an
  error, and the third failure fails the rollout. One node and one teacher GPU only (SGLang TP > 1 produces garbage for
  Qwen3.5).
- SGLang serves the tmax checkpoints, which have no vision tower or image-processor config, with `--language-only`
  and the student's directory as `--tokenizer-path`: SGLang loads Qwen3.5's image processor from there, the processor
  configs are identical across Qwen3.5 sizes, and the vocabulary is shared (the tmax tokenizers add seven unused audio
  tokens; requests carry token ids). Its `--language-model-only` rejects Qwen3.5.
- On the 16 trajectories of the first step of `01M5K694GXJ85TX6RZ9A014QTH` (Megatron teacher), `allenai/tmax-4b` served
  by SGLang 0.5.21 on an L40S matched the Megatron teacher's log-probs to 0.0106 per trained token (median 0, p99
  0.13), and both give a reverse KL of 0.1788 per token. `allenai/tmax-9b` gives 0.162 on the same trajectories.
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
  sandboxes. Current Miles images are CUDA 13 only (upstream retired the CUDA 12 images). On 570-series drivers
  (Jupiter's and Ceres' H100s) they run through the image's CUDA forward-compatibility driver, which the entry script
  loads ([Images](/examples/tmax/beaker#images)).
- On 80 GB GPUs pass `--log-probs-backend fused`, as the prod configs do. The default backend upcasts the logits to
  fp32: 13.85 GiB for one ~15k-token episode, which ran an H100 out of memory in the training step.
- `openai_session_url` already appends `/v1`; agent functions pass its result to the OpenAI client as is.
- Qwen3.5 `torch_dist` checkpoints converted before the head-sharded linear-attention layout (upstream #3610)
  are rejected at load; re-convert them from HF into a new `--model-dir`.
- On B300 with TP=1 and no optimizer offload, Qwen3.5-9B training uses ~58 GB allocated per GPU, and a
  weight sync to the engines takes ~2.6 s.

### Known gaps

Open differences from open-instruct and the work on them are tracked in [ROADMAP.md](https://github.com/radixark/miles/blob/main/examples/tmax/ROADMAP.md); the
[comparison](/deep-dive/open-instruct-comparison) describes each. The largest:

- No DPPO in Miles: the port trains with PPO clip.
- The first-turn prompt is not token-identical: SGLang's tool schema adds `"strict": false` and reorders the tool
  definition's keys, about 17 tokens. Tool observations are framed differently too (two newlines, trimmed output,
  several calls in one turn share one user block).
- Podman comes from Ubuntu's apt (4.9) at job start; open-instruct's image builds 5.6.2 and crun 1.14.3.
- Every agent process runs on the head node (`--pin-rollout-manager-to-head`); open-instruct's environment actors
  run on every node. The sandboxes still spread: the head drives the other nodes' podman over TCP.
- Group-granular dropping: one infra failure discards its whole group.
