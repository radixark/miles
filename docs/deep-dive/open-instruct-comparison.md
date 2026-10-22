---
title: Miles vs open-instruct for Terminal RL
description: How Miles and open-instruct differ as RL frameworks, as Terminal-RL harnesses and as Beaker infrastructure, with a flag-by-flag mapping of the TMax recipe.
---
This page compares open-instruct's production Terminal-RL job with its Miles port, [TMax Terminal-RL](/examples/tmax).
It describes the differences. The work they imply is tracked in the port's
[roadmap](https://github.com/shatu/miles/blob/shashank/tmax-terminal-rl/examples/tmax/ROADMAP.md).

- **open-instruct:** branch `opd` at `1bf48f1a7`, script
  `scripts/general_agent/terminal/rl/qwen35_9b_dppo_repro_4node_64k.sh`, and its holmes/CUDA 13 variant
  (`--attn_implementation flash_4`, an 8 h minimum runtime with auto-resume, `--checkpoint_state_freq 5`).
  Code: `open_instruct/grpo_fast.py`, `grpo_utils.py`, `data_loader.py`, `vllm_utils.py`, `environments/`, `mason.py`,
  `scripts/docker/`.
- **Miles:** branch `shashank/tmax-terminal-rl` at `edf2c7ae3e` (rebased onto upstream `e30d9a8ec6`),
  `examples/tmax/` plus the Miles code it drives. Line numbers drift; symbol names are given where they help.

## At a glance

| | open-instruct | Miles |
|---|---|---|
| Driver | `grpo_fast.py`: a main loop plus a weight-sync thread, a `DataPreparationActor` that collects, filters and packs results, and an `ActorManager` that pauses generation | `train_async.py --fully-async` (or `train.py`): Ray actors for training and rollout, controllers that run them as fault-tolerant cells (`miles/ray/train/`, `miles/ray/rollout/`) |
| Trainer | DeepSpeed ZeRO-3 with Ulysses sequence parallelism (`PolicyTrainerRayProcess`) | Megatron-LM with TP / CP / SP (`MegatronTrainRayActor`) |
| Generator | vLLM engines (`LLMRayActor`), TP 1 | SGLang engines behind the SGLang router, 1 GPU each |
| Agent loop | Inside the generator actor (`vllm_utils.process_request`): it calls vLLM, parses tool calls itself and steps an environment actor from `EnvironmentPool` | A user agent function (`examples/tmax/tmax_agent_function.run`), one process per episode, that talks to the policy through the OpenAI chat API of a Miles session server |
| Token record | open-instruct tokenizes the conversation itself and appends generated tokens | The session server records token ids, logprobs and loss masks for every call (TITO); the training sample is assembled when the episode is collected |
| Weight sync | NCCL broadcast from learner rank 0 to every engine, driven by `weight_sync_thread` through Ray RPCs; 2 h timeout (`WEIGHT_SYNC_TIMEOUT_S`) | NCCL broadcast from trainer rank 0 in buckets, each announced to every engine by an HTTP `POST /update_weights_from_distributed`; no timeout on the HTTP wait |
| Async | `--async_steps N`, `--inflight_updates` (engines pause with `mode="keep"`) | `--fully-async`, `--max-weight-staleness`, `--async-data-buffer-capacity-factor`, `--pause-generation-mode in_place` |
| Policy loss | DPPO (binary TV divergence, threshold 0.1) | DPPO, `--policy-loss-type dppo_tv` (PPO clip 0.2 / 0.28 before branch `shashank/tmax-oi-parity`) |
| Launch | `mason.py`: the code is baked into the Beaker image | `examples/tmax/beaker/submit.py`: the job clones the repository at a pinned commit |

## One step, side by side

**open-instruct.** The `DataPreparationActor` keeps `async_steps` batches of prompts queued. Each vLLM actor runs
`process_request` per sample: generate a turn, parse tool calls with the `vllm_qwen3_xml` parser, step the
`swerl_vanillux_sandbox` environment held by a pool actor, append the observation tokens, repeat. Finished samples
come back with the engine's model step. The data actor drops results more than `async_steps` steps old, applies
active sampling (drop all-pass and all-fail groups, keep sampling until 8 groups survive), packs them to
`pack_length` and hands them to the learners. The learners run DPPO and step the optimizer once per 256
trajectories. The weight-sync thread then pauses generation, broadcasts the weights, wakes the engines and stamps
them with the new model step.

**Miles.** The rollout executor keeps up to `--async-max-concurrent-samples` agent calls running
(`miles/rollout/generate_hub/agentic_tool_call.generate`). Each call opens a session on the session server, runs the
agent function in its own process, and the agent drives the conversation with plain chat completions. SGLang's
`qwen3` reasoning and `qwen3_coder` tool-call parsers split each turn. When the agent returns, the generate function
collects the session: under session server v2 the owning worker converts the recorded calls into one training sample
(truncated at `--max-seq-len`, tool tokens loss-masked). Groups land in the fully-async data buffer, which drops
groups older than `--max-weight-staleness` and applies the zero-std filter. The trainer takes
`--rollout-batch-size` groups, runs PPO on the rollout log-probs, and pushes the new weights to the engines. With
`--pause-generation-mode in_place`, in-flight requests pause and continue on the new weights.

## Async, staleness and weight updates

[Async & off-policy §8](/deep-dive/async-and-off-policy) has the general `grpo_fast` comparison; this section is
what matters for the TMax recipe.

- **Staleness is counted from different origins.** open-instruct drops a result when
  `training_step - result.model_step > async_steps` (`data_loader.py`, `max_result_age_steps`); training step 1
  consumes results from model step 0, so fresh data has lag 1. Miles computes a group's staleness as the current
  weight version minus the oldest weight version among its tokens
  (`miles/rollout/fully_async_data_buffer.py`, `oldest_lag`), so fresh data is 0. The same number therefore admits
  one more version in Miles: `--max-weight-staleness 4` corresponds to `--async_steps 5`, and open-instruct's
  `--async_steps 4` to `--max-weight-staleness 3`. The prod config uses 4, one version looser.
- **Reading Miles' staleness metrics.** `rollout/fully_async/avg_staleness` is 0 for data generated entirely on
  current weights. On the 4-node run it climbs 0, 1, 2, 3, 4 over the first five steps, while the buffer filled
  before step 0 drains, then averages about 2 (max 3-4), with a few stale groups dropped (13 at step 5).
- **In-flight updates.** Both sides pause in-flight requests, swap the weights and continue them. Each request's
  own KV cache, computed under old weights, is kept on both sides (open-instruct: `pause_generation(mode="keep")`
  unless `--inflight_updates_recompute_kv_cache`; Miles: `in_place`). A trajectory can therefore span several weight
  versions; Miles records per-token versions (`weight_versions` on each sample, `rollout/weight_version/*`).
- **Prefix cache across trajectories.** open-instruct enables vLLM prefix caching but salts it per trajectory, so
  nothing is shared between trajectories. Miles shares SGLang's radix cache; with session server v2 and
  `--namespaced-radix-cache` (on by default with in-place updates in fully async), prefix KV from old weights never
  serves a later call.
- **Weight-sync transport.** open-instruct's sync is Ray RPCs plus NCCL, with a 2 h timeout and a health check every
  step. Miles' sync is NCCL plus one HTTP request per bucket per engine, from trainer rank 0. The HTTP client has no
  read timeout, so a slow sync just waits. A dropped connection (`httpx.ReadError`) fails the sync and the job; that
  happened once in the old 4-node run, during a 64 s sync after step 118 (normally 1.5-3 s). The cause is not known.
- **Timeouts.** open-instruct's `--backend_timeout 1200` (minutes, i.e. 20 h) covers its collectives. Miles defaults
  are 10 min (`--distributed-timeout-minutes`) and 300 s (`--sglang-watchdog-timeout`); both killed early 4-node runs
  during stalled cross-node collectives, so `prod_4node.yaml` sets 120 min and 1800 s.

## Inference engines

Both run 16 engines with TP 1, one GPU each, 8 per inference node (open-instruct: `--vllm_num_engines 16
--vllm_tensor_parallel_size 1`, a `PACK` placement group of one-GPU bundles). TP 2 in the Miles config is Megatron's
training layout; SGLang TP > 1 produces garbage for Qwen3.5 (sgl-project/sglang#21039).

Where the agent loop runs differs:

- **open-instruct:** each engine's Ray actor runs vLLM in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) and also the
  agent loop for its trajectories. It pulls whole trajectories from the prompt queue, up to what fits in its KV
  cache at full length, so agent work spreads over both inference nodes.
- **Miles:** the router pins each session to the engine it starts on (`--policy manual --assignment-mode min_load`).
  All agent processes, the router and the session servers run on the head node.

| open-instruct (vLLM) | Miles (SGLang, as resolved in production) | |
|---|---|---|
| `--lm_head_fp32 true`: vLLM patched to compute Qwen3.5 logits in fp32 | `enable_fp32_lm_head = False`; the trainer's logits are bf16 too | Missing; worth a third of the train/rollout gap ([measured](https://github.com/shatu/miles/blob/shashank/tmax-terminal-rl/examples/tmax/INVESTIGATIONS.md)) |
| Attention: vLLM's automatic choice on Blackwell | `attention_backend = triton` (SGLang's choice for Qwen3.5 on B300) | `trtllm_mha` is 1.75× faster at 32k context ([measured](https://github.com/shatu/miles/blob/shashank/tmax-terminal-rl/examples/tmax/INVESTIGATIONS.md)) |
| `gpu_memory_utilization 0.9` | `mem_fraction_static 0.8` | KV usage stays at a few percent; not a limit |
| `mamba_ssm_cache_dtype = "float32"` | SGLang defaults the linear-attention state to float32 | Same |
| `--vllm_gdn_prefill_backend triton` | `linear_attn_backend = triton` | Same |
| `generation_config="vllm"` (ignore the model's sampling defaults) | `sampling_defaults = model`, but the model ships no `generation_config.json` and Miles sends temperature 1.0, top_p 1.0, top_k -1 | Same in effect |
| Concurrent trajectories per engine = KV capacity at full length | `max_running_requests` automatic | No difference at current load |
| `seed + i` per engine, CUDA graphs, bf16 | `seed + engine offset`, CUDA graphs, bf16 | Same |

## Policy loss and numerics

| | open-instruct | Miles (`prod_4node.yaml`) |
|---|---|---|
| Loss | `--loss_fn dppo --dppo_divergence_type tv --dppo_divergence_threshold 0.1`. Per token: `r = π/μ` (unclamped), `D = |μ − π|` on the sampled token, mask `M = 0` when `D > 0.1` and the update moves further away (`A > 0` with `r > 1`, or `A < 0` with `r < 1`); loss `−A·r·M`; no ratio clip (`grpo_utils.py`) | PPO clip, `--eps-clip 0.2 --eps-clip-high 0.28`. The two differ most on low-probability tokens: at `μ = 0.05` with `A > 0`, PPO stops the update at `π = 0.064`; DPPO allows up to `π = 0.15` |
| Off-policy anchor | `--use_vllm_logprobs true` | `--use-rollout-logprobs` |
| TIS, KL | TIS cap 0 (off); `--beta 0.0` (the reference model is still loaded and four KL estimators are logged at weight 0) | TIS off; no KL term in the loss, but the KL to the base model is logged as `train/kl_loss` (`log_kl_to_base`, k3 = open-instruct's kl2) |
| Advantages | `centered`: reward minus the group mean | `grpo --disable-grpo-std-normalization` (same) |
| Aggregation | Token denominator, but each (rank, micro-batch) shard is weighted separately, so tokens in sparse shards weigh more | Exact global token mean, `--calculate-per-token-loss` |
| Packing | `--pack_length 67584`, one pack per micro-batch; packs not divisible by DP are dropped | Dynamic batching, `--max-tokens-per-gpu 34816` (≥ max_seq_len / CP), balanced, nothing dropped |
| Logits | `--lm_head_fp32 true` in trainer and vLLM | bf16 output layer; SGLang's `--sglang-enable-fp32-lm-head` exists but is not passed. bf16 rounding is most of the train/rollout gap, and only fixing both sides helps (see [Inference engines](#inference-engines)) |
| Log-prob memory | `--use_liger_grpo_loss --liger_grpo_loss_chunk_size 8` (open-instruct's tiled LM-head loss) | `--log-probs-backend fused` |
| Optimizer | AdamW, lr 1e-6, betas (0.9, 0.999), eps 1e-8, wd 0, constant LR, grad clip 1.0 | Adam, lr 1e-6, betas (0.9, 0.999), eps 1e-8, wd 0, constant LR, `--clip-grad 1.0` (beta2 was 0.98 before branch `shashank/tmax-oi-parity`) |
| Seed | 42 | `--seed 42` (Miles' default 1234 before branch `shashank/tmax-oi-parity`; rollout seed 42) |
| Parallelism | ZeRO-3, Ulysses SP 4, 16 learners | TP 2 × CP 2 (+ SP), DP 4 on 16 GPUs |
| Recompute | Full gradient checkpointing | `--recompute-granularity full --recompute-method uniform --recompute-num-layers 1` |
| Attention | Auto FA2/FA3; holmes variant `flash_4` | `--attention-backend flash` |

Both train truncated and non-submitting episodes with their reward (0) and mask tool-output tokens out of the loss.

## Harness: the agent loop

The episode logic was ported line by line from `vllm_utils.process_request` and `swerl_vanillux_sandbox`. 4,390 of
4,391 real open-instruct turns parse identically with SGLang's `qwen3` + `qwen3_coder` parsers; the exception is
malformed and ends the episode on both sides. What differs is tokenization, timeout handling and where code runs.

| | open-instruct | Miles |
|---|---|---|
| System prompt | `swerl_vanillux_sandbox_system_prompt.txt` | `tmax_prompts.TMAX_SYSTEM_PROMPT`, byte-identical |
| Tool definition in the prompt | Rendered as written: `{"type", "function": {"name", "description", "parameters"}}` | Goes through SGLang's tool schema model first, which adds `"strict": false` and orders `description` before `name`: about 17 extra tokens in every system turn |
| Tool-response framing | `<|im_end|><|im_start|>user\n<tool_response>\n{raw output}\n</tool_response>\n<|im_end|>\n<|im_start|>assistant\n<think>\n`; several calls in one turn each get their own user block | The chat template's framing: two newlines differ, output is whitespace-trimmed, and several calls in one turn share one user block. Loss-masked either way, but the policy conditions on it |
| Calls | Unknown tools ignored; a turn with no valid call ends the episode; every call runs in order and counts toward `max_steps` (64); format-error feedback, last-step warning and turns-remaining off | Same |
| Token budget | Per turn: the least of the remaining response budget (tool tokens included), the remaining context and 16,384; tool output truncated to fit; prompts over 2,048 tokens filtered out (2 of 14,601) | The least of 16,384 and what was left after the previous call; the newest observation is not counted, so the last turn can overrun and the episode ends `context_full` (about 1-1.5% of episodes) |
| Turn cut by the length limit | Still parsed; a complete tool call continues the episode | Episode ends (`length`, about 0.1%) |
| Step timeout | 120 s around the whole step, a submit's test run included (`tool_call_timeout`), so tests slower than 120 s never score | Same (`tool_call_timeout_s: 120`) |
| After a timeout | The command keeps running; the next one queues behind it | The command keeps running; the next one runs alongside it |
| Error text | `Step 'bash' timed out after 120.0s. Args: {…}`; failures carry Ray's error text | `…after 120s`; plain exception text |
| Tests | Uploaded at submit; `test_timeout` raised to a 600 s floor; reward from `/logs/verifier/reward.txt`, clamped, else 0 | Same |
| Shell | Persistent-shell wrapper (cwd and exports kept), 120 s per command, 1 MB cap, 5k + 5k head/tail truncation with a hint, `(exit_code=N)` | Same text |
| `test.sh` missing | Checked inside the container (two tries); a failure is an observation and the episode continues | Checked on the host; the episode ends as submitted with reward 0 |
| Sandbox start failure | 5 attempts with backoff; the failing daemon shard is benched on every node; the episode scores 0 through a one-token sample | 5 attempts with backoff; only that endpoint is benched; scores 0 through a one-token turn (`start_failure: zero`) |
| Container vanished | Recreated without the shell wrapper, so later commands fail | Recreated once with the wrapper re-installed; the episode continues without its state |
| OOM | 409 + `OOMKilled` ends with reward 0 | Same (whether memory limits apply at all with `cgroups="disabled"` is unverified on both sides) |
| Where agent code runs | 512 environment actors spread over the nodes by Ray, each driving its own node's podman | All agent processes on the head node (`--pin-rollout-manager-to-head`); other nodes' podman reached over TCP |
| Timing logs | `SWERL_SANDBOX_TIMING_LOGS=1`: start, pull, create, exec and upload phases over 1 s | Per-episode totals only |

## Sandboxes, images and Beaker

| | open-instruct | Miles |
|---|---|---|
| Where sandboxes run | Podman daemons in the training job; branch `literegistry_podman` adds `backend: gateway`, a LiteRegistry podman fleet behind a co-located gateway | Same two choices: `sandbox_backend: podman` (default) or `gateway` (`tmax_gateway.py`, a port of open-instruct's `GatewayBackend`) |
| Podman | 5.6.2 and crun 1.14.3 built into the image | Ubuntu apt podman 4.9, installed at job start |
| Bring-up | `scripts/docker/docker_login.sh`: N daemons per node with their own storage, mirror config, Docker Hub login, janitors | Same script, vendored byte-identical in `examples/tmax/beaker/podman/` |
| Container options | 4 GB memory and swap, host network, auto user namespace, cgroups disabled | Same `containers.conf` |
| Container lifetime | Kept after the episode until its pool actor's next reset | Removed when the episode ends |
| Per-node load | 512 pool actors over 4 nodes (about 128 each) | Hard cap `sandboxes_per_node: 128`; start and exec limits per node (64 / 256) as open-instruct's file semaphores |
| Image cache | Janitor prunes unused images older than 10 min every 60 s per daemon | Same |
| Leaked containers | Janitor reaps exited sandboxes by label (off in prod, auto-remove) | None; a failed removal leaks the container until the job ends |
| Mirrors | `MIRROR_URL` copied into each script by hand | `mirror_url: auto` resolves the running `registry_mirror.py` mirrors at submit time |
| Pull credentials | docker-py sends `~/.docker/config.json` with each pull | Sent explicitly with each pull (`TMAX_REGISTRY_*`); podman 4.9 ignores its own login when the request carries none |
| Code in the job | Baked into the image | `git clone --shared` of the weka checkout at the pinned commit |
| Secrets | Looked up by Beaker identity, only if they exist, including `BEAKER_TOKEN` | Three fixed names from the local user; no existence check; no `BEAKER_TOKEN` |
| Multi-node | Replicas with leader selection, host networking, failure and preemption propagation, NCCL over InfiniBand; a worker exits 1 after 6 failed `ray status` checks of the head (`ray_node_setup.sh`) | Same, plus `synchronizedStartTimeout` (1 h) and workers that exit 0 after a successful run. No head check: workers rely on Beaker's failure propagation, since open-instruct's check ended a run whose starved head was only slow ([INVESTIGATIONS.md](../../examples/tmax/INVESTIGATIONS.md) option M) |
| Result dataset | `/output` | None |

## Checkpoints, resume and logging

| | open-instruct | Miles |
|---|---|---|
| Training state | DeepSpeed model, optimizer, scheduler, per-rank RNG, counters, dataloader and data-prep state every `--checkpoint_state_freq` steps (holmes: 5) | Megatron `torch_dist` model, optimizer, scheduler and RNG, plus the dataset position, every `save_interval` steps (5); about 117 GB each for the 9B |
| Retention | Last 3; incomplete checkpoints removed at start | Newest 3 kept by the entry script's pruner (`keep_last_n_checkpoints`) |
| HF snapshots | Every 20 steps (`--save_freq`), never pruned | `hf_snapshot_every: 20`: `--save-hf` at every save (every 5 steps), the snapshots after every 20 steps kept, the rest pruned |
| Which jobs resume | Retries of one submission | Retries of one submission: `submit.py` names each submission `<exp_name>_YYYYMMDD_HHMMSS`; `--exp-name` continues an existing run |
| In-flight rollouts at a restart | Lost; their prompts are skipped | Lost; their prompts are skipped |
| Trained batches on disk | `--save_traces` JSONL per trained sample with advantages; `--save_trainer_logprobs` | `dump_training_data`: `rollout_data/<step>.pt` and `train_data/<step>_<rank>.pt` (about 200 MB per step in prod) |
| Every episode as text | No (filtered rollouts only behind a flag) | `trace_dir`: every episode's messages, reward and metrics, filtered groups included |
| W&B | Entity `ai2-llm`, one run per submission, tags (commit, branch, PR), Beaker URL in the config | One run per submission; entity `ai2-llm` (`wandb_entity`); no tags or Beaker link |
| Beaker description | Updated after every step with the W&B URL and progress/ETA (needs `BEAKER_TOKEN`) | Fixed at submit; includes the W&B run's URL |

## Metrics

| Signal | open-instruct | Miles |
|---|---|---|
| End reasons, tool calls, tool errors, start failures, model vs env time | `tools/bash/*`, `tools/env_reset/failure_rate`, truncated / non-submitting fractions | `rollout/agent/*`, `rollout/tools/{bash,env_reset}/*` from `examples/tmax/tmax_metrics` (a `--custom-rollout-log-function-path` hook, so in both modes); per episode in the trace files |
| Reward | `scores`, `val/avg_group_performance_pre_filter` | `rollout/raw_reward`, `rollout/raw_reward_unfiltered` |
| Solve-rate distribution | `val/solve_rate_hist`, `batch/percent_solved_hist` | `rollout/unfiltered/frac_solve_rate_*` (per-group solve rate before the filter, in bins) |
| Zero-variance drops | `batch/filtered_prompts*` | `rollout/dynamic_filter/drop_zero_std_*`; `rollout/zero_std/*` is computed after the filter and always 0 |
| Train/rollout mismatch | `debug/vllm_vs_local_logprob_diff_{mean,max,std}` | `train/train_rollout_logprob_abs_diff`, `train/train_rollout_kl` (mean only) |
| Staleness | `model_step_*`, `stale_results_dropped` | `rollout/fully_async/*` (average, max, token-weighted, version span), `rollout/weight_version/*` |
| Throughput | `val/actor_tokens_per_second` | `perf/*`; in fully async `perf/rollout_time` measures the buffer drain, so throughput values are meaningless |
| Weight sync | `time/weight_sync*` | `perf/update_weights_time`, `perf/update_weights_implementation_time` |

[Metrics & debugging](/deep-dive/metrics-and-debugging) has every Miles formula.

## Flag-by-flag: the production recipe

open-instruct's `qwen35_9b_dppo_repro_4node_64k.sh` against `examples/tmax/configs/prod_4node.yaml` and the
flags `run_tmax_qwen3_5.py` adds.

| open-instruct | Miles |
|---|---|
| `--dataset_mixer_list allenai/tmax-15k-open-instruct 1.0`, split `train` | `prepare` writes `miles_prompts.jsonl` from the same dataset (`--prompt-data`, `--rollout-shuffle`) |
| `--max_prompt_token_length 2048` | No prompt filter; `max_seq_len` counts the prompt |
| `--response_length 65536`, `--pack_length 67584` | `max_seq_len: 67584` (`--max-seq-len`) |
| `--per_turn_max_tokens 16384` | `rollout_max_response_len: 16384` (per turn) |
| `--per_device_train_batch_size 1` | `--use-dynamic-batch-size`, `max_tokens_per_gpu: 34816` |
| `--num_unique_prompts_rollout 8`, `--num_samples_per_prompt_rollout 32` | `rollout_batch_size: 8`, `n_samples_per_prompt: 32` (global batch 256) |
| `--total_episodes 128000`, `--num_epochs 1` | `num_rollout: 500` |
| `--async_steps 4` | `max_weight_staleness: 4` (one version looser, see above), `async_data_buffer_capacity_factor: 4` |
| `--inflight_updates true` | `--pause-generation-mode in_place` |
| `--pool_size 512` | `async_max_concurrent_samples: 512`, `sandboxes_per_node: 128` |
| `--num_learners_per_node 8 8`, `--vllm_num_engines 16`, `--vllm_tensor_parallel_size 1` | `num_nodes: 4`, `async_rollout_gpus: 16` (2 training + 2 rollout nodes), `--rollout-num-gpus-per-engine 1` |
| `--deepspeed_stage 3`, `--sequence_parallel_size 4` | `tensor_model_parallel_size: 2`, `context_parallel_size: 2`, `--sequence-parallel` |
| `--model_name_or_path hamishivi/Qwen3.5-9B` | `model_name: Qwen3.5-9B-tmax-template` |
| `--temperature 1.0` | `--rollout-temperature 1.0` |
| `--learning_rate 1e-6`, `--lr_scheduler_type constant` | `--lr 1e-6 --lr-decay-style constant` |
| `--beta 0.0` | No KL flag |
| `--use_vllm_logprobs true` | `--use-rollout-logprobs` |
| `--truncated_importance_sampling_ratio_cap 0.0` | TIS off (default) |
| `--advantage_normalization_type centered` | `--advantage-estimator grpo --disable-grpo-std-normalization` |
| `--loss_fn dppo --dppo_divergence_type tv --dppo_divergence_threshold 0.1` | `--policy-loss-type dppo_tv --dppo-tv-threshold 0.1` |
| `--lm_head_fp32 true` | Not set |
| `--use_liger_grpo_loss --liger_grpo_loss_chunk_size 8` | `--log-probs-backend fused` |
| `--gradient_checkpointing` | `--recompute-granularity full --recompute-method uniform --recompute-num-layers 1` |
| `--seed 42` | `--seed 42` |
| `--active_sampling` | `filter_zero_std: true` (`apply_reward_nonzero_std_filter`) |
| `--vllm_enable_prefix_caching` | SGLang radix cache (default), namespaced under v2 |
| `--vllm_gdn_prefill_backend triton` | n/a (SGLang's own linear-attention kernels) |
| `--backend_timeout 1200` | `--distributed-timeout-minutes 120 --sglang-watchdog-timeout 1800` |
| `--tools swerl_vanillux_sandbox`, `--tool_configs {test_timeout: 120, ...}` | `tmax_agent_function` + `tmax_sandbox` (per-task image from metadata, 600 s test floor) |
| `--tool_parser_type vllm_qwen3_xml` | `--sglang-tool-call-parser qwen3_coder --sglang-reasoning-parser qwen3` |
| `--system_prompt_override_file …swerl_vanillux_sandbox_system_prompt.txt` | `tmax_prompts.TMAX_SYSTEM_PROMPT` |
| `--max_steps 64`, `--verification_reward 1.0` | `max_steps: 64`; reward is `reward.txt` clamped to [0, 1] |
| `--checkpoint_state_freq 10` (holmes: 5) | `save_interval: 5` |
| `--save_freq 20` | No HF export |
| `--save_traces`, `--save_trainer_logprobs true`, `--rollouts_save_path` | `trace_dir`, `dump_training_data: true` |
| `--with_tracking --wandb_project oe-general-agents` | `wandb_project: oe-general-agents` |
| `--local_eval_every 10` | None (the prod run has no eval set either) |
| `--output_dir /output`, `--exp_name` | `output_dir`, `exp_name` (+ the submission's date-time suffix) |
| holmes `--attn_implementation flash_4` | `--attention-backend flash` |
| mason `--num_nodes 4 --max_retries 5 --gpus 8 --priority urgent`, holmes `--min_runtime 8h --auto_resume` | `beaker: nodes: 4, retries: 10, gpus: 8, priority: urgent, min_runtime: 8h` (Beaker rejects `preemptible` together with a minimum runtime). Retries raised from 5 after the 2026-10-16 run used all 5 |
| `SWERL_PODMAN_SERVICE_COUNT=4` | `beaker.sandbox_shards: 4` |
| `SWERL_DOCKER_START_CONCURRENCY=64`, `SWERL_DOCKER_EXEC_CONCURRENCY=256` | `sandbox_start_concurrency: 64`, `sandbox_exec_concurrency: 256` |
| `SWERL_RESET_FAILURE_ZERO_REWARD=1` | `start_failure: zero` |
| `SWERL_DOCKER_AUTO_REMOVE=1` | Containers are always removed at episode end |
| `SWERL_PODMAN_IMAGE_JANITOR_*` (60 s, 10 m) | Same defaults in `tmax_beaker_entry.sh` |
| `MIRROR_URL=…`, `PODMAN_NUM_LOCKS=65536` | `beaker.mirror_url: auto`; same lock count |
| `--secret DOCKER_PAT=…` | `beaker.secrets` (default `<user>_DOCKER_PAT`, `_HF_TOKEN`, `_WANDB_API_KEY`) |
| `SWERL_SANDBOX_TIMING_LOGS=1` | Not available |
