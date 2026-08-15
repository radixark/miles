# Polar + SWE-Gym GRPO (Miles-native launch example)

Train **NVIDIA Nemotron-3-Nano-30B-A3B** (BF16, hybrid Mamba + GQA + MoE) with
**asynchronous GRPO** on **SWE-Gym** tasks, driving agent rollouts through
**Polar** while **Miles** (Megatron backend) trains. This is the Miles-native
port of ProRL-Agent-Server's `examples/swegym_slime_grpo/` — the Slime flags are
swapped for Miles' own `train.py` CLI, and the rollout/reward/config hooks point
at the Miles `polar_*` bridge modules under `miles/rollout/`.

Every agent turn in a Polar session becomes one training sample (dynamic
history), so gradients are computed over **every** trace, not just the last one.

## How the pieces fit together

```
                 +-------------------------- Polar rollout server (pre-existing) ----------+
  train.py       |   task submit / poll           per-task Docker sandbox (docker backend)  |
  (Ray job) ---> miles.rollout.polar_rollout:generate_rollout_polar_async -------------->  |
     |           miles.rollout.polar_reward:custom_rm       <- async reward, rm_hub awaits  |
     |           miles.rollout.polar_config:resolve_polar_slime_config                     |
     |           miles.rollout.polar_data_source:CeilEpochRolloutDataSourceWithBuffer      |
     v                                                                                      |
  Megatron GRPO trainer (4 GPUs, ref model colocated) --weight sync-> SGLang engines (4 GPUs)|
     +---- refinery: 2 SGLang engines x TP=2 (router :9000)  <----- agent LLM calls ---------+
```

Miles holds the SGLang engines in-process; the Polar gateway proxies the
agent's LLM calls to them through the Miles SGLang router. Fresh weights are
synced GPU-to-GPU into the engines every step.

## Prerequisites

- **Polar server running.** The Polar rollout server (task submission/polling)
  and gateway (agent session dispatching) must already be up — e.g. started with
  `polar serve_rollout -c <topology>` + `polar serve_gateway -c <topology>`.
  `run.sh` does **not** start them; it only renders the Polar custom config,
  boots Ray, and submits the Miles training job. The gateway must be pointed at
  the Miles SGLang router (`http://<host>:9000`).
- **Xyne backend (optional).** If you want Node-based agent harnesses as an
  alternative to the in-image Docker `codex` harness, prepare the shared Node/agent
  CLI dir (`AGENT_CLI_DIR`) as the reference does; otherwise the Docker task image
  already ships everything the `codex` harness needs.
- **SWE-Gym Docker-JSONL data.** Each row must have `prompt`, `label`,
  `metadata`, and `metadata.registry_image` (the SWE-Gym Docker image reference)
  plus `metadata.instance` for the swebench harness. See `PROMPT_DATA`.
- **Converted Megatron `torch_dist` checkpoint** for the reference model
  (`REF_LOAD`). The actor initializes directly from `--hf-checkpoint` via Miles
  AutoBridge; the KL reference model (`--use-kl-loss`) is a **frozen copy
  colocated on the actor ranks** and is loaded from this Megatron checkpoint.
- **8 x {H100, H200, B200}** (or any >=48 GB HBM GPUs) on one node.
- Miles environment with the **Megatron backend** + `sglang` + `sglang-router`
  installed, and the HF checkpoint at
  `/data/siraj/hf_models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.

## GPU split (fixed): 4 train / 4 inference

`run.sh` locks an 8-GPU layout:

| GPUs | Role | Shape |
|------|------|-------|
| 0–3  | Megatron actor/trainer | TP=2, PP=1, MoE at **EP1/ETP1**; **reference model colocated on the actor ranks** |
| 4–7  | SGLang inference | **2 engines × TP=2** (`--rollout-num-gpus 4` / `--rollout-num-gpus-per-engine 2`) served via LoRA rank 16 |

Rationale:

- **4 trainer GPUs** keep Megatron's activation/optimizer state well within HBM
  for the 30B-A3B checkpoint at the LoRA rank 16 target, while the small active
  params (A3B) and TP=2 leave headroom for a reasonable `--max-tokens-per-gpu`.
- **Ref model is colocated on the actor ranks** — Miles has **no
  `--ref-num-nodes 0` flag** (the Slime flag is dropped, see Flag mapping). In
  Miles the reference model always shares the actor GPUs; it is enabled by
  `--use-kl-loss` (`miles/ray/placement_group.py`: `with_ref = kl_coef != 0 or
  use_kl_loss`) and loaded from `--ref-load`. This is the native "colocated ref"
  convention and is what keeps the 8-GPU split feasible.
- **2 SGLang engines × TP=2** gives the LoRA-served MoE model enough per-engine
  tensor parallelism for expert latency while keeping two engines for router
  load-balancing and overlap during agent-heavy rollout.

## Launch

```bash
export WANDB_API_KEY=<your-key>            # only if you want tracking
# 1. Start Polar (separately) if not already running
#    polar serve_rollout -c <topology> & polar serve_gateway -c <topology> --node-id <id> &
# 2. Launch Miles
bash examples/polar_swegym_grpo/run.sh
```

Override any default via env vars: `HF_CHECKPOINT`, `REF_LOAD`, `PROMPT_DATA`,
`AGENT_CLI_DIR`, `NUM_ROLLOUT`, `ROLLOUT_BATCH_SIZE`, `N_SAMPLES_PER_PROMPT`,
`LORA_RANK`, `KL_LOSS_COEF`, `LR`, and the `SGLANG_ROUTER_*` vars.

## Expected smoke criteria

A healthy first rollout should show:

- **Reward variance nonzero.** Because Polar returns real swebench-harness
  scores, the per-group reward array must not be all-equal — `post_process_rewards`
  (leave-one-trajectory-out normalization) only produces nonzero advantages when
  rewards differ within a group. All-zero/constant advantages ⇒ check that
  `metadata.registry_image` actually evaluates (image pull / harness errors).
- **Finite loss.** The `train/` loss (and the `low_var_kl` KL term) must be
  finite after the first training step. NaN/Inf ⇒ divergence (lower `LR`) or a
  bad LoRA/serving mismatch (check `--sglang-lora-backend triton` alignment with
  the fused-MoE LoRA path).
- **Polar sessions complete.** The rollout worker reaches `--num-rollout` steps
  without dropping groups; `drain_completed` logs show completed task groups.
- **`--ref-load` warning gone.** Confirm the ref torch_dist checkpoint exists so
  the KL term has a valid frozen reference.

## The async `custom_rm` contract

`miles.rollout.polar_reward:custom_rm` is defined **`async`**:

```python
async def custom_rm(args, sample) -> float:
    return compute_reward(args, sample)   # sync float extraction of the Polar score
```

Miles' reward runtime (`miles/rollout/rm_hub/__init__.py` — `async_rm` /
`batched_async_rm`) does `await rm_function(args, sample, **kwargs)` on the
callable registered via `--custom-rm-path`, so the hook **must be a coroutine
function** for the async trainer to `await` it. The synchronous `compute_reward`
holds the actual Polar score extraction and is reused by `custom_rm`, by the
Slime-compatible `reward_func`, and by `post_process_rewards`.

## Files

| File | Purpose |
|------|---------|
| `run.sh` | Renders the Polar config, starts Ray, submits `train.py` with all wiring |
| `model_args.sh` | Nemotron-3-Nano-30B-A3B Megatron MODEL_ARGS (mirrors `scripts/models/nemotron-3-nano-30b-a3b.sh`) |
| `polar_config_docker.yaml` | `--custom-config-path` input: Polar rollout/reward/task-template settings for the Docker task backend |
| `README.md` | This file |

## Flag mapping (Slime → Miles)

Every flag in `run.sh` / `model_args.sh` is a real Miles `train.py` CLI flag.
`miles/utils/arguments.py` defines the Miles-side args; model-structure / MoE /
parallelism flags are delegated to the Megatron backend
(`miles/backends/megatron_utils/arguments.py` → `megatron.training.arguments`);
`--sglang-*` serve args are SGLang `ServerArgs` fields registered with a
`--sglang-` prefix (`miles/backends/sglang_utils/arguments.py`).

| Slime (reference) | Miles (`train.py`) | Definition site in Miles |
|---|---|---|
| `--rollout-function-path slime_bridge.rollout.generate_rollout_polar_async` | `--rollout-function-path miles.rollout.polar_rollout:generate_rollout_polar_async` | `miles/utils/arguments.py` (`add_rollout` section) |
| `--custom-rm-path slime_bridge.reward.reward_func` | `--custom-rm-path miles.rollout.polar_reward:custom_rm` | `miles/utils/arguments.py` (`add_reward_model_arguments`) |
| `--custom-reward-post-process-path slime_bridge.reward_post_process.post_process_rewards` | `--custom-reward-post-process-path miles.rollout.polar_reward:post_process_rewards` | `miles/utils/arguments.py` (`add_reward_model_arguments`) |
| `--custom-config-path <yaml>` | `--custom-config-path <yaml>` (same) | `miles/utils/arguments.py` (reset to load YAML into `args`) |
| `--data-source-path slime_bridge.data_source.CeilEpochRolloutDataSourceWithBuffer` | `--data-source-path miles.rollout.polar_data_source:CeilEpochRolloutDataSourceWithBuffer` | `miles/utils/arguments.py` (`add_data_arguments`) |
| `--ref-num-nodes 0` | *(dropped)* — Miles colocates the ref on actor ranks; enabled by `--use-kl-loss`, loaded from `--ref-load` | `miles/ray/placement_group.py` `with_ref=` |
| `--sglang-lora-backend triton` | `--sglang-lora-backend triton` (same) | SGLang `ServerArgs.lora_backend` via `--sglang-` prefix |
| `--lora-rank 16` | `--lora-rank 16` (same) | `miles/utils/arguments.py` (`add_lora_arguments`, default 0) |
| `--kl-loss-type low_var_kl` | `--kl-loss-type low_var_kl` (same) | `miles/utils/arguments.py` |
| `--num-steps-per-rollout N` | `--num-steps-per-rollout 1` (Miles arg, kept) | `miles/utils/arguments.py` |
| Polar config YAML keys (`polar_*`) | consumed by `miles.rollout.polar_config:resolve_polar_slime_config` — **unchanged** | `miles/rollout/polar_config.py` |
