# Miles Server Arguments

This document provides a detailed list of command-line arguments used to configure Miles for RL training and inference. These arguments enable precise control over cluster resources, training backends (Megatron/FSDP), inference optimization via SGLang, and RL algorithmic hyperparameters.

You can find all arguments by running:
```bash
python3 train.py --help
```

## Hardware Support

Miles supports several GPU hardware platforms:
- **NVIDIA H-Series (H100/H200)**
- **NVIDIA B-Series (B200)**
- **AMD Support**: Available via ROCm. Refer to the [AMD Usage Tutorial](../platform_support/amd_tutorial.md).

## Table of Contents
1. [Cluster and Resource Management](#cluster-and-resource-management)
2. [Training Backend](#training-backend)
3. [Rollout Management](#rollout-management)
4. [Sampling and Filtering](#sampling-and-filtering)
5. [Data Arguments](#data-arguments)
6. [Evaluation Arguments](#evaluation-arguments)
7. [Algorithm and RL Arguments](#algorithm-and-rl-arguments)
8. [Logging and Monitoring](#logging-and-monitoring)
9. [Fault Tolerance](#fault-tolerance)
10. [Miles Router](#miles-router)
11. [Reward Model Arguments](#reward-model-arguments)
12. [Rollout Buffer Management](#rollout-buffer-management)
13. [Multi-Token Prediction (MTP) Arguments](#multi-token-prediction-mtp-arguments)
14. [SGLang Backend Arguments](#sglang-backend-arguments)
15. [FSDP Specific Arguments](#fsdp-specific-arguments)
16. [Debug and Profiling](#debug-and-profiling)
17. [Environment Variables](#environment-variables)

---

## Cluster and Resource Management

Arguments for configuring Ray cluster resources and GPU allocation.

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--actor-num-nodes` | Number of nodes for training the Actor. | `1` | Type: int |
| `--actor-num-gpus-per-node` | Number of GPUs per node for training the Actor. | `8` | Type: int |
| `--critic-num-nodes` | Number of nodes for the Critic. Defaults to `--actor-num-nodes`. | `None` | Type: int |
| `--critic-num-gpus-per-node` | Number of GPUs per node for the Critic. Defaults to `--actor-num-gpus-per-node`. | `None` | Type: int |
| `--rollout-num-gpus` | Total number of GPUs required for rollout (inference). In `--colocate` mode, this is ignored and set to match Actor resources. | `None` | Type: int |
| `--rollout-num-gpus-per-engine` | GPUs per inference engine (SGLang `tp_size`). For multi-node serving, this should be the total GPU count for the engine. | `1` | Type: int |
| `--num-gpus-per-node` | Total GPUs per node on the machine. Specify this if using fewer than 8 GPUs per node in colocate mode. | `8` | Type: int |
| `--colocate` | Deploy training and rollout on the same GPUs. Enables `--offload-train` and `--offload-rollout`. | `False` | bool flag (set to enable) |
| `--offload` | Equivalent to setting both `--offload-train` and `--offload-rollout` to true. | `False` | bool flag (set to enable) |
| `--offload-train` | Offload the Actor to CPU during the rollout phase. Always enabled when `--colocate` is set. | `None` | bool flag (set to enable) |
| `--offload-rollout` | Offload the rollout generator to CPU during the training phase. Always enabled when `--colocate` is set. | `None` | bool flag (set to enable) |
| `--distributed-backend` | Backend for distributed communication. | `nccl` | `nccl`, `gloo` |
| `--distributed-timeout-minutes` | Timeout for distributed operations in minutes. | `10` | Type: int |
| `--prefill-num-servers` | Number of dedicated prefill servers for PD disaggregation. | `None` | Type: int |

---

## Training Backend

Arguments for configuring the training engine (Megatron or FSDP).

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--train-backend` | The backend for training. | `megatron` | `megatron`, `fsdp` |
| `--qkv-format` | The QKV layout for the Megatron backend. | `thd` | `thd`, `bshd` |
| `--true-on-policy-mode` | Enable true-on-policy mode. | `False` | bool flag (set to enable) |
| `--train-env-vars` | Extra environment variables for the training process (e.g., PyTorch memory management). | `{}` | Type: JSON / Dict |
| `--train-memory-margin-bytes` | Reserved memory margin for training in bytes. Defaults to 1GB. | `1073741824` | Type: int |
| `--disable-weights-backuper` | Disable weights backup to host memory to save host memory. | `True` (enabled) | bool flag (set to enable) |
| `--megatron-to-hf-mode` | Method to convert Megatron weights to HuggingFace format for SGLang integration. | `raw` | `raw`, `bridge` |
| `--custom-model-provider-path` | Path to a custom model provider function (e.g., `my_module.my_provider`). | `None` | Type: str |
| `--recompute-loss-function` | Enable recomputing the loss function to save VRAM during training. | `False` | bool flag (set to enable) |
| `--log-probs-chunk-size` | Chunk size for computing log probabilities to save memory. `-1` means no chunking. | `-1` | Type: int |

---

## Rollout Management

Arguments for configuring the rollout (inference) process and custom rollout logic.

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--hf-checkpoint` | Path to the HuggingFace checkpoint. Used to initialize SGLang and provide the tokenizer. Parameters are synced from Megatron before training. | `None` | Type: str |
| `--model-name` | Name of the model, used for converting Megatron weights into HuggingFace format. | `None` | Type: str |
| `--rollout-function-path` | Path to the rollout generation function. Use this to inject custom logic (e.g., for multi-turn or tool use). | (internal path) | Type: str |
| `--rollout-temperature` | Sampling temperature for the inference engine during rollout. | `1.0` | Type: float |
| `--rollout-top-p` | Top-p (nucleus) sampling threshold during rollout. | `1.0` | Type: float |
| `--rollout-top-k` | Top-k sampling threshold during rollout. `-1` means disabled. | `-1` | Type: int |
| `--rollout-max-context-len` | Maximum context length for the inference engine. Must not exceed the model's `max_position_embeddings`. | `None` | Type: int |
| `--rollout-max-prompt-len` | Maximum length of the prompt. Longer prompts are filtered during dataset initialization. | `None` | Type: int |
| `--rollout-max-response-len` | Maximum length of the response (`max_tokens` in SGLang). | `None` | Type: int |
| `--rollout-skip-special-tokens` | Skip special tokens in the response. Useful when the response is used as a prompt for the next rollout. | `False` | bool flag (set to enable) |
| `--rollout-stop` | Stop words for the inference engine. Can be a single string or a list of strings. | `None` | Type: List[str] |
| `--rollout-stop-token-ids` | Stop token IDs for the inference engine. | `None` | Type: List[int] |
| `--rollout-shuffle` | Shuffle the prompts during rollout. | `False` | bool flag (set to enable) |
| `--rollout-seed` | Seed for the random number generator during rollout (used for shuffling and sampling). | `42` | Type: int |
| `--rollout-external` | Use external SGLang instances instead of launching them inside the framework. | `False` | bool flag (set to enable) |
| `--rollout-external-engine-addrs` | Addresses and ports of the external engines. | `None` | Type: List[str] |
| `--custom-generate-function-path` | Path to override only the `generate` step within the default rollout function. | `None` | Type: str |
| `--custom-rollout-log-function-path` | Path to a custom function for logging training rollout data. | `None` | Type: str |
| `--custom-eval-rollout-log-function-path` | Path to a custom function for logging evaluation rollout data. | `None` | Type: str |
| `--rollout-data-postprocess-path` | Path to a function called after all rollout data (including log probs) is ready. | `None` | Type: str |
| `--keep-old-actor` | Keep the Actor model loaded during the training process. | `False` | bool flag (set to enable) |
| `--update-weight-buffer-size` | Buffer size for updating weights, in bytes. Useful for large MoE models. | `536870912` | Type: int |
| `--update-weights-interval` | Interval (in rollout rounds) for syncing weights to inference engines. | `1` | Type: int |

---

## Sampling and Filtering

Arguments for sampling strategies and data filtering during rollout and buffer management.

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--over-sampling-batch-size` | Granularity of the sampling batch. Used to trigger oversampling when filtered samples drop below target. If `None`, defaults to `rollout_batch_size`. | `None` | Type: int |
| `--dynamic-sampling-filter-path` | Path to a filter function for dynamic sampling (e.g., DAPO). Ensures sample diversity (e.g., non-zero reward variance). | `None` | Type: str |
| `--partial-rollout` | Enable partial rollout. Unfinished samples are cached and resumed in the next round. Useful for long responses. | `False` | bool flag (set to enable) |
| `--mask-offpolicy-in-partial-rollout` | Mask previous generation in partial rollout; ensures only on-policy generated tokens are used in training. | `False` | bool flag (set to enable) |
| `--buffer-filter-path` | Path to a function to filter or sort samples in the rollout buffer before training (e.g., `pop_first`). | `None` | Type: str |
| `--rollout-sample-filter-path` | Path to a function that marks individual samples to be excluded from loss calculation. | `None` | Type: str |
| `--rollout-all-samples-process-path` | Path to a function to process all samples (including filtered ones) after rollout. | `None` | Type: str |

---

## Data Arguments

Arguments for dataset configuration, prompt mapping, and training batch sizes.

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--num-rollout` | Total number of rollout steps (sampling-training loop iterations). | `None` | Type: int |
| `--num-epoch` | Number of training epochs. `num_rollout` is calculated as `(num_epoch * dataset_size) // rollout_batch_size`. | `None` | Type: int |
| `--prompt-data` | Path to the JSONL prompt dataset. Each line must be a JSON object. | `None` | Type: str |
| `--input-key` | JSON dataset key for the input/prompt text. | `input` | Type: str |
| `--label-key` | JSON dataset key for the ground truth label. | `None` | Type: str |
| `--metadata-key` | JSON dataset key for extra structured metadata passed to custom functions. | `metadata` | Type: str |
| `--multimodal-keys` | JSON mapping for multimodal media types to data keys (e.g., `'{"image": "image_file"}'`). | `None` | Type: JSON / Dict |
| `--tool-key` | JSON key for tool definitions in the prompt dataset (used when applying chat templates). | `tools` | Type: str |
| `--apply-chat-template` | Format the input using the model's chat template. Recommended for Instruct models. | `False` | bool flag (set to enable) |
| `--apply-chat-template-kwargs` | Extra arguments for the chat template processing. | `{}` | Type: JSON / Dict |
| `--rollout-batch-size` | Number of prompts sampled in each rollout round. | (Required) | Type: int |
| `--n-samples-per-prompt` | Number of responses generated for each prompt (used for algorithms like GRPO). | `1` | Type: int |
| `--global-batch-size` | Total samples per optimizer step. Automatically calculated if `num_steps_per_rollout` is set. | `None` | Type: int |
| `--num-steps-per-rollout` | Number of training steps performed on the data from a single rollout round. | `None` | Type: int |
| `--micro-batch-size` | Micro batch size per GPU. Ignored when `--use-dynamic-batch-size` is enabled. | `1` | Type: int |
| `--use-dynamic-batch-size` | Pack samples of varying lengths to maximize GPU utilization. Ignores `micro_batch_size`. | `False` | bool flag (set to enable) |
| `--max-tokens-per-gpu` | Maximum tokens per GPU when using dynamic batching. | `None` | Type: int |
| `--log-probs-max-tokens-per-gpu` | Maximum tokens per GPU for calculating log probabilities. | `None` | Type: int |
| `--balance-data` | Balance token counts between DP ranks using the `karmarkar_karp` algorithm. | `False` | bool flag (set to enable) |
| `--data-pad-size-multiplier` | Multiplier for data padding size in data processing. | `128` | Type: int |

---

## Evaluation Arguments

Arguments for configuring the evaluation process during training.

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--eval-interval` | Interval (in rollout steps) between evaluations. | `None` | Type: int |
| `--eval-prompt-data` | Path to evaluation dataset(s). Format: `name path [name path ...]`. | `None` | Type: List[str] |
| `--eval-config` | Path to an OmegaConf YAML/JSON file describing evaluation datasets. | `None` | Type: str |
| `--skip-eval-before-train` | Skip the evaluation step before training starts. | `False` | bool flag (set to enable) |
| `--n-samples-per-eval-prompt` | Number of responses to generate for each eval prompt. | `1` | Type: int |
| `--eval-temperature` | Temperature for evaluation sampling. | `None` | Type: float |
| `--eval-top-p` | Top-p sampling threshold for evaluation. | `None` | Type: float |
| `--eval-top-k` | Top-k sampling threshold for evaluation. | `None` | Type: int |
| `--eval-max-response-len` | Maximum response length for evaluation. | `None` | Type: int |
| `--eval-max-prompt-len` | Maximum prompt length for evaluation. | `None` | Type: int |
| `--eval-min-new-tokens` | Minimum tokens to generate for evaluation responses. | `None` | Type: int |
| `--eval-max-context-len` | Maximum context length for evaluation. | `None` | Type: int |
| `--eval-function-path` | Path to a custom evaluation function. | `None` | Type: str |

---

## Algorithm and RL Arguments

Arguments for reinforcement learning algorithms, loss calculation, and off-policy correction.

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--advantage-estimator` | Selection of the reinforcement learning algorithm. | `grpo` | `grpo`, `gspo`, `ppo`, `reinforce_plus_plus`, `on_policy_distillation` |
| `--loss-type` | Selection of the training loss function. | `policy_loss` | `policy_loss`, `sft_loss`, `custom_loss` |
| `--custom-loss-function-path` | Path to a custom loss calculation function (requires `--loss-type custom_loss`). | `None` | Type: str |
| `--lr` | Learning rate for the Actor. | `1e-6` | Type: float |
| `--critic-lr` | Learning rate for the Critic. Defaults to `--lr`. | `None` | Type: float |
| `--critic-lr-warmup-iters` | Number of iterations for Critic learning rate linear warmup. | `0` | Type: int |
| `--num-critic-only-steps` | Number of initial steps dedicated to training only the Critic. | `0` | Type: int |
| `--eps-clip` | PPO clipping range for policy loss. | `0.2` | Type: float |
| `--eps-clip-high` | Upper bound for PPO clip range. Defaults to `--eps-clip`. | `None` | Type: float |
| `--eps-clip-c` | Lower bound for Dual-clip PPO. | `None` | Type: float |
| `--value-clip` | Clipping range for the value function loss (PPO). | `0.2` | Type: float |
| `--kl-coef` | KL penalty coefficient applied to the reward signal for shaping. | `0.0` | Type: float |
| `--use-kl-loss` | Enable KL loss term in the final objective (as in GRPO). | `False` | bool flag (set to enable) |
| `--kl-loss-coef` | Weight of the KL loss term in the final objective. | `0.0` | Type: float |
| `--kl-loss-type` | Selection of the KL loss implementation. | `k1` | `k1`, `k2`, `k3`, `low_var_kl` |
| `--use-unbiased-kl` | Enable unbiased KL estimation. | `False` | bool flag (set to enable) |
| `--entropy-coef` | Entropy loss coefficient to encourage exploration. | `0.0` | Type: float |
| `--gamma` | GAE discount factor. | `1.0` | Type: float |
| `--lambd` | GAE lambda parameter. | `1.0` | Type: float |
| `--normalize-advantages` | Normalize advantages within each batch. | `False` | bool flag (set to enable) |
| `--disable-compute-advantages-and-returns` | Skip advantage calculation (useful for SFT or custom losses). | `False` | bool flag (set to enable) |
| `--use-tis` | Enable Token-level Importance Sampling (TIS) for off-policy correction. | `False` | bool flag (set to enable) |
| `--tis-clip` | Upper clipping threshold for TIS ratios. | `2.0` | Type: float |
| `--tis-clip-low` | Lower clipping threshold for TIS ratios. | `0.0` | Type: float |
| `--custom-tis-function-path` | Path to a custom TIS or Mismatch Importance Sampling (MIS) function. | `None` | Type: str |
| `--custom-pg-loss-reducer-function-path` | Custom reducer function for policy gradient loss (e.g., for Dr.GRPO). | `None` | Type: str |
| `--use-routing-replay` | Enable MoE routing consistency (forward-backward) during training. | `False` | bool flag (set to enable) |
| `--use-rollout-routing-replay` | Enable R3: Replay routing from rollout. Requires Miles Router. | `False` | bool flag (set to enable) |
| `--use-opsm` | Enable Off-Policy Sequence Masking (OPSM). | `False` | bool flag (set to enable) |
| `--opsm-delta` | Threshold for Off-Policy Sequence Masking. | `1e-4` | Type: float |
| `--ref-update-interval` | Interval (in rollout steps) to update the reference model from the Actor. | `None` | Type: int |
| `--reset-optimizer-states` | Reset optimizer history after each rollout round. | `False` | bool flag (set to enable) |
| `--calculate-per-token-loss` | Calculate loss on a per-token basis instead of per-sample. | `False` | bool flag (set to enable) |
| `--disable-grpo-std-normalization` | Disable standard deviation normalization for GRPO. | `False` | bool flag (set to enable) |
| `--disable-rewards-normalization` | Disable normalization of the reward signals. | `False` | bool flag (set to enable) |
| `--use-rollout-entropy` | Calculate entropy during logprob computation. | `False` | bool flag (set to enable) |
| `--use-rollout-logprobs` | Use rollout logprobs for importance sampling ratios. | `False` | bool flag (set to enable) |

---

## Logging and Monitoring

Arguments for WandB, Tensorboard, and general logging.

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--use-wandb` | Enable WandB logging. | `False` | bool flag (set to enable) |
| `--wandb-mode` | WandB operating mode. Overrides `WANDB_MODE`. | `None` | `online`, `offline`, `disabled` |
| `--wandb-project` | WandB project name. | `None` | Type: str |
| `--wandb-group` | WandB group name. | `None` | Type: str |
| `--wandb-team` | WandB team name. | `None` | Type: str |
| `--wandb-host` | WandB host address. | `None` | Type: str |
| `--wandb-key` | WandB API key. | `None` | Type: str |
| `--wandb-run-id` | Specific WandB run ID to resume. | `None` | Type: str |
| `--disable-wandb-random-suffix` | Prevent adding a random suffix to the WandB run name. | `False` | bool flag (set to enable) |
| `--wandb-always-use-train-step` | Use training steps instead of rollout steps for the x-axis. | `False` | bool flag (set to enable) |
| `--use-tensorboard` | Enable Tensorboard logging. | `False` | bool flag (set to enable) |
| `--tb-project-name` | Tensorboard project directory. | `None` | Type: str |
| `--tb-experiment-name` | Tensorboard experiment name. | `None` | Type: str |
| `--log-multi-turn` | Log detailed information for multi-turn conversations. | `False` | bool flag (set to enable) |
| `--log-passrate` | Enable logging of pass@n metrics. | `False` | bool flag (set to enable) |
| `--log-correct-samples` | Explicitly log metrics for correct samples. | `False` | bool flag (set to enable) |
| `--log-reward-category` | JSON key in the reward dictionary used to categorize failures. | `None` | Type: str |

---

## Fault Tolerance

Arguments for handling server failures during rollout.

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--use-fault-tolerance` | Enable fault tolerance for rollout engines. Periodically sends `/health_generate` heartbeats. | `False` | bool flag (set to enable) |
| `--rollout-health-check-interval` | Interval (seconds) between engine heartbeats. | `30.0` | Type: float |
| `--rollout-health-check-timeout` | Timeout (seconds) for heartbeat responses before restarting the engine. | `30.0` | Type: float |
| `--rollout-health-check-first-wait` | Initial grace period (seconds) before starting health checks (useful for model compilation). | `0.0` | Type: float |

---

## Miles Router

Arguments for the specialized Miles text-based router.

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--use-miles-router` | Use text-based routing instead of token-based routing. | `False` | bool flag (set to enable) |
| `--miles-router-middleware-paths` | Paths to custom MilesRouter middleware functions. | `""` | Type: List[str] |
| `--miles-router-timeout` | Timeout for router HTTP requests in seconds. | `None` | Type: float |
| `--miles-router-max-connections` | Maximum concurrent connections for the router. | `None` | Type: int |
| `--miles-router-health-check-failure-threshold` | Number of failures allowed before marking a worker as dead. | `3` | Type: int |

---

## Reward Model Arguments

Arguments for configuring reward signals and post-processing.

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--rm-type` | Built-in reward model selection. | `None` | `math`, `deepscaler`, `gpqa`, `ifbench`, `remote_rm` |
| `--rm-url` | URL for the reward model service (used with `--rm-type remote_rm`). | `None` | Type: str |
| `--reward-key` | JSON key to extract the numerical reward from a returned dictionary. | `None` | Type: str |
| `--eval-reward-key` | Evaluation variant for `--reward-key`. | `None` | Type: str |
| `--custom-rm-path` | Path to a custom Python reward function. | `None` | Type: str |
| `--group-rm` | Compute rewards for an entire group of samples at once. | `False` | bool flag (set to enable) |
| `--custom-reward-post-process-path` | Path to a custom reward post-processor (e.g., for reward shaping). | `None` | Type: str |
| `--custom-convert-samples-to-train-data-path` | Path to a custom data format converter. | `None` | Type: str |

---

## Rollout Buffer Management

Arguments for managing the rollout data buffer.

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--rollout-buffer-url` | URL for the rollout buffer service. | `None` | Type: str |
| `--fetch-trajectory-retry-times` | Retries for fetching trajectory data. `-1` means unlimited. | `-1` | Type: int |
| `--min-batch-collection-ratio` | Minimum batch collection ratio before proceeding. | `1.0` | Type: float |
| `--disable-rollout-trim-samples` | Disable trimming of samples in the buffer. | `False` | bool flag (set to enable) |
| `--use-dynamic-global-batch-size` | Enable dynamic global batch size based on buffer availability. | `False` | bool flag (set to enable) |

---

## Multi-Token Prediction (MTP) Arguments

Arguments for MTP-based training.

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--enable-mtp-training` | Enable MTP layer parameter updates during training. | `False` | bool flag (set to enable) |
| `--mtp-num-layers` | Number of MTP layers to include. | `None` | Type: int |
| `--mtp-loss-scaling-factor` | Scaling factor applied to the MTP loss. | `0.2` | Type: float |

---

## SGLang Backend Arguments

Almost all SGLang server arguments can be passed through by adding the `--sglang-` prefix. For a full list, refer to the [SGLang Server Arguments documentation](https://docs.sglang.io/advanced_features/server_arguments.html).

Commonly used prefixed arguments:
- `--sglang-mem-fraction-static`: Fraction of GPU memory reserved for SGLang.
- `--sglang-context-length`: Maximum context length for SGLang.
- `--sglang-server-concurrency`: Maximum number of concurrent requests.

---

## FSDP Specific Arguments

Arguments applicable when using `--train-backend fsdp`. **Note: The FSDP backend is still under development and experimental.**

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--optimizer` | Optimizer type for FSDP. | `adam` | `adam`, `sgd` |
| `--fsdp-cpu-offload` | Offload parameters and gradients to CPU. | `False` | bool flag (set to enable) |
| `--fsdp-state-dict-cpu-offload` | Offload full state dict to CPU during collection. | `True` | bool flag (set to enable) |
| `--context-parallel-size` | Size of context parallelism. | `1` | Type: int |
| `--attn-implementation` | Selection of the attention implementation. | `flash_attention_2` | `flash_attention_2`, `sdpa`, `eager` |

---

## Debug and Profiling

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--check-weight-update-equal` | Verify that weight updates are equal across ranks. | `False` | bool flag (set to enable) |
| `--save-debug-rollout-data` | Path to save rollout data for offline analysis. | `None` | Type: str |
| `--load-debug-rollout-data` | Path to load debug rollout data (bypasses SGLang). | `None` | Type: str |
| `--debug-rollout-only` | Run the rollout phase only without training. | `False` | bool flag (set to enable) |
| `--debug-train-only` | Run the training phase only without launching SGLang servers. | `False` | bool flag (set to enable) |
| `--dump-details` | Dump exhaustive training details for post-hoc visualization. | `None` | Type: str |
| `--memory-snapshot-dir` | Directory for PyTorch memory snapshots. | `.` | Type: str |
| `--memory-recorder` | Selection of the memory recording backend. | `torch` | `torch`, `memray` |
| `--profile-target` | Selection of training components to profile. | `train_overall` | `train_overall`, `train_actor`, `train_log_probs` |

---

## Environment Variables

Miles recognizes several environment variables for advanced configuration.

| Variable | Description |
| :--- | :--- |
| `MILES_EXPERIMENTAL_ROLLOUT_REFACTOR` | Set to `1` to enable the experimental rollout implementation refactor. |
| `ENABLE_ROUTING_REPLAY` | Internal variable used to enable MoE routing consistency checks during training. |
| `TENSORBOARD_DIR` | Base directory for Tensorboard logs. |
| `MILES_HOST_IP` | Overrides the host IP used for distributed communication. |
| `PYTHONPATH` | Must include the path to your `Megatron-LM` installation when using the Megatron backend. |
