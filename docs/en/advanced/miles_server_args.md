# Miles Server Arguments

This document provides a detailed list of command-line arguments used to configure Miles for RL training and inference. These arguments enable precise control over cluster resources, training backends (Megatron/FSDP), inference optimization via SGLang, and RL algorithmic hyperparameters.

You can find all arguments by running:
```bash
python3 train.py --help
```

## Table of Contents
1. [Cluster and Resource Management](#cluster-and-resource-management)
2. [Training Backend](#training-backend)
3. [Rollout Management](#rollout-management)
4. [Sampling and Filtering](#sampling-and-filtering)
5. [Data Arguments](#data-arguments)
6. [Evaluation Arguments](#evaluation-arguments)
7. [Checkpointing and Resuming](#checkpointing-and-resuming)
8. [Algorithm and RL Arguments](#algorithm-and-rl-arguments)
9. [Logging and Monitoring](#logging-and-monitoring)
10. [Fault Tolerance](#fault-tolerance)
11. [Miles Router](#miles-router)
12. [Reward Model Arguments](#reward-model-arguments)
13. [Rollout Buffer Management](#rollout-buffer-management)
14. [Multi-Token Prediction (MTP) Arguments](#multi-token-prediction-mtp-arguments)
15. [SGLang Backend Arguments](#sglang-backend-arguments)
16. [Megatron Specific Arguments](#megatron-specific-arguments)
17. [FSDP Specific Arguments](#fsdp-specific-arguments)
18. [Debug and Profiling](#debug-and-profiling)
19. [Environment Variables](#environment-variables)
20. [Multi-Turn and Agentic Arguments](#multi-turn-and-agentic-arguments)
21. [Advanced Developer Hooks and CI](#advanced-developer-hooks-and-ci)
22. [Miscellaneous and System](#miscellaneous-and-system)

---

## Cluster and Resource Management

Arguments for configuring Ray cluster resources and GPU allocation.

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--actor-num-nodes` | Number of nodes for training the Actor. | `1` | Type: int |
| `--actor-num-gpus-per-node` | Number of GPUs per node for training the Actor. | `8` | Type: int |
| `--critic-num-nodes` | Number of nodes for the Critic. Defaults to `--actor-num-nodes`. | `None` | Type: int |
| `--critic-num-gpus-per-node` | Number of GPUs per node for the Critic. Defaults to `--actor-num-gpus-per-node`. | `None` | Type: int |
| `--rollout-num-gpus` | Total number of GPUs required for rollout (inference). In `--colocate` mode, this is ignored and set to `actor-num-gpus-per-node * actor-num-nodes` (and plus critic GPUs if enabled). | `None` | Type: int |
| `--rollout-num-gpus-per-engine` | Number of GPUs per inference engine, same as `tp_size` in SGLang. For multi-node serving, this should be the total GPU count for the engine. | `1` | Type: int |
| `--num-gpus-per-node` | Total GPUs per node on the machine. Specify if using fewer than 8 GPUs per node in colocate mode. | `8` | Type: int |
| `--colocate` | Deploy training and rollout on the same GPUs. Enables `--offload-train` and `--offload-rollout`. | `False` | bool flag (set to enable) |
| `--offload` | Equivalent to setting both `--offload-train` and `--offload-rollout` to true. | `False` | bool flag (set to enable) |
| `--offload-train` | Offload the Actor to CPU during the rollout phase. Always enabled when `--colocate` is set. | `None` | bool flag (set to enable) |
| `--offload-rollout` | Offload the rollout generator to CPU during the training phase. Always enabled when `--colocate` is set. | `None` | bool flag (set to enable) |
| `--prefill-num-servers` | Number of dedicated prefill servers for PD disaggregation. | `None` | Type: int |
| `--distributed-backend` | Backend for distributed communication. | `nccl` | `nccl`, `gloo` |
| `--distributed-timeout-minutes` | Timeout for distributed operations in minutes. | `10` | Type: int |

---

## Training Backend

Arguments for configuring the training engine (Megatron or FSDP).

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--train-backend` | The backend for training. | `"megatron"` | `megatron`, `fsdp` |
| `--qkv-format` | The QKV layout. | `"thd"` | `thd`, `bshd` |
| `--optimizer` | Optimizer type. | `adam` | `adam`, `sgd` |
| `--lr` | Learning rate for the Actor. | `1e-6` | Type: float |
| `--lr-warmup-init` | Initial learning rate for warmup. | `0.0` | Type: float |
| `--min-lr` | Minimum learning rate after decay. | `0.0` | Type: float |
| `--lr-decay-style` | Learning rate decay style. | `constant`(FSDP), `linear`(Megatron) | Type: str |
| `--lr-warmup-iters` | Number of iterations for warmup. | `0` | Type: int |
| `--lr-decay-iters` | Number of iterations for learning rate decay. | `None` | Type: int |
| `--lr-warmup-fraction` | Fraction of total steps to warmup. | `None` | Type: float |
| `--adam-beta1` | Beta1 for Adam optimizer. | `0.9` | Type: float |
| `--adam-beta2` | Beta2 for Adam optimizer. | `0.95` | Type: float |
| `--adam-eps` | Epsilon for Adam optimizer. | `1e-8` | Type: float |
| `--true-on-policy-mode` | Enable True On-Policy mode, which strictly ensures that data is generated by the current policy during training (only for FSDP). | `False` | bool flag (set to enable) |
| `--train-env-vars` | Extra environment variables for training process, e.g. PyTorch memory management ones. | `{}` | Type: JSON / Dict |
| `--train-memory-margin-bytes` | Reserved memory margin for training in bytes. Defaults to 1GB. | `1073741824` | Type: int |
| `--disable-weights-backuper` | Disable weights backuper to save host memory. By default, this feature is enabled. | `False` | bool flag (set to disable) |
| `--custom-model-provider-path` | Path to a custom function that replaces the default model provider. [details](../get_started/customization.md#20-model-provider---custom-model-provider-path) | `None` | Type: str |
| `--recompute-loss-function` | Enable recomputing the loss function to save memory during training. | `False` | bool flag (set to enable) |
| `--log-probs-chunk-size` | Chunk size for computing log probabilities to save memory. `-1` means no chunking. | `-1` | Type: int |
| `--keep-old-actor` | Keep the Actor model loaded during the training process. | `False` | bool flag (set to enable) |
| `--update-weight-buffer-size` | Buffer size for updating weights, in bytes. This is used for updating weights by chunk and should be useful for MoE models. | `536870912` | Type: int |
| `--update-weights-interval` | Interval (in rollout rounds) for syncing weights to inference engines. | `1` | Type: int |
| `--fp16` | Enable FP16 mixed precision. | `False` | bool flag |
| `--context-parallel-size` | Size of context parallelism. | `1` | Type: int |
| `--deterministic-mode` | Enable deterministic mode for reproducibility. | `False` | bool flag |

---

## Rollout Management

Arguments for configuring the rollout (inference) process and custom rollout logic.

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--hf-checkpoint` | Path to the Huggingface checkpoint used to initialize SGLang and provide the tokenizer. It must have the same architecture as the model being trained. It doesn't necessary need to contain the most up-to-date parameters. | `None` | Type: str |
| `--model-name` | The name of the model, this is used to convert the megatron weights into huggingface format. If not set, we will use `type(AutoConfig.from_pretrained(args.hf_checkpoint)).__name__.lower()` as model_name. Also, sometimes this will help alleviate the bug that transformers cannot find certain model. | `None` | Type: str |
| `--rollout-function-path` | Path to the rollout generation function. Use this to inject custom logic (e.g., for multi-turn or tool use). For more details, see [customization](../get_started/customization.md#1-rollout-function---rollout-function-path). | `miles.rollout.sglang_rollout.generate_rollout` (or `miles.rollout.inference_rollout.inference_rollout_common.InferenceRolloutFn` when `MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1`) | Type: str |
| `--rollout-temperature` | Sampling temperature for the inference engine during rollout. | `1.0` | Type: float |
| `--rollout-top-p` | Top-p (nucleus) sampling threshold during rollout. | `1.0` | Type: float |
| `--rollout-top-k` | Top-k sampling threshold during rollout. `-1` means disabled. | `-1` | Type: int |
| `--rollout-max-context-len` | The maximum context size for the inference engine during rollout. It should no exceed the `max_position_embeddings` in Huggingface model's `config.json`. | `None` | Type: int |
| `--rollout-max-prompt-len` | Maximum length of the prompt. Longer prompts are filtered during dataset initialization. This is not recommended if the dataset is large. | `None` | Type: int |
| `--rollout-max-response-len` | Maximum length of the response (`max_tokens` in SGLang). | `None` | Type: int |
| `--rollout-skip-special-tokens` | Skip special tokens in the response. Useful when the response is used as a prompt for the next rollout. | `False` | bool flag (set to enable) |
| `--rollout-stop` | Stop words for the inference engine. Can be a single string or a list of strings. It may be hard to pass special tokens in command line, in that case `--rollout-stop-token-ids` can be used. | `None` | Type: List[str] |
| `--rollout-stop-token-ids` | Stop token IDs for the inference engine. | `None` | Type: List[int] |
| `--rollout-shuffle` | Shuffle the prompts during rollout. | `False` | bool flag (set to enable) |
| `--rollout-seed` | Seed for the random number generator during rollout (used for shuffling and sampling). | `42` | Type: int |
| `--rollout-external` | Use external SGLang instances instead of launching them inside the framework. | `False` | bool flag (set to enable) |
| `--rollout-external-engine-addrs` | Addresses and ports of the external engines. | `None` | Type: List[str] |
| `--custom-generate-function-path` | Path to override only the `generate` step within the default rollout function. See [customization](../get_started/customization.md#2-custom-generate-function---custom-generate-function-path) for more details. | `None` | Type: str |
| `--custom-rollout-log-function-path` | Path to a custom function for logging training rollout data. See [customization](../get_started/customization.md#14-logging-functions) for more details. | `None` | Type: str |
| `--custom-eval-rollout-log-function-path` | Path to a custom function for logging evaluation rollout data. See [customization](../get_started/customization.md#14-logging-functions) for more details. | `None` | Type: str |
| `--rollout-data-postprocess-path` | Path to a function called after all rollout data (including log probs) is ready. See [customization](../get_started/customization.md#8-rollout-data-postprocess---rollout-data-postprocess-path) for more details. | `None` | Type: str |

---

## Sampling and Filtering

Arguments for sampling strategies and data filtering during rollout and buffer management.

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--over-sampling-batch-size` | Defines the granularity of the sampling batch in the rollout function. When the number of available samples falls below the target, a sampling operation of size `over_sampling_batch_size` will be triggered. Regardless of whether partial rollout is used or filters are applied, the sampling granularity is always determined by this value. If this value is `None`, `rollout_batch_size` will be used as the default `over_sampling_batch_size`. | `None` | Type: int |
| `--dynamic-sampling-filter-path` | Path to a filter function for dynamic sampling. See [customization](../get_started/customization.md#4-dynamic-sampling-filter---dynamic-sampling-filter-path) for more details. | `None` | Type: str |
| `--partial-rollout` | Enable partial rollout (unfinished samples during dynamic sampling will be recycled back to data buffer). Useful for long responses. | `False` | bool flag (set to enable) |
| `--mask-offpolicy-in-partial-rollout` | Mask previous generation in partial rollout. Ensures only on-policy generated tokens are used in training. | `False` | bool flag (set to enable) |
| `--buffer-filter-path` | Path to a function to filter or sort samples in the rollout buffer before training. See [customization](../get_started/customization.md#5-buffer-filter---buffer-filter-path) for more details. | `None` | Type: str |
| `--rollout-sample-filter-path` | Path to a function that marks individual samples to be excluded from loss calculation. See [customization](../get_started/customization.md#6-rollout-sample-filter---rollout-sample-filter-path) for more details. | `None` | Type: str |
| `--rollout-all-samples-process-path` | Path to a function to process all samples (including filtered ones) after rollout. See [customization](../get_started/customization.md#7-rollout-all-samples-process---rollout-all-samples-process-path) for more details. | `None` | Type: str |

---

## Data Arguments

Arguments for dataset configuration, prompt mapping, and training batch sizes.

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--num-rollout` | Number of rollout steps. If not set, miles will calculate the number of rollout steps from the dataset size. | `None` | Type: int |
| `--num-epoch` | Number of epochs for the training. If set, `num_rollout` is calculated as `(num_epoch * dataset_size) // rollout_batch_size`. If both `--num-epoch` and `--num-rollout` are set, `--num-epoch` will be ignored. | `None` | Type: int |
| `--disable-rollout-global-dataset` | Disable the global dataset for rollout. If set, the rollout will use the `--prompt-data` as the prompt dataset, and the prompts for rollout will be sampled from the dataset. If not set, you need to manage the data by your self. | `False` | bool flag (set to disable) |
| `--data-source-path` | Path to a custom Python class for the rollout data source. See [customization](../get_started/customization.md#15-data-source---data-source-path) for more details. | `miles.rollout.data_source.RolloutDataSourceWithBuffer` | Type: str |
| `--prompt-data` | Path to the prompt dataset (JSONL format) and each line should contains `--input-key` and `--label-key` which will be used as the prompt and the label respectively. If you want to use a custom template, you can set `--apply-chat-template` to true | `None` | Type: str |
| `--input-key` | Key in the JSONL data representing the user input/prompt. | `"input"` | Type: str |
| `--label-key` | Key in the JSONL data representing the label/ground truth. | `None` | Type: str |
| `--metadata-key` | When need to add tools during apply_chat_template, you should provide the key for the tools in the prompt dataset. | `"metadata"` | Type: str |
| `--multimodal-keys` | JSON string for multimodal data mapping media types to data keys. Example: `'{"image": "image_file"}'` | `None` | Type: str |
| `--tool-key` | JSON key for tool definitions in the prompt dataset (used when applying chat templates). | `"tools"` | Type: str |
| `--apply-chat-template` | Whether to apply the chat template to the input prompt. The input should be the same structure as an openai message, e.g. `[{'role': 'user', 'content': 'blabla'}]`. | `False` | bool flag (set to enable) |
| `--apply-chat-template-kwargs` | Extra arguments for the chat template processing (JSON string). | `"{}"` | Type: str |
| `--rollout-batch-size` | Number of prompts per rollout batch. The total data returned should be `rollout_batch_size` * `n_samples_per_prompt`. | Required | Type: int |
| `--n-samples-per-prompt` | Number of responses to generate for each prompt. | `1` | Type: int |
| `--num-steps-per-rollout` | Number of training steps to perform per rollout batch. It is equivalent to setting gbs as `rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout`. | `None` | Type: int |
| `--use-dynamic-batch-size` | Dynamically packs variable-length samples into micro-batches to maximize GPU utilization, ensuring the total token count per batch does not exceed `--max-tokens-per-gpu`. For example, with a 300-token limit, samples of lengths 100, 200, and 300 would be packed into two batches: `[100, 200]` and `[300]`. Ignores `micro_batch_size`. | `False` | bool flag (set to enable) |
| `--max-tokens-per-gpu` | The maximum number of tokens per GPU for dynamic batch size. Note that when enabling context parallel (CP), the max tokens per gpu should be around `max_response_len // cp_size` instead of `max_response_len`. | `None` | Type: int |
| `--log-probs-max-tokens-per-gpu` | The maximum number of tokens per GPU for calculating log probs. This is used to calculate the log probs of the responses during rollout, and should be set to a larger value than `max_tokens_per_gpu` if you want better performance. | `None` | Type: int |
| `--balance-data` | Balance the number of tokens between data parallel ranks with `karmarkar_karp` for verl. Note that this may allocate the different response of the same prompt into different training steps. | `False` | Type: bool |
| `--data-pad-size-multiplier` | Multiplier for data padding size in data processing. | `128` | Type: int |
| `--micro-batch-size` | Micro batch size per GPU. Ignored when `--use-dynamic-batch-size` is enabled. | `1` | Type: int |
| `--global-batch-size` | Total samples per optimizer step. Automatically calculated if `num_steps_per_rollout` is set. | `None` | Type: int |

---

## Evaluation Arguments

Arguments for configuring the evaluation process during training.

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--eval-interval` | Interval (in rollout steps) between evaluations. | `None` | Type: int |
| `--eval-prompt-data` | List of name and path pairs for evaluation datasets (e.g., `aime /path/to/aime.jsonl`). | `None` | Type: List[str] |
| `--eval-config` | Path to an OmegaConf YAML/JSON file describing evaluation datasets (overrides `--eval-prompt-data`). | `None` | Type: str |
| `--skip-eval-before-train` | Skip the evaluation step before training starts. | `False` | bool flag (set to enable) |
| `--n-samples-per-eval-prompt` | Nnumber of responses for each prompt in generation. | `1` | Type: int |
| `--eval-temperature` | Temperature for evaluation (defaults to rollout temperature if not set). | `None` | Type: float |
| `--eval-top-p` | Top-p sampling threshold for evaluation (defaults to rollout top-p if not set). | `None` | Type: float |
| `--eval-top-k` | Top-k sampling threshold for evaluation (defaults to rollout top-k if not set). | `None` | Type: int |
| `--eval-max-response-len` | Maximum response length for evaluation (defaults to rollout max response length if not set). | `None` | Type: int |
| `--eval-max-prompt-len` | Maximum prompt length for evaluation. | `None` | Type: int |
| `--eval-min-new-tokens` | Minimum tokens to generate for evaluation responses (Not used). | `None` | Type: int |
| `--eval-max-context-len` | Maximum context length for evaluation (defaults to rollout max context length if not set). | `None` | Type: int |
| `--eval-function-path` | Path to a custom evaluation function. See [customization](../get_started/customization.md#16-evaluation-function---eval-function-path) for more details. | `None` | Type: str |
| `--eval-input-key` | JSON key for input text in evaluation datasets. | `None` | Type: str |
| `--eval-label-key` | JSON key for ground truth labels in evaluation datasets. | `None` | Type: str |
| `--eval-tool-key` | JSON key for tool definitions in evaluation datasets. | `None` | Type: str |

---

## Checkpointing and Resuming

Arguments for saving and loading model states.

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--load` | Path to the training model checkpoint to load. | `None` | Type: str |
| `--save` | Path to save checkpoints. | `None` | Type: str |
| `--save-interval` | Interval (in rollout steps) to save checkpoints. Requires `--save` to be set. | `None` | Type: int |
| `--async-save` | Enable asynchronous checkpoint saving (Megatron backend only). | `False` | bool flag (set to enable) |
| `--save-hf` | Path to save the model in HuggingFace format when using Megatron backend. The model will be saved to `save_hf.format(rollout_id)`. | `None` | Type: str |
| `--no-save-optim` | If set, optimizer state is not saved with checkpoints to reduce size, but prevents resumption of training. | `False` | Type: bool |
| `--ref-load` | Path to the reference model checkpoint. Used as initial checkpoint if `--load` is not set. | `None` | Type: str |
| `--ref-ckpt-step` | The checkpoint step for reference model. | `None` | Type: int |
| `--critic-load` | Checkpoint to load for the critic model. | value of `--load` | Type: str |
| `--critic-save` | Path to save the critic model. | `None` | Type: str |
| `--start-rollout-id` | The starting rollout step, if not set, will try to load the step from `--load` when doing continue training, otherwise will be set to `0`, meaning training from start. | `None` | Type: int |

---

## Algorithm and RL Arguments

Arguments for reinforcement learning algorithms and loss calculation.

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--advantage-estimator` | Advantage estimator to use. | `"grpo"` | `grpo`, `gspo`, `ppo`, `reinforce_plus_plus`, `reinforce_plus_plus_baseline`, `on_policy_distillation` |
| `--loss-type` | Type of loss function to use. | `"policy_loss"` | `policy_loss`, `sft_loss`, `custom_loss` |
| `--custom-loss-function-path` | Path to a custom loss calculation function (requires `--loss-type custom_loss`). see [customization](../get_started/customization.md#9-custom-loss-function---custom-loss-function-path) for more details. | `None` | Type: str |
| `--critic-lr` | Learning rate for the Critic. Defaults to `--lr`. | `None` | Type: float |
| `--critic-lr-warmup-iters` | Number of iterations for Critic learning rate linear warmup. | `0` | Type: int |
| `--num-critic-only-steps` | Number of initial steps dedicated to training only the Critic. | `0` | Type: int |
| `--eps-clip` | PPO clip range. | `0.2` | Type: float |
| `--eps-clip-high` | PPO clip upper range (defaults to `--eps-clip` if not set). | `None` | Type: float |
| `--eps-clip-c` | Lower bound for [Dual-clip PPO](https://arxiv.org/pdf/1912.09729). | `None` | Type: float |
| `--value-clip` | Clip range for value loss. | `0.2` | Type: float |
| `--kl-coef` | KL penalty coefficient for reward shaping. This is applied to the reward signal before advantage calculation. | `0.00` | Type: float |
| `--use-kl-loss` | Enable KL loss term in the final objective (as in GRPO). | `False` | bool flag (set to enable) |
| `--kl-loss-coef` | Weight of the KL loss term in the final objective. | `0.0` | Type: float |
| `--kl-loss-type` | Selection of the KL loss implementation. | `k1` | `k1`, `k2`, `k3`, `low_var_kl` |
| `--use-unbiased-kl` | Enable unbiased KL estimation. | `False` | bool flag (set to enable) |
| `--entropy-coef` | Coefficient for entropy regularization. | `0.0` | Type: float |
| `--gamma` | PPO GAE gamma. | `1.0` | Type: float |
| `--lambd` | PPO GAE lambda. | `1.0` | Type: float |
| `--normalize-advantages` | Normalize advantages within each batch. | `False` | bool flag (set to enable) |
| `--disable-compute-advantages-and-returns` | Disables the calculation of advantages and returns. This is typically used for SFT or custom loss functions where value estimation is not required. | `False` | bool flag (set to enable) |
| `--use-tis` | Enable Token-level Importance Sampling (TIS) from this [blog](https://fengyao.notion.site/off-policy-rl#279721e3f6c48092bbe2fcfe0e9c6b33). | `False` | Type: bool |
| `--tis-clip` | Clipping threshold C for importance sampling ratios to control variance. | `2.0` | Type: float |
| `--tis-clip-low` | Lower bound clipping threshold C for importance sampling ratios to control variance. | `0.0` | Type: float |
| `--custom-tis-function-path` | Path to a custom TIS or MIS function. See [customization](../get_started/customization.md#10-custom-tisrs-function---custom-tis-function-path) for more details. | `None` | Type: str |
| `--custom-pg-loss-reducer-function-path` | Custom reducer function for policy gradient loss. See [customization](../get_started/customization.md#11-custom-pg-loss-reducer---custom-pg-loss-reducer-function-path) for more details. | `None` | Type: str |
| `--use-routing-replay` | Enable [Routing Replay](https://arxiv.org/abs/2507.18071). | `False` | bool flag (set to enable) |
| `--use-rollout-routing-replay` | Enable R3: [Rollout Routing Replay](https://arxiv.org/pdf/2510.11370). | `False` | bool flag (set to enable) |
| `--use-opsm` | Enable Off-Policy Sequence Masking (OPSM). | `False` | bool flag (set to enable) |
| `--opsm-delta` | The threshold for Off-Policy Sequence Masking (OPSM). | `1e-4` | Type: float |
| `--get-mismatch-metrics` | Whether to calculate the mismatch metrics. It will **only return mismatch metrics** but not change the loss in any way. | `False` | bool flag (set to enable) |
| `--ref-update-interval` | Interval (in rollout steps) to update ref model from actor. If `None`, ref model is not updated. | `None` | Type: int |
| `--reset-optimizer-states` | Resets the optimizer state after each rollout round. This clears the optimization history, which can improve stability or satisfy specific experimental requirements. | `False` | bool flag (set to enable) |
| `--disable-grpo-std-normalization` | Disable standard deviation normalization for GRPO. From [Dr.GRPO](https://arxiv.org/pdf/2503.20783) | `False` | bool flag (set to enable) |
| `--disable-rewards-normalization` | Disable rewards normalization. | `False` | bool flag (set to enable) |
| `--use-rollout-entropy` | Enable entropy calculation when calculating the logprobs from actor and reference model. This is useful for doing special loss mask. | `False` | bool flag (set to enable) |
| `--use-rollout-logprobs` | Use rollout logprobs for importance sampling ratios, use the logprobs from the actor model if not set. | `False` | bool flag (set to enable) |
| `--calculate-per-token-loss` | Calculate loss on a per-token basis. | `False` | bool flag (set to enable) |
| `--seed` | Random seed for the training process. | `1234` | Type: int |
| `--clip-grad` | Maximum gradient norm for gradient clipping. | `1.0` | Type: float |

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
| `--wandb-dir` | Directory to store wandb logs. Default is ./wandb in current directory. | `None` | Type: str |
| `--disable-wandb-random-suffix` | Disable adding a random suffix to the wandb run name. By default, we will add a random 6 length string with characters to the run name. | `False` | bool flag (set to enable) |
| `--wandb-always-use-train-step` | Use training steps instead of rollout steps for the x-axis. | `False` | bool flag (set to enable) |
| `--use-tensorboard` | Enable Tensorboard logging. | `False` | bool flag (set to enable) |
| `--tb-project-name` | Tensorboard project directory. | `None` | Type: str |
| `--tb-experiment-name` | Tensorboard experiment name. | `None` | Type: str |
| `--tensorboard-dir` | Directory to store Tensorboard logs. | `None` | Type: str |
| `--log-multi-turn` | Log detailed information for multi-turn conversations. | `False` | bool flag (set to enable) |
| `--log-passrate` | Enable logging of `pass@n` metrics. | `False` | bool flag (set to enable) |
| `--log-correct-samples` | Explicitly log metrics for correct samples. | `False` | bool flag (set to enable) |
| `--log-reward-category` | Log statistics of the category of reward, such as why the reward function considers it as failed. Specify the key in the reward dict using this argument. | `None` | Type: str |

---

## Fault Tolerance

Arguments for handling server failures during rollout.

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--use-fault-tolerance` | Enable fault tolerance for rollout engines. Periodically sends `/health_generate` heartbeats. | `False` | bool flag (set to enable) |
| `--rollout-health-check-interval` | Interval in seconds between rollout engine `/health_generate` checks during generate/eval. | `30.0` | Type: float |
| `--rollout-health-check-timeout` | Timeout in seconds to wait for a rollout engine `/health_generate` response before killing it. | `30.0` | Type: float |
| `--rollout-health-check-first-wait` | Initial grace period (in seconds) before starting health checks. This allows time for model compilation and initialization. Increase this value significantly when using deepgemm. | `0.0` | Type: float |

---

## Miles Router

Arguments for the specialized Miles text-based router.

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--use-miles-router` | Use text-based routing instead of token-based routing. | `False` | bool flag (set to enable) |
| `--miles-router-middleware-paths` | Paths to custom MilesRouter middleware functions. See [customization](../get_started/customization.md#18-miles-router-middleware---miles-router-middleware-paths) for more details. | `""` | Type: List[str] |
| `--miles-router-timeout` | Timeout for router HTTP requests in seconds. | `None` | Type: float |
| `--miles-router-max-connections` | Max connections for MilesRouter HTTP client. | `None` | Type: int |
| `--miles-router-health-check-failure-threshold` | Number of consecutive failures before marking a worker as unhealthy. | `3` | Type: int |

---

## Reward Model Arguments

Arguments for configuring reward signals and post-processing.

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--rm-type` | Built-in reward model selection. | `None` | `remote_rm`, `deepscaler`, `dapo`, `math`, `f1`, `gpqa`, `ifbench`, `random` |
| `--rm-url` | URL for the reward model service (used with `--rm-type remote_rm`). | `None` | Type: str |
| `--reward-key` | JSON key to extract the numerical reward from a returned dictionary if reward model return a dict instead of a value. | `None` | Type: str |
| `--eval-reward-key` | Evaluation variant for `--reward-key`. | `None` | Type: str |
| `--custom-rm-path` | Path to a custom Python reward function. See [customization](../get_started/customization.md#3-reward-model---custom-rm-path) for more details. | `None` | Type: str |
| `--group-rm` | Compute rewards for an entire group of samples at once. | `False` | bool flag (set to enable) |
| `--custom-reward-post-process-path` | Path to a custom reward post-processor. See [customization](../get_started/customization.md#12-reward-post-processing---custom-reward-post-process-path) for more details. | `None` | Type: str |
| `--custom-convert-samples-to-train-data-path` | Path to a custom data format converter. See [customization](../get_started/customization.md#13-samples-to-train-data-conversion---custom-convert-samples-to-train-data-path) for more details. | `None` | Type: str |

---

## Rollout Buffer Management

Arguments for managing the rollout data buffer.

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--rollout-buffer-url` | URL for the rollout buffer service. | `None` | Type: str |
| `--fetch-trajectory-retry-times` | Number of times to retry fetching trajectory, -1 means unlimited retry. | `-1` | Type: int |
| `--min-batch-collection-ratio` | Minimum batch collection ratio before proceeding. | `1.0` | Type: float |
| `--disable-rollout-trim-samples` | Disable trim samples in rollout buffer when converting samples to train data. | `False` | bool flag (set to enable) |
| `--use-dynamic-global-batch-size` | Enable dynamic global batch size, disable trim samples in rollout buffer when converting samples to train data. | `False` | bool flag (set to enable) |
| `--rollout-task-type` | Type of task being performed. | `math` | Type: str |
| `--loss-mask-type` | Selection of the token masking logic. | `"qwen"` | `qwen`, `qwen3`, `distill_qwen` |

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

Most SGLang server arguments can be passed through by adding the `--sglang-` prefix (some are intentionally skipped, e.g. `model_path`, `tp_size`, `port`, `nnodes`, `node_rank`). For a full list, refer to the [SGLang Server Arguments documentation](https://docs.sglang.io/advanced_features/server_arguments.html).

Commonly used arguments:

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--sglang-mem-fraction-static` | Fraction of GPU memory to reserve for SGLang KV cache. | `0.9` | Type: float |
| `--sglang-server-concurrency` | Maximum number of concurrent requests. | `512` | Type: int |
| `--sglang-router-ip` | IP address of the SGLang router. | `None` | Type: str |
| `--sglang-router-port` | Port of the SGLang router. | `None` | Type: int |
| `--sglang-router-request-timeout-secs` | Timeout for requests to the SGLang router. | `14400` | Type: int |

---

## Megatron Specific Arguments

Arguments applicable when using `--train-backend megatron`.

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--megatron-to-hf-mode` | Method to convert Megatron weights to HuggingFace format for SGLang integration. | `raw` | `raw`, `bridge` |

---

## FSDP Specific Arguments

Arguments applicable when using `--train-backend fsdp`. **Note: The FSDP backend is still under development and experimental.**

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--warmup-ratio` | Ratio of total steps for warmup. | `0.03` | Type: float |
| `--weight-decay` | Weight decay for the optimizer. | `0.0` | Type: float |
| `--gradient-checkpointing` | Enable gradient checkpointing. | `False` | bool flag |
| `--fsdp-cpu-offload` | Offload parameters and gradients to CPU. | `False` | bool flag (set to enable) |
| `--fsdp-state-dict-cpu-offload` | Offload full state dict to CPU during collection. | `False` | bool flag (set to enable) |
| `--fsdp-cpu-backend` | CPU backend for FSDP CPU offload. | `gloo` | `gloo`, `None` |
| `--attn-implementation` | Selection of the attention implementation. | `flash_attention_2` | `flash_attention_2`, `sdpa`, `eager` |
| `--use-pytorch-profiler` | Enable PyTorch-native profiling. | `False` | bool flag |
| `--profile-step-start` | Starting step for profiling. | `10` | Type: int |
| `--profile-step-end` | Ending step for profiling. | `12` | Type: int |
| `--lr-wsd-decay-iters` | Number of iterations for WSD decay. | `None` | Type: int |
| `--lr-wsd-decay-style` | Decay style for WSD. | `None` | Type: str |
| `--use-checkpoint-lr-scheduler` | Use the checkpoint's LR scheduler state. | `False` | bool flag (set to enable) |
| `--override-lr-scheduler` | Override the loaded LR scheduler state. | `False` | bool flag (set to enable) |
| `--wandb-run-name` | Specific run name for WandB (FSDP backend). | `None` | Type: str |

---

## Debug and Profiling

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--check-weight-update-equal` | Verify that weight updates are equal across ranks. | `False` | bool flag (set to enable) |
| `--save-debug-rollout-data` | Path to save rollout data for offline analysis. | `None` | Type: str |
| `--load-debug-rollout-data` | Path to load debug rollout data (bypasses SGLang). | `None` | Type: str |
| `--load-debug-rollout-data-subsample` | Percentage of debug data to load (0.0 to 1.0). | `None` | Type: float |
| `--debug-rollout-only` | Run the rollout phase only without training. | `False` | bool flag (set to enable) |
| `--debug-train-only` | Run the training phase only without launching SGLang servers. | `False` | bool flag (set to enable) |
| `--save-debug-train-data` | Path to save training batches for offline math debugging. | `None` | Type: str |
| `--dump-details` | Dump exhaustive training details for post-hoc visualization. | `None` | Type: str |
| `--memory-snapshot-path` | Path to save memory snapshots. | `snapshot.pickle` | Type: str |
| `--record-memory-history` | Record memory history for snapshots. | `False` | bool flag (set to enable) |
| `--memory-snapshot-dir` | Directory for PyTorch memory snapshots. | `.` | Type: str |
| `--memory-snapshot-num-steps` | Number of steps to record before saving snapshot. | `None` | Type: int |
| `--memory-recorder` | Selection of the memory recording backend. | `torch` | `torch`, `memray` |
| `--profile-target` | Training components to profile (accepts multiple). | `train_overall` | `train_overall`, `train_actor`, `train_log_probs` |

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

---

## Multi-Turn and Agentic Arguments

Arguments for managing interactions and tools. Only available when `MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1` and the rollout/generate function exposes `add_arguments`.

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--generate-max-turns` | Maximum number of turns in a conversation. | `16` | Type: int |
| `--generate-tool-specs-path` | Path to the tool specifications (JSON). | `None` | Type: str |
| `--generate-tool-call-parser` | The parser used to extract tool calls from text. | `None` | Type: str |
| `--generate-execute-tool-function-path` | Path to the function that executes the tool. | `None` | Type: str |
| `--generate-multi-samples` | Whether to generate multiple samples within one turn. | `False` | bool flag (set to enable) |

---

## Advanced Developer Hooks and CI

Hooks for custom logic and Continuous Integration testing flags.

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--custom-megatron-init-path` | Path to custom Megatron initialization logic. See [customization](../get_started/customization.md#17-megatron-hooks) for more details. | `None` | Type: str |
| `--custom-megatron-before-log-prob-hook-path` | Hook called before calculating log probabilities. See [customization](../get_started/customization.md#17-megatron-hooks) for more details.| `None` | Type: str |
| `--custom-megatron-before-train-step-hook-path` | Hook called before each training step. See [customization](../get_started/customization.md#17-megatron-hooks) for more details. | `None` | Type: str |
| `--ci-test` | Enable Continuous Integration testing mode. | `False` | bool flag |
| `--ci-disable-kl-checker` | Disable KL divergence sanity checks in CI. | `False` | bool flag |
| `--ci-metric-checker-key` | Metric key to monitor for pass/fail in CI. | `None` | Type: str |
| `--ci-metric-checker-threshold` | Pass/fail threshold (minimum value) for the monitored metric. | `None` | Type: float |
| `--ci-save-grad-norm` | Path to save gradient norms for CI comparison. | `None` | Type: str |
| `--ci-load-grad-norm` | Path to load gradient norms for CI verification. | `None` | Type: str |

---

## Miscellaneous and System

General arguments for infrastructure and configuration overrides.

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--http-proxy` | HTTP proxy server for remote reward model calls. | `None` | Type: str |
| `--use-distributed-post` | Use distributed POST requests for remote reward models. | `False` | bool flag (set to enable) |
| `--custom-config-path` | Path to the YAML config for custom function arguments. | `None` | Type: str |
| `--padded-vocab-size` | Manually specify the vocab size for padding. | `None` | Type: int |
