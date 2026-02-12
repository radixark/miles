# Miles Server Arguments

This document provides a detailed list of command-line arguments used to configure Miles for RL training and inference. These arguments enable precise control over cluster resources, training backends (Megatron/FSDP), inference optimization via SGLang, and RL algorithmic hyperparameters.

You can find all arguments by running:
```bash
python3 train.py --help
```

Note that this document is based on commit `a93d484` and was last updated on 02/09/2026. We try our best to ensure the quality and accuracy of these documents. Even so, it's hard to accurately describe all the hundreds of parameters' effect on such complex RL scenarios. This doc is for reference and may contain some tiny errors.

## Argument Sources

Miles acts as an orchestrator that integrates multiple frameworks. To help identify where an argument is directed, we follow these prefix conventions:

*   **`--sglang-*`**: Arguments passed directly to the **SGLang** rollout.
*   **`--router-*`**: Arguments directed to the **SGLang Model Gateway/Router**.
*   **No Prefix**: Default arguments corresponding to **Megatron-LM** (when using the Megatron backend) or **Miles native** configuration.
*   **`--fsdp-*`**: Specific arguments for the experimental **FSDP** backend.

**Note** that Arguments labeled as **Megatron-LM (Reset by Miles)** are native Megatron-LM parameters where Miles has modified the default value or behavior to better suit RL training workflows.

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

## Cluster and Resource Management

Arguments for configuring Ray cluster resources and GPU allocation.

| Argument | Description | Default | Options | Source |
| :--- | :--- | :--- | :--- | :--- |
| `--actor-num-nodes` | Number of nodes for training the Actor. | `1` | Type: int | Miles Native |
| `--actor-num-gpus-per-node` | Number of GPUs per node for training the Actor. | `8` | Type: int | Miles Native |
| `--critic-num-nodes` | Number of nodes for the Critic. Defaults to `--actor-num-nodes`. | `None` | Type: int | Miles Native |
| `--critic-num-gpus-per-node` | Number of GPUs per node for the Critic. Defaults to `--actor-num-gpus-per-node`. | `None` | Type: int | Miles Native |
| `--rollout-num-gpus` | Total number of GPUs required for rollout (inference). In `--colocate` mode, this is ignored and set to `actor-num-gpus-per-node * actor-num-nodes` (plus critic GPUs if enabled). | `None` | Type: int | Miles Native |
| `--rollout-num-gpus-per-engine` | Number of GPUs per inference engine, same as `tp_size` in SGLang. For multi-node serving, this should be the total GPU count / `tp_size` for each SGLang instance. | `1` | Type: int | Miles Native |
| `--num-gpus-per-node` | Total GPUs per node on the physical machine. This informs the Ray scheduler of the hardware capacity. In **Colocate mode**, it is required that the machine has fewer than 8 GPUs to calculate correct VRAM offsets. In **Disaggregated mode**, it ensures SGLang engines are distributed correctly across nodes without exceeding per-node GPU limits. | `8` | Type: int | Miles Native |
| `--colocate` | Deploy training and rollout on the same GPUs. This mode automatically enables `--offload-train` and `--offload-rollout` to facilitate weight-swapping between the training actor and inference engine. **Note:** The offload parameters are currently only used for AMD GPUs and will be removed soon. **Memory Tip:** When colocating, it is highly recommended to set `--sglang-mem-fraction-static` to **0.8** (especially on **NVIDIA Blackwell B200/B300** GPUs). This leaves sufficient VRAM (~20%) for Megatron to initialize its structures before the first weight offload to CPU occurs. On GB200/GB300, values up to 0.75 are safer for long-running jobs to prevent potential OOMs. #TODO: Verify optimal fraction for Blackwell in production | `False` | bool flag (set to enable) | Miles Native |
| `--prefill-num-servers` | Number of dedicated prefill servers for PD disaggregation. | `None` | Type: int | Miles Native |
| `--distributed-backend` | Backend for distributed communication. | `nccl` | `nccl`, `gloo` | Megatron-LM (Reset by Miles) |
| `--distributed-timeout-minutes` | Timeout for distributed operations in minutes. | `10` | Type: int | Megatron-LM (Reset by Miles) |

Note that most use cases do not need to consider offload parameters, including `--offload-rollout, --no-offload-rollout, --offload-train, --no-offload-train`.

## Training Backend

Arguments for configuring the training engine (Megatron or FSDP).

| Argument | Description | Default | Options | Source |
| :--- | :--- | :--- | :--- | :--- |
| `--train-backend` | The backend for training. Highly suggest Megatron for numerical stability and efficiency. | `"megatron"` | `megatron`, `fsdp` | Miles Native |
| `--qkv-format` | Whether to pack variable-length sequences into the token dimension for training. `thd` (T-H-D, a.k.a. varlen / packed sequence) concatenates sequences and uses `cu_seqlens` to avoid padding; it is the default and is usually faster by reducing padding overhead. `bshd` (B-S-H-D) uses fixed-shape padded batches; use it for newer models with novel attention architectures (e.g., sparse attention, attention sink) where the training backend does not support `thd`. | `"thd"` | `thd`, `bshd` | Miles Native |
| `--optimizer` | Optimizer type. | `adam` | `adam`, `sgd` | Megatron-LM & FSDP |
| `--lr` | Learning rate for the Actor. | `1e-6` | Type: float | Megatron-LM (Reset by Miles) & FSDP |
| `--lr-warmup-init` | Initial learning rate for warmup. | `0.0` | Type: float | Megatron-LM & FSDP |
| `--min-lr` | Minimum learning rate after decay. | `0.0` | Type: float | Megatron-LM & FSDP |
| `--lr-decay-style` | Learning rate decay style. | `constant`(FSDP), `linear`(Megatron) | Type: str | Megatron-LM & FSDP |
| `--lr-warmup-iters` | Number of iterations for warmup. | `0` | Type: int | Megatron-LM & FSDP |
| `--lr-decay-iters` | Number of iterations for learning rate decay. | `None` | Type: int | Megatron-LM & FSDP |
| `--lr-warmup-fraction` | Fraction of total steps to warmup. | `None` | Type: float | Megatron-LM & FSDP |
| `--adam-beta1` | Beta1 for Adam optimizer. | `0.9` | Type: float | Megatron-LM & FSDP |
| `--adam-beta2` | Beta2 for Adam optimizer. | `0.95` | Type: float | Megatron-LM & FSDP |
| `--adam-eps` | Epsilon for Adam optimizer. | `1e-8` | Type: float | Megatron-LM & FSDP |
| `--true-on-policy-mode` | Strictly align SGLang's log probs and training engine's log probs to bit-wise equal. This parameter is only used for FSDP right now. [Ref](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/mismatch/blog-en.md#truly-on-policy-training) | `False` | bool flag (set to enable) | Miles Native |
| `--train-env-vars` | Extra environment variables for training process, e.g., PyTorch memory management ones. | `{}` | Type: JSON / Dict | Miles Native |
| `--train-memory-margin-bytes` | Reserved memory margin for training in bytes. Defaults to 1GB. | `1073741824` | Type: int | Miles Native |
| `--disable-weights-backuper` | Applies to `megatron` training backend only. Disables the system that backs up model weights (Actor, Ref, Old Actor) to CPU RAM. Disabling saves significant host memory but prevents features that rely on weight-swapping, such as computing the KL-divergence against a reference model. **Note**: do not set `--ref-load` and `--keep-old-actor` if disable weights backuper. | `False` | bool flag (set to disable) | Miles Native |
| `--custom-model-provider-path` | Path to a custom function that replaces the default model provider. [Ref](../get_started/customization.md#20-model-provider---custom-model-provider-path) | `None` | Type: str | Miles Native |
| `--recompute-loss-function` | Enable recomputing the loss function to save memory during training. | `False` | bool flag (set to enable) | Miles Native |
| `--log-probs-chunk-size` | Specifies the chunk size for logprobs computation to reduce peak memory usage. Processing logits in smaller batches, it prevents CUDA OOM errors during long-context prefilling or re-computation. Set to `-1` to disable chunking. [Ref](https://github.com/sgl-project/sglang/pull/6318) | `-1` | Type: int | Miles Native |
| `--keep-old-actor` | Maintains a "Model Queue" (Actor, Rollout Actor, Old Actor) to ensure importance sampling ratios are calculated against the exact policy version that generated the data. Essential for asynchronous RL where training and inference are decoupled, preventing mathematical incorrectness due to model staleness. It consumes additional Host Memory (extra ~1x model size for `update_weights_interval > 1` or 2x for `update_weights_interval == 1`) depending on update interval. | `False` | bool flag (set to enable) | Miles Native |
| `--update-weight-buffer-size` | Buffer size for updating weights, in bytes. [Ref](https://hebiao064.github.io/rl-weight-sync#42-optimizing-sglang-server-calls-with-tensor-bucketing-from-50s-to-30s) | `536870912` | Type: int | Miles Native |
| `--update-weights-interval` | Interval (in rollout rounds) for syncing weights to inference engines. Set to `>1` for async RL. | `1` | Type: int | Miles Native |
| `--fp16` | Enable FP16 mixed precision. | `False` | bool flag (set to enable) | Megatron-LM & FSDP |
| `--context-parallel-size` | Size of context parallelism. | `1` | Type: int | Megatron-LM & FSDP |
| `--deterministic-mode` | Enable deterministic mode for reproducibility. [Ref](https://lmsys.org/blog/2025-09-22-sglang-deterministic/) | `False` | bool flag (set to enable) | Megatron-LM & FSDP |

## Rollout Management

Arguments for configuring the rollout (inference) process and custom rollout logic.

| Argument | Description | Default | Options | Source |
| :--- | :--- | :--- | :--- | :--- |
| `--hf-checkpoint` | Path to the HuggingFace checkpoint used to initialize SGLang and provide the tokenizer. | `None` | Type: str | Miles Native |
| `--model-name` | The name of the model that is used to convert the Megatron weights into HuggingFace format. If not set, we will use `type(AutoConfig.from_pretrained(args.hf_checkpoint)).__name__.lower()` as `model_name`. Providing this argument can also help in cases where transformers cannot find certain models. | `None` | Type: str | Miles Native |
| `--rollout-function-path` | Path to the rollout generation function. Use this to inject custom logic (e.g., for multi-turn or tool use). [Ref](../get_started/customization.md#1-rollout-function---rollout-function-path) | `miles.rollout.sglang_rollout.generate_rollout` (or `miles.rollout.inference_rollout.inference_rollout_common.InferenceRolloutFn` when `MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1`) | Type: str | Miles Native |
| `--rollout-temperature` | Sampling temperature for the inference engine during rollout. | `1.0` | Type: float | Miles Native |
| `--rollout-top-p` | Top-p (nucleus) sampling threshold during rollout. | `1.0` | Type: float | Miles Native |
| `--rollout-top-k` | Top-k sampling threshold during rollout. `-1` means disabled. | `-1` | Type: int | Miles Native |
| `--rollout-max-context-len` | The maximum context size for the inference engine during rollout. It should not exceed the `max_position_embeddings` in the HuggingFace model's `config.json`. **Note:** This acts as a hard cap for the total tokens (Prompt + Response). | `None` | Type: int | Miles Native |
| `--rollout-max-prompt-len` | Maximum length of the prompt. Longer prompts are filtered during dataset initialization. This is not recommended if the dataset is large. **Note:** Defaults to `rollout-max-context-len - 1` if not set, ensuring at least one token can be generated. | `None` | Type: int | Miles Native |
| `--rollout-max-response-len` | Maximum length of the response (`max_tokens` in SGLang). **Note:** Generation will stop when either this limit is reached or the total session length hits `rollout-max-context-len`. | `None` | Type: int | Miles Native |
| `--rollout-skip-special-tokens` | Skip special tokens (e.g., `<\|im_end\|>`, `<\|endoftext\|>`) in the decoded response string. **Critical for Multi-Turn RL:** Ensures that when a response is appended to the conversation history for the next turn, it doesn't include terminal special tokens that would interfere with chat template formatting or cause early termination in subsequent turns. | `False` | bool flag (set to enable) | Miles Native |
| `--rollout-stop` | A list of strings that trigger termination of generation if they appear in the output (e.g., `"\nUser:"`). | `None` | Type: List[str] | Miles Native |
| `--rollout-stop-token-ids` | A list of numerical token IDs that trigger termination. This is the token-level equivalent of `--rollout-stop` and is preferred for special control tokens that are difficult to input as strings. | `None` | Type: List[int] | Miles Native |
| `--rollout-shuffle` | Shuffle the prompts during rollout. | `False` | bool flag (set to enable) | Miles Native |
| `--rollout-seed` | Seed for the random number generator during rollout (used for shuffling and sampling). | `42` | Type: int | Miles Native |
| `--rollout-external` | Use external SGLang instances instead of launching them inside the framework. | `False` | bool flag (set to enable) | Miles Native |
| `--rollout-external-engine-addrs` | Addresses and ports of the external engines. | `None` | Type: List[str] | Miles Native |
| `--custom-generate-function-path` | Path to override only the `generate` step within the default rollout function. If your custom `generate` returns `list[Sample]` (multi-sample), make sure your rollout pipeline can handle it; the default rollout expects a flat `list[Sample]` of length `--n-samples-per-prompt` for each prompt group. [Ref](../get_started/customization.md#2-custom-generate-function---custom-generate-function-path) | `None` | Type: str | Miles Native |
| `--custom-rollout-log-function-path` | Path to a custom function for logging training rollout data. [Ref](../get_started/customization.md#14-logging-functions) | `None` | Type: str | Miles Native |
| `--custom-eval-rollout-log-function-path` | Path to a custom function for logging evaluation rollout data. [Ref](../get_started/customization.md#14-logging-functions) | `None` | Type: str | Miles Native |
| `--rollout-data-postprocess-path` | Path to a function called after all rollout data (including log probs) is ready. [Ref](../get_started/customization.md#8-rollout-data-postprocess---rollout-data-postprocess-path) | `None` | Type: str | Miles Native |

## Sampling and Filtering

Arguments for sampling strategies and data filtering during rollout and buffer management.

| Argument | Description | Default | Options | Source |
| :--- | :--- | :--- | :--- | :--- |
| `--over-sampling-batch-size` | Number of prompts requested in each **oversampling** round when **dynamic sampling** is enabled. Miles samples `over_sampling_batch_size` prompts, generates `--n-samples-per-prompt` responses per prompt asynchronously, and then keeps/discards each prompt group via `--dynamic-sampling-filter-path`. If filtering is strict and the remaining accepted batch size drops below the target `--rollout-batch-size`, Miles automatically triggers another oversampling round of the same size. If unset, defaults to `--rollout-batch-size`. See [Dynamic Sampling](../get_started/quick_start.md#dynamic-sampling). | `None` | Type: int | Miles Native |
| `--dynamic-sampling-filter-path` | Path to the filter function for dynamic sampling. [Ref](../get_started/customization.md#4-dynamic-sampling-filter---dynamic-sampling-filter-path) | `None` | Type: str | Miles Native |
| `--partial-rollout` | Enable partial rollout for **dynamic sampling**: cache partially generated (aborted/unfinished) samples and resume generation in later rollout steps, reducing wasted compute for long responses. Cached samples are stored in the rollout buffer and can be prioritized/selected via `--buffer-filter-path` (default FIFO behavior). See [Partial Rollout](../get_started/quick_start.md#partial-rollout). | `False` | bool flag (set to enable) | Miles Native |
| `--mask-offpolicy-in-partial-rollout` | When using partial rollout, mask the previously generated (cached) response tokens so they do not contribute to the loss; only tokens generated after resuming are used for training. This helps avoid training on a cached prefix produced by an older policy version. See [Partial Rollout](../get_started/quick_start.md#partial-rollout). | `False` | bool flag (set to enable) | Miles Native |
| `--buffer-filter-path` | Path to the function to filter or sort samples in the rollout buffer before training. [Ref](../get_started/customization.md#5-buffer-filter---buffer-filter-path) | `None` | Type: str | Miles Native |
| `--rollout-sample-filter-path` | Path to the function that marks individual samples to be excluded from loss calculation. [Ref](../get_started/customization.md#6-rollout-sample-filter---rollout-sample-filter-path) | `None` | Type: str | Miles Native |
| `--rollout-all-samples-process-path` | Path to the function to process all samples (including filtered ones) after rollout. [Ref](../get_started/customization.md#7-rollout-all-samples-process---rollout-all-samples-process-path) | `None` | Type: str | Miles Native |

## Data Arguments

Arguments for dataset configuration, prompt mapping, and training batch sizes.

| Argument | Description | Default | Options | Source |
| :--- | :--- | :--- | :--- | :--- |
| `--prompt-data` | Path to the prompt dataset (JSONL format), and each line should contain `--input-key` and `--label-key`, which will be used as the prompt and the label, respectively. | `None` | Type: str | Miles Native |
| `--disable-rollout-global-dataset` | Disable the global dataset for rollout. By default, Miles loads `--prompt-data` into a global dataset and samples from it for rollout. Setting this flag turns off this behavior. Use this flag only when providing a custom `--rollout-function-path` (and usually a custom `--data-source-path`) that handles data loading independently. | `False` | bool flag (set to disable) | Miles Native |
| `--data-source-path` | Path to a custom Python class for the rollout data source. [Ref](../get_started/customization.md#15-data-source---data-source-path) | `miles.rollout.data_source.RolloutDataSourceWithBuffer` | Type: str | Miles Native |
| `--input-key` | Key in the JSONL data representing the user input/prompt. | `"input"` | Type: str | Miles Native |
| `--label-key` | Key in the JSONL data representing the label/ground truth. | `None` | Type: str | Miles Native |
| `--metadata-key` | When adding tools during `apply_chat_template`, provide the key for the tools to the prompt dataset. | `"metadata"` | Type: str | Miles Native |
| `--multimodal-keys` | JSON string for multimodal data mapping media types to data keys. Example: `'{"image": "image_file"}'` | `None` | Type: str | Miles Native |
| `--tool-key` | JSON key for tool definitions in the prompt dataset (used when applying chat templates). | `"tools"` | Type: str | Miles Native |
| `--apply-chat-template` | Whether to apply the chat template to the input prompt. The input should be the same structure as an OpenAI message, e.g., `[{'role': 'user', 'content': 'blabla'}]`. | `False` | bool flag (set to enable) | Miles Native |
| `--apply-chat-template-kwargs` | Extra arguments for the chat template processing (JSON string). | `"{}"` | Type: str | Miles Native |
| `--num-rollout` | Number of rollout steps. If not set, Miles will calculate the number of rollout steps from the dataset size. **Note:** This value will be overwritten if `--num-epoch` is also set. | `None` | Type: int | Miles Native |
| `--num-epoch` | Number of epochs for the training. If set, `num_rollout` is calculated as `(num_epoch * dataset_size) // rollout_batch_size`. **Note:** This argument takes precedence and will overwrite `--num-rollout` if both are specified. | `None` | Type: int | Miles Native |
| `--rollout-batch-size` | Number of prompts per rollout batch. The total data returned should be `rollout_batch_size` * `n_samples_per_prompt`. | Required | Type: int | Miles Native |
| `--n-samples-per-prompt` | Number of responses to generate for each prompt, e.g., the group size of GRPO. The default rollout pipeline expects each prompt group to contain exactly `n_samples_per_prompt` samples. | `1` | Type: int | Miles Native |
| `--global-batch-size` | Total samples per optimizer step. Automatically calculated or **overridden** if `num_steps_per_rollout` is set. | `None` | Type: int | Megatron-LM (Reset by Miles) |
| `--num-steps-per-rollout` | The number of training steps to perform using the data collected in a single rollout round. Setting this to `n` means the policy model will be updated `n` times using the same batch of rollout data. Miles ensures that `(rollout-batch-size * n-samples-per-prompt) = (global-batch-size * num-steps-per-rollout)`. If this value is not provided, you have to set `--global-batch-size` explicitly. If both are provided, `--num-steps-per-rollout`  will **override** the global batch size with `num_steps_per_rollout = (rollout_batch_size * n_samples_per_prompt) // num_steps_per_rollout`. | `None` | Type: int | Miles Native |
| `--use-dynamic-batch-size` | Dynamically packs variable-length samples into micro-batches to maximize GPU utilization, ensuring the total token count per batch does not exceed `--max-tokens-per-gpu`. For example, with a 300-token limit, samples of lengths 100, 200, and 300 would be packed into two batches: `[100, 200]` and `[300]`. **Note:** Miles ensures that enabling this optimization does not affect the mathematical correctness of per-sample or per-token loss calculation. It is **strongly recommended** to enable this for maximum efficiency. **Compatibility:** only supported when `--qkv-format` is `thd` (does not work for `bshd`). | `False` | bool flag (set to enable) | Miles Native |
| `--max-tokens-per-gpu` | The maximum number of tokens (Prompt + Response combined) per GPU for dynamic batch size. This parameter defines the total sequence length budget for packing samples into micro-batches during training. Note that when enabling context parallel (CP), the effective capacity is shared, so the value should be approximately `(Total_Sequence_Length) // cp_size`. | `None` | Type: int | Miles Native |
| `--log-probs-max-tokens-per-gpu` | The maximum number of tokens per GPU for calculating log probs. This is used to calculate the log probs of the responses during rollout, and should be set to a larger value than `max_tokens_per_gpu` if you want better performance. | `None` | Type: int | Miles Native |
| `--balance-data` | Repartition each rollout batch so each data-parallel rank gets a similar total token count via the Karmarkar-Karp method. It may be beneficial for training speed, but changes per-rank sample grouping and adds a small CPU scheduling overhead. | `False` | bool flag (set to enable) | Miles Native |
| `--data-pad-size-multiplier` | Multiplier used to calculate the sequence padding boundary. Miles rounds sequence lengths up to a multiple of `tensor_parallel_size * data_pad_size_multiplier`. This optimization ensures that matrix dimensions are aligned with NVIDIA Tensor Core requirements, maximizing throughput and reducing VRAM fragmentation. **Note:** better not change this; values `<128` may trigger accuracy loss under `--qkv-format thd` when `TP>=4`. | `128` | Type: int | Miles Native |
| `--micro-batch-size` | Micro batch size per GPU. Ignored when `--use-dynamic-batch-size` is enabled. Works for both `--qkv-format thd` and `--qkv-format bshd` (and is required for `bshd` because dynamic batch size is unsupported). | `1` | Type: int | Megatron-LM (Reset by Miles) |

## Evaluation Arguments

Arguments for configuring the evaluation process during training.

| Argument | Description | Default | Options | Source |
| :--- | :--- | :--- | :--- | :--- |
| `--eval-interval` | Interval (in rollout steps) between evaluations. | `None` | Type: int | Megatron-LM (Reset by Miles) |
| `--eval-prompt-data` | List of name and path pairs for evaluation datasets (e.g., `aime /path/to/aime.jsonl`). | `None` | Type: List[str] | Miles Native |
| `--eval-config` | Path to an OmegaConf YAML/JSON file describing evaluation datasets (overrides `--eval-prompt-data`). | `None` | Type: str | Miles Native |
| `--skip-eval-before-train` | Skip the evaluation step before training starts. | `False` | bool flag (set to enable) | Miles Native |
| `--n-samples-per-eval-prompt` | Number of responses for each prompt in generation. | `1` | Type: int | Miles Native |
| `--eval-temperature` | Temperature for evaluation (defaults to rollout temperature if not set). | `None` | Type: float | Miles Native |
| `--eval-top-p` | Top-p sampling threshold for evaluation (defaults to rollout top-p if not set). | `None` | Type: float | Miles Native |
| `--eval-top-k` | Top-k sampling threshold for evaluation (defaults to rollout top-k if not set). | `None` | Type: int | Miles Native |
| `--eval-max-response-len` | Maximum response length for evaluation (defaults to rollout max response length if not set). | `None` | Type: int | Miles Native |
| `--eval-max-prompt-len` | Maximum prompt length for evaluation. | `None` | Type: int | Miles Native |
| `--eval-min-new-tokens` | Minimum tokens to generate for evaluation responses (Not used). | `None` | Type: int | Miles Native |
| `--eval-max-context-len` | Maximum context length for evaluation (defaults to rollout max context length if not set). | `None` | Type: int | Miles Native |
| `--eval-function-path` | Path to a custom evaluation function. [Ref](../get_started/customization.md#16-evaluation-function---eval-function-path) | `None` | Type: str | Miles Native |
| `--eval-input-key` | JSON key for input text in evaluation datasets. | `None` | Type: str | Miles Native |
| `--eval-label-key` | JSON key for ground truth labels in evaluation datasets. | `None` | Type: str | Miles Native |
| `--eval-tool-key` | JSON key for tool definitions in evaluation datasets. | `None` | Type: str | Miles Native |

## Checkpointing and Resuming

Arguments for saving and loading model states.

| Argument | Description | Default | Options | Source |
| :--- | :--- | :--- | :--- | :--- |
| `--load` | Path to the training model checkpoint to load. | `None` | Type: str | Megatron-LM (Reset by Miles) |
| `--save` | Path to save checkpoints. | `None` | Type: str | Megatron-LM (Reset by Miles) |
| `--save-interval` | Interval (in rollout steps) to save checkpoints. Requires `--save` to be set. | `None` | Type: int | Megatron-LM (Reset by Miles) |
| `--async-save` | Enable asynchronous checkpoint saving (Megatron backend only). | `False` | bool flag (set to enable) | Megatron-LM (Reset by Miles) |
| `--save-hf` | Path to save the model in HuggingFace format when using Megatron backend. The model will be saved to `save_hf.format(rollout_id)`. | `None` | Type: str | Miles Native |
| `--no-save-optim` | If set, optimizer state is not saved with checkpoints to reduce size, but prevents resumption of training. | `False` | bool flag (set to enable) | Megatron-LM (Reset by Miles) |
| `--ref-load` | Path to the reference model checkpoint. Used as an initial checkpoint if `--load` is not set. | `None` | Type: str | Miles Native |
| `--ref-ckpt-step` | The checkpoint step for the reference model. | `None` | Type: int | Miles Native |
| `--critic-load` | Checkpoint to load for the critic model. | value of `--load` | Type: str | Miles Native |
| `--critic-save` | Path to save the critic model. | `None` | Type: str | Miles Native |
| `--start-rollout-id` | The starting rollout step. If not set, it is inferred from the --load checkpoint when resuming training. Otherwise, if training is not continuous, Miles will start training from scratch | `None` | Type: int | Miles Native |

---

## Algorithm and RL Arguments

Arguments for reinforcement learning algorithms and loss calculation.

| Argument | Description | Default | Options | Source |
| :--- | :--- | :--- | :--- | :--- |
| `--advantage-estimator` | Advantage estimator to use. | `"grpo"` | `grpo`, `gspo`, `ppo`, `reinforce_plus_plus`, `reinforce_plus_plus_baseline`, `on_policy_distillation` | Miles Native |
| `--loss-type` | Type of loss function to use. | `"policy_loss"` | `policy_loss`, `sft_loss`, `custom_loss` | Miles Native |
| `--custom-loss-function-path` | Path to a custom loss calculation function (requires `--loss-type custom_loss`). [Ref](../get_started/customization.md#9-custom-loss-function---custom-loss-function-path) | `None` | Type: str | Miles Native |
| `--critic-lr` | Learning rate for the Critic. Defaults to `--lr`. | `None` | Type: float | Miles Native |
| `--critic-lr-warmup-iters` | Number of iterations for Critic learning rate linear warmup. | `0` | Type: int | Miles Native |
| `--num-critic-only-steps` | Number of initial steps dedicated to training only the Critic. | `0` | Type: int | Miles Native |
| `--eps-clip` | PPO clip range. | `0.2` | Type: float | Miles Native |
| `--eps-clip-high` | PPO clip upper range (defaults to `--eps-clip` if not set). | `None` | Type: float | Miles Native |
| `--eps-clip-c` | Lower bound for [Dual-clip PPO](https://arxiv.org/pdf/1912.09729). | `None` | Type: float | Miles Native |
| `--value-clip` | Clip range for value loss. | `0.2` | Type: float | Miles Native |
| `--kl-coef` | KL penalty coefficient for reward shaping. This is applied to the reward signal before advantage calculation for PPO and REINFORCE-style estimator. | `0.00` | Type: float | Miles Native |
| `--use-kl-loss` | Enable KL loss term in the final objective (as in GRPO). | `False` | bool flag (set to enable) | Miles Native |
| `--kl-loss-coef` | Weight of the KL loss term in the final objective. | `0.0` | Type: float | Miles Native |
| `--kl-loss-type` | Selection of the KL loss implementation. See [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html) for more details. | `k1` | `k1`, `k2`, `k3`, `low_var_kl` | Miles Native |
| `--use-unbiased-kl` | Apply Importance Sampling (IS) correction to the KL estimator. Reduces bias from distribution shift. | `False` | bool flag (set to enable) | Miles Native |
| `--entropy-coef` | Coefficient for entropy regularization term. Penalizes low entropy to encourage exploration and prevent premature convergence. | `0.0` | Type: float | Miles Native |
| `--gamma` | Discount factor for future rewards. Used in PPO (GAE) and REINFORCE++. | `1.0` | Type: float | Miles Native |
| `--lambd` | PPO GAE lambda. | `1.0` | Type: float | Miles Native |
| `--normalize-advantages` | Performs distributed masked whitening of advantages. Normalization statistics are computed globally across the Data-Parallel group, ignoring padding tokens. | `False` | bool flag (set to enable) | Miles Native |
| `--disable-compute-advantages-and-returns` | Disables the calculation of advantages and returns. This is typically used for SFT or custom loss functions where value estimation is not required. | `False` | bool flag (set to enable) | Miles Native |
| `--use-tis` | Enable Token-level Importance Sampling (TIS) from this [blog](https://fengyao.notion.site/off-policy-rl#279721e3f6c48092bbe2fcfe0e9c6b33). | `False` | bool (set to enable) | Miles Native |
| `--tis-clip` | Clipping threshold C for importance sampling ratios to control variance. | `2.0` | Type: float | Miles Native |
| `--tis-clip-low` | Lower bound clipping threshold C for importance sampling ratios to control variance. | `0.0` | Type: float | Miles Native |
| `--custom-tis-function-path` | Path to a custom TIS or MIS function. [Ref](../get_started/customization.md#10-custom-tisrs-function---custom-tis-function-path) | `None` | Type: str | Miles Native |
| `--custom-pg-loss-reducer-function-path` | Custom reducer function for policy gradient loss. [Ref](../get_started/customization.md#11-custom-pg-loss-reducer---custom-pg-loss-reducer-function-path) | `None` | Type: str | Miles Native |
| `--use-routing-replay` | Enable R2 (Routing Replay) for MoE: record expert routing decisions during forward and replay them during backward. [Paper](https://arxiv.org/abs/2507.18071) **Note:** automatically set to `True` when `--use-rollout-routing-replay` is enabled. | `False` | bool flag (set to enable) | Miles Native |
| `--use-rollout-routing-replay` | Enable R3 (Rollout Routing Replay) for MoE: record expert routing decisions during rollout and replay them during training. **Requires `--use-miles-router`**. [Paper](https://arxiv.org/abs/2510.11370) [Ref](miles-router.md#22-rollout-routing-replay-r3-for-moe) | `False` | bool flag (set to enable) | Miles Native |
| `--use-opsm` | Enable Off-Policy Sequence Masking (OPSM). Filters sequences that have **BOTH** negative advantages (bad results) AND high KL divergence (stale data). This stabilizes training by preventing updates from unreliable, highly off-policy samples. | `False` | bool flag (set to enable) | Miles Native |
| `--opsm-delta` | The threshold for Off-Policy Sequence Masking (OPSM). | `1e-4` | Type: float | Miles Native |
| `--get-mismatch-metrics` | Calculate mismatch metrics. If it is set, you need to provide a custom TIS function via `--custom-tis-function-path`. | `False` | bool flag (set to enable) | Miles Native |
| `--ref-update-interval` | Interval (in rollout steps) to update ref model from actor. If `None`, ref model is not updated. | `None` | Type: int | Miles Native |
| `--reset-optimizer-states` | Resets the optimizer state after each rollout round. This clears the optimization history, which can improve stability or satisfy specific experimental requirements. | `False` | bool flag (set to enable) | Miles Native |
| `--disable-grpo-std-normalization` | Disable standard deviation normalization for GRPO. From [Dr.GRPO](https://arxiv.org/pdf/2503.20783) | `False` | bool flag (set to enable) | Miles Native |
| `--disable-rewards-normalization` | Disable the default group-wise reward normalization for GRPO, GSPO, and REINFORCE++. This effectively skips the baseline subtraction step. | `False` | bool flag (set to enable) | Miles Native |
| `--use-rollout-entropy` | Enable entropy calculation when calculating the logprobs from actor and reference model. This is useful for implementing custom entropy-based loss masking. | `False` | bool flag (set to enable) | Miles Native |
| `--use-rollout-logprobs` | Use rollout logprobs as the old-policy logprobs when computing importance sampling ratios / PPO-style KL in GRPO/GSPO/PPO. If not set, Miles recomputes old-policy logprobs with the training actor (e.g., `old_actor` or `actor`, depending on configuration). If `--get-mismatch-metrics` is set, the log probs will still be recomputed by the training engine (one more forward pass will be applied). | `False` | bool flag (set to enable) | Miles Native |
| `--calculate-per-token-loss` | Calculate loss on a per-token basis. | `False` | bool flag (set to enable) | Megatron-LM (Reset by Miles) |
| `--seed` | Random seed for the training process. **Also passed to SGLang servers as `random_seed`** (Miles uses `seed + engine_rank` so each engine has a distinct but reproducible seed). | `1234` | Type: int | Megatron-LM (Reset by Miles) |
| `--clip-grad` | Maximum gradient norm for gradient clipping. | `1.0` | Type: float | Megatron-LM (Reset by Miles) |

---

## Logging and Monitoring

Arguments for WandB, Tensorboard, and general logging.

| Argument | Description | Default | Options | Source |
| :--- | :--- | :--- | :--- | :--- |
| `--use-wandb` | Enable WandB logging. | `False` | bool flag (set to enable) | Miles Native |
| `--wandb-mode` | WandB operating mode. Overrides `WANDB_MODE`. | `None` | `online`, `offline`, `disabled` | Miles Native |
| `--wandb-project` | WandB project name. | `None` | Type: str | Megatron-LM (Reset by Miles) |
| `--wandb-group` | WandB group name. | `None` | Type: str | Miles Native |
| `--wandb-team` | WandB team name. | `None` | Type: str | Miles Native |
| `--wandb-host` | WandB host address. | `None` | Type: str | Miles Native |
| `--wandb-key` | WandB API key. | `None` | Type: str | Miles Native |
| `--wandb-run-id` | Specific WandB run ID to resume. | `None` | Type: str | Miles Native |
| `--wandb-dir` | Directory to store WandB logs. Default is `./wandb` in current directory. | `None` | Type: str | Miles Native |
| `--disable-wandb-random-suffix` | Disable adding a random suffix to the WandB run name. By default, we will add a random 6 length string with characters to the run name. | `False` | bool flag (set to enable) | Miles Native |
| `--wandb-always-use-train-step` | Use training steps instead of rollout steps for the x-axis. | `False` | bool flag (set to enable) | Miles Native |
| `--use-tensorboard` | Enable Tensorboard logging. | `False` | bool flag (set to enable) | Miles Native |
| `--tb-project-name` | Tensorboard project directory. | `None` | Type: str | Miles Native |
| `--tb-experiment-name` | Tensorboard experiment name. | `None` | Type: str | Miles Native |
| `--tensorboard-dir` | Directory to store Tensorboard logs. | `None` | Type: str | Miles Native |
| `--log-multi-turn` | Log detailed information for multi-turn conversations. | `False` | bool flag (set to enable) | Miles Native |
| `--log-passrate` | Enable logging of `pass@n` metrics. | `False` | bool flag (set to enable) | Miles Native |
| `--log-correct-samples` | Explicitly log metrics for correct samples. | `False` | bool flag (set to enable) | Miles Native |
| `--log-reward-category` | Log reward-category statistics (e.g., why the reward function marked a failure). Use this argument to specify the key in the reward dict. | `None` | Type: str | Miles Native |

---

## Fault Tolerance

Arguments for handling server failures during rollout.

| Argument | Description | Default | Options | Source |
| :--- | :--- | :--- | :--- | :--- |
| `--use-fault-tolerance` | Enable fault tolerance for rollout engines. Periodically sends `/health_generate` heartbeats. | `False` | bool flag (set to enable) | Miles Native |
| `--rollout-health-check-interval` | Interval in seconds between rollout engine `/health_generate` checks during generate/eval. | `30.0` | Type: float | Miles Native |
| `--rollout-health-check-timeout` | Timeout in seconds to wait for a rollout engine `/health_generate` response before killing it. | `30.0` | Type: float | Miles Native |
| `--rollout-health-check-first-wait` | Initial grace period (in seconds) before starting health checks. This allows time for model compilation and initialization. Increase this value significantly when using deepgemm. | `0.0` | Type: float | Miles Native |

---

## Miles Router

Arguments for the specialized Miles text-based router.

| Argument | Description | Default | Options | Source |
| :--- | :--- | :--- | :--- | :--- |
| `--use-miles-router` | Use Miles Router (FastAPI passthrough proxy) instead of SGLang Model Gateway for rollout routing. Required for features that depend on preserving extra rollout metadata (e.g., R3). [Ref](miles-router.md) | `False` | bool flag (set to enable) | Miles Native |
| `--miles-router-middleware-paths` | Paths to custom MilesRouter middleware functions. [Ref](../get_started/customization.md#18-miles-router-middleware---miles-router-middleware-paths) | `""` | Type: List[str] | Miles Native |
| `--miles-router-timeout` | Timeout for router HTTP requests in seconds. | `None` | Type: float | Miles Native |
| `--miles-router-max-connections` | Max connections for MilesRouter HTTP client. | `None` | Type: int | Miles Native |
| `--miles-router-health-check-failure-threshold` | Number of consecutive failures before marking a worker as unhealthy. | `3` | Type: int | Miles Native |

---

## Reward Model Arguments

Arguments for configuring reward signals and post-processing.

| Argument | Description | Default | Options | Source |
| :--- | :--- | :--- | :--- | :--- |
| `--rm-type` | Built-in reward model selection. | `None` | `remote_rm`, `deepscaler`, `dapo`, `math`, `f1`, `gpqa`, `ifbench`, `random` | Miles Native |
| `--rm-url` | URL for the reward model service (used with `--rm-type remote_rm`). | `None` | Type: str | Miles Native |
| `--reward-key` | JSON key to extract the numerical reward from a returned dictionary if reward model returns a dict instead of a value. | `None` | Type: str | Miles Native |
| `--eval-reward-key` | Evaluation variant for `--reward-key`. | `None` | Type: str | Miles Native |
| `--custom-rm-path` | Path to a custom Python reward function. [Ref](../get_started/customization.md#3-reward-model---custom-rm-path) | `None` | Type: str | Miles Native |
| `--group-rm` | Defer reward computation to process the entire group of samples (per-prompt) at once. Essential for comparative/ranking reward models and improves throughput. **Not supported in eval**. | `False` | bool flag (set to enable) | Miles Native |
| `--custom-reward-post-process-path` | Path to a custom reward post-processor. [Ref](../get_started/customization.md#12-reward-post-processing---custom-reward-post-process-path) | `None` | Type: str | Miles Native |
| `--custom-convert-samples-to-train-data-path` | Path to a custom data format converter. [Ref](../get_started/customization.md#13-samples-to-train-data-conversion---custom-convert-samples-to-train-data-path) | `None` | Type: str | Miles Native |

---

## Rollout Buffer Management

Arguments for managing the rollout data buffer.

| Argument | Description | Default | Options | Source |
| :--- | :--- | :--- | :--- | :--- |
| `--rollout-buffer-url` | URL for the rollout buffer service. | `None` | Type: str | Miles Native |
| `--fetch-trajectory-retry-times` | Number of times to retry fetching trajectory, -1 means unlimited retry. | `-1` | Type: int | Miles Native |
| `--min-batch-collection-ratio` | Minimum batch collection ratio before proceeding. | `1.0` | Type: float | Miles Native |
| `--disable-rollout-trim-samples` | Disable trim samples in rollout buffer when converting samples to train data. | `False` | bool flag (set to enable) | Miles Native |
| `--use-dynamic-global-batch-size` | Enable dynamic global batch size, disable trim samples in rollout buffer when converting samples to train data. | `False` | bool flag (set to enable) | Miles Native |
| `--rollout-task-type` | Type of task being performed. | `math` | Type: str | Miles Native |
| `--loss-mask-type` | Selection of the token masking logic. | `qwen` | `qwen`, `qwen3`, `distill_qwen` | Miles Native |

---

## Multi-Token Prediction (MTP) Arguments

Arguments for MTP-based training.

| Argument | Description | Default | Options | Source |
| :--- | :--- | :--- | :--- | :--- |
| `--enable-mtp-training` | Enable MTP layer parameter updates during training. | `False` | bool flag (set to enable) | Miles Native |
| `--mtp-num-layers` | Number of MTP layers to include. | `None` | Type: int | Megatron-LM (Reset by Miles) |
| `--mtp-loss-scaling-factor` | Scaling factor applied to the MTP loss. | `0.2` | Type: float | Megatron-LM (Reset by Miles) |

---

## SGLang Backend Arguments

Most SGLang server arguments can be passed through by adding the `--sglang-` prefix (some are intentionally skipped, e.g. `model_path`, `tp_size`, `port`, `nnodes`, `node_rank`). For a full list, refer to the [SGLang Server Arguments documentation](https://docs.sglang.io/advanced_features/server_arguments.html).

Commonly used arguments:

| Argument | Description | Default | Options | Source |
| :--- | :--- | :--- | :--- | :--- |
| `--sglang-mem-fraction-static` | Fraction of GPU memory to reserve for SGLang KV cache. | `0.9` | Type: float | SGLang |
| `--sglang-server-concurrency` | Maximum number of concurrent requests. | `512` | Type: int | SGLang |
| `--sglang-router-ip` | IP address of the SGLang router and Miles Router. | `None` | Type: str | SGLang Gateway & Miles Router |
| `--sglang-router-port` | Port of the SGLang router and Miles Router. | `None` | Type: int | SGLang Gateway & Miles Router |
| `--sglang-router-request-timeout-secs` | Timeout for requests to the SGLang router. | `14400` | Type: int | SGLang Gateway |

---

## Megatron Specific Arguments

Arguments applicable when using `--train-backend megatron`.

| Argument | Description | Default | Options | Source |
| :--- | :--- | :--- | :--- | :--- |
| `--megatron-to-hf-mode` | Method to convert Megatron weights to HuggingFace format for SGLang integration. | `raw` | `raw`, `bridge` | Miles Native |
| `--seq-length` | Megatron’s “maximum sequence length” parameter. **In miles training, this parameter has no effect in most setups**: miles uses varlen/packed samples (no truncation based on `seq_length`), forces variable sequence lengths for PP communication buffers, and uses all-to-all token dispatch for MoE. This parameter mainly matters in Megatron’s dataset pipeline. | `None` | Type: int | Megatron-LM |

---

## FSDP Specific Arguments

Arguments applicable when using `--train-backend fsdp`. **Note: The FSDP backend is still under development and experimental.**

| Argument | Description | Default | Options | Source |
| :--- | :--- | :--- | :--- | :--- |
| `--warmup-ratio` | Ratio of total steps for warmup. | `0.03` | Type: float | Miles Native |
| `--weight-decay` | Weight decay for the optimizer. | `0.0` | Type: float | Miles Native |
| `--gradient-checkpointing` | Enable gradient checkpointing. | `False` | bool flag (set to enable) | Miles Native |
| `--fsdp-cpu-offload` | Offload parameters and gradients to CPU. | `False` | bool flag (set to enable) | Miles Native |
| `--fsdp-state-dict-cpu-offload` | Offload full state dict to CPU during collection. | `False` | bool flag (set to enable) | Miles Native |
| `--fsdp-cpu-backend` | CPU backend for FSDP CPU offload. | `gloo` | `gloo`, `None` | Miles Native |
| `--attn-implementation` | Selection of the attention implementation. | `flash_attention_2` | `flash_attention_2`, `sdpa`, `eager` | Miles Native |
| `--use-pytorch-profiler` | Enable PyTorch-native profiling. | `False` | bool flag (set to enable) | Miles Native |
| `--profile-step-start` | Starting step for profiling. | `10` | Type: int | Miles Native |
| `--profile-step-end` | Ending step for profiling. | `12` | Type: int | Miles Native |
| `--lr-wsd-decay-iters` | Number of iterations for WSD decay. | `None` | Type: int | Miles Native |
| `--lr-wsd-decay-style` | Decay style for WSD. | `None` | Type: str | Miles Native |
| `--use-checkpoint-lr-scheduler` | Use the checkpoint's LR scheduler state. | `False` | bool flag (set to enable) | Miles Native |
| `--override-lr-scheduler` | Override the loaded LR scheduler state. | `False` | bool flag (set to enable) | Miles Native |
| `--wandb-run-name` | Specific run name for WandB (FSDP backend). | `None` | Type: str | Miles Native |

---

## Debug and Profiling

| Argument | Description | Default | Options | Source |
| :--- | :--- | :--- | :--- | :--- |
| `--check-weight-update-equal` | Use SGLang's weight checker to check and ensure that the loaded weight from HF checkpoint and received from Megatron are bit-wise equal. | `False` | bool flag (set to enable) | Miles Native |
| `--save-debug-rollout-data` | Path to save rollout data for offline analysis. [Ref](../developer_guide/debug.md) | `None` | Type: str | Miles Native |
| `--load-debug-rollout-data` | Path to load debug rollout data (bypasses SGLang). [Ref](../developer_guide/debug.md) | `None` | Type: str | Miles Native |
| `--load-debug-rollout-data-subsample` | Percentage of debug data to load (0.0 to 1.0). [Ref](../developer_guide/debug.md) | `None` | Type: float | Miles Native |
| `--debug-rollout-only` | Run the rollout phase only without training. [Ref](../developer_guide/debug.md) | `False` | bool flag (set to enable) | Miles Native |
| `--debug-train-only` | Run the training phase only without launching SGLang servers. [Ref](../developer_guide/debug.md) | `False` | bool flag (set to enable) | Miles Native |
| `--save-debug-train-data` | Path to save training batches for offline math debugging. | `None` | Type: str | Miles Native |
| `--dump-details` | Dump exhaustive training details for post-hoc visualization. | `None` | Type: str | Miles Native |
| `--memory-snapshot-path` | Path to save memory snapshots. | `snapshot.pickle` | Type: str | Miles Native |
| `--record-memory-history` | Record memory history for snapshots. | `False` | bool flag (set to enable) | Miles Native |
| `--memory-snapshot-dir` | Directory for PyTorch memory snapshots. | `.` | Type: str | Miles Native |
| `--memory-snapshot-num-steps` | Number of steps to record before saving snapshot. | `None` | Type: int | Miles Native |
| `--memory-recorder` | Selection of the memory recording backend. | `torch` | `torch`, `memray` | Miles Native |
| `--profile-target` | Training components to profile (accepts multiple). | `train_overall` | `train_overall`, `train_actor`, `train_log_probs` | Miles Native |

---

## Environment Variables

Miles recognizes several environment variables for advanced configuration.

| Variable | Description | Source |
| :--- | :--- | :--- |
| `MILES_EXPERIMENTAL_ROLLOUT_REFACTOR` | Set to `1` to enable the experimental rollout implementation refactor. | Miles Native |
| `ENABLE_ROUTING_REPLAY` | Internal variable used to enable MoE routing consistency checks during training. | Miles Native |
| `TENSORBOARD_DIR` | Base directory for Tensorboard logs. | Miles Native |
| `MILES_HOST_IP` | Overrides the host IP used for distributed communication. | Miles Native |
| `PYTHONPATH` | Must include the path to your `Megatron-LM` installation when using the Megatron backend. | System |
| `NCCL_SOCKET_IFNAME` | Specifies the network interface for NCCL communication (e.g., `eth0`, `bond0`). | System |
| `GLOO_SOCKET_IFNAME` | Specifies the network interface for GLOO communication. | System |
| `NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME` | Network interface for NVSHMEM bootstrap. | System |

---

## Multi-Turn and Agentic Arguments

Arguments for managing interactions and tools. Only available when `MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1` and the rollout/generate function exposes `add_arguments`.

| Argument | Description | Default | Options | Source |
| :--- | :--- | :--- | :--- | :--- |
| `--generate-max-turns` | Maximum number of turns in a conversation. | `16` | Type: int | Miles Native |
| `--generate-tool-specs-path` | Path to the tool specifications (JSON). | `None` | Type: str | Miles Native |
| `--generate-tool-call-parser` | The parser used to extract tool calls from text. | `None` | Type: str | Miles Native |
| `--generate-execute-tool-function-path` | Path to the function that executes the tool. | `None` | Type: str | Miles Native |
| `--generate-multi-samples` | Whether to generate multiple samples within one turn. | `False` | bool flag (set to enable) | Miles Native |

---

## Advanced Developer Hooks and CI

Hooks for custom logic and Continuous Integration testing flags.

| Argument | Description | Default | Options | Source |
| :--- | :--- | :--- | :--- | :--- |
| `--custom-megatron-init-path` | Path to custom Megatron initialization logic. [Ref](../get_started/customization.md#17-megatron-hooks) | `None` | Type: str | Miles Native |
| `--custom-megatron-before-log-prob-hook-path` | Hook called before calculating log probabilities. [Ref](../get_started/customization.md#17-megatron-hooks) | `None` | Type: str | Miles Native |
| `--custom-megatron-before-train-step-hook-path` | Hook called before each training step. [Ref](../get_started/customization.md#17-megatron-hooks) | `None` | Type: str | Miles Native |
| `--ci-test` | Enable Continuous Integration testing mode. | `False` | bool flag (set to enable) | Miles Native |
| `--ci-disable-kl-checker` | Disable KL divergence sanity checks in CI. | `False` | bool flag (set to enable) | Miles Native |
| `--ci-metric-checker-key` | Metric key to monitor for pass/fail in CI. | `None` | Type: str | Miles Native |
| `--ci-metric-checker-threshold` | Pass/fail threshold (minimum value) for the monitored metric. | `None` | Type: float | Miles Native |
| `--ci-save-grad-norm` | Path to save gradient norms for CI comparison. | `None` | Type: str | Miles Native |
| `--ci-load-grad-norm` | Path to load gradient norms for CI verification. | `None` | Type: str | Miles Native |

---

## Miscellaneous and System

General arguments for infrastructure and configuration overrides.

| Argument | Description | Default | Options | Source |
| :--- | :--- | :--- | :--- | :--- |
| `--http-proxy` | HTTP proxy server for remote reward model calls. | `None` | Type: str | Miles Native |
| `--use-distributed-post` | Use distributed POST requests for remote reward models. | `False` | bool flag (set to enable) | Miles Native |
| `--custom-config-path` | Path to the YAML config for custom function arguments. | `None` | Type: str | Miles Native |
| `--padded-vocab-size` | Manually specify the vocab size for padding. | `None` | Type: int | Miles Native |
