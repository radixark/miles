import argparse
import json
import logging
import os
from typing import Any

import yaml
from sglang_router.launch_router import RouterArgs
from transformers import AutoConfig

from miles.backends.sglang_utils.arguments import add_sglang_arguments
from miles.backends.sglang_utils.arguments import validate_args as sglang_validate_args
from miles.utils.environ import enable_experimental_rollout_refactor
from miles.utils.eval_config import EvalDatasetConfig, build_eval_dataset_configs, ensure_dataset_list
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import load_function

logger = logging.getLogger(__name__)


def reset_arg(parser, name, **kwargs):
    """
    Reset the default value of a Megatron argument.
    :param parser: The argument parser.
    :param name: The name of the argument to reset.
    :param default: The new default value.
    """
    for action in parser._actions:
        if name in action.option_strings:
            if "default" in kwargs:
                action.default = kwargs["default"]
            break
    else:
        parser.add_argument(name, **kwargs)


def get_miles_extra_args_provider(add_custom_arguments=None):
    def add_miles_arguments(parser):
        # Ray
        def add_cluster_arguments(parser):
            parser.add_argument(
                "--actor-num-nodes", type=int, default=1, help="Number of nodes for training the Actor."
            )
            parser.add_argument(
                "--actor-num-gpus-per-node",
                type=int,
                default=8,
                help="Number of GPUs per node for training the Actor.",
            )
            parser.add_argument(
                "--critic-num-nodes",
                type=int,
                default=None,
                help="Number of nodes for the Critic. Defaults to `--actor-num-nodes`.",
            )
            parser.add_argument(
                "--critic-num-gpus-per-node",
                type=int,
                default=None,
                help="Number of GPUs per node for the Critic. Defaults to `--actor-num-gpus-per-node`.",
            )

            parser.add_argument(
                "--rollout-num-gpus",
                type=int,
                default=None,
                help=(
                    "Total number of GPUs required for rollout (inference). In `--colocate` mode, this is ignored and set to `actor-num-gpus-per-node * actor-num-nodes` (plus critic GPUs if enabled)."
                ),
            )
            parser.add_argument(
                "--rollout-num-gpus-per-engine",
                type=int,
                default=1,
                help="Number of GPUs per inference engine, same as `tp_size` in SGLang. For multi-node serving, this should be the total GPU count / `tp_size` for each SGLang instance.",
            )
            parser.add_argument(
                "--num-gpus-per-node",
                type=int,
                default=8,
                help=(
                    "Total GPUs per node on the physical machine. This informs the Ray scheduler of the hardware capacity. In **Colocate mode**, it is required that the machine has fewer than 8 GPUs to calculate correct VRAM offsets. In **Disaggregated mode**, it ensures SGLang engines are distributed correctly across nodes without exceeding per-node GPU limits."
                ),
            )
            parser.add_argument(
                "--colocate",
                action="store_true",
                default=False,
                help=(
                    "Deploy training and rollout on the same GPUs. This mode automatically enables `--offload-train` and `--offload-rollout` to facilitate weight-swapping between the training actor and inference engine. **Note:** The offload parameters are currently only used for AMD GPUs and will be removed soon. **Memory Tip:** When colocating, it is highly recommended to set `--sglang-mem-fraction-static` to **0.8** (especially on **NVIDIA Blackwell B200/B300** GPUs). This leaves sufficient VRAM (~20%) for Megatron to initialize its structures before the first weight offload to CPU occurs. On GB200/GB300, values up to 0.75 are safer for long-running jobs to prevent potential OOMs. #TODO: Verify optimal fraction for Blackwell in production"
                ),
            )
            parser.add_argument(
                "--offload",
                action="store_true",
                default=False,
                help=("Equivalent to --offload-train + --offload-rollout. "),
            )
            parser.add_argument(
                "--offload-train",
                action=argparse.BooleanOptionalAction,
                help=(
                    "Whether to offload the training actor to CPU during training. "
                    "This will always be true when --colocate is set."
                ),
            )
            parser.add_argument(
                "--offload-rollout",
                action=argparse.BooleanOptionalAction,
                help=(
                    "Whether to offload the rollout generator to CPU during training. "
                    "This will always be true when --colocate is set."
                ),
            )

            reset_arg(
                parser,
                "--distributed-backend",
                type=str,
                default="nccl",
                help="Backend for distributed communication.",
            )
            reset_arg(
                parser,
                "--distributed-timeout-minutes",
                type=int,
                default=10,
                help="Timeout for distributed operations in minutes.",
            )

            return parser

        def add_train_arguments(parser):
            parser.add_argument(
                "--train-backend",
                type=str,
                choices=["megatron", "fsdp"],
                default="megatron",
                help="The backend for training. Highly suggest Megatron for numerical stability and efficiency.",
            )
            parser.add_argument(
                "--qkv-format",
                type=str,
                choices=["thd", "bshd"],
                default="thd",
                help="Whether to pack variable-length sequences into the token dimension for training. `thd` (T-H-D, a.k.a. varlen / packed sequence) concatenates sequences and uses `cu_seqlens` to avoid padding; it is the default and is usually faster by reducing padding overhead. `bshd` (B-S-H-D) uses fixed-shape padded batches; use it for newer models with novel attention architectures (e.g., sparse attention, attention sink) where the training backend does not support `thd`.",
            )
            parser.add_argument(
                "--true-on-policy-mode",
                action="store_true",
                default=False,
                help="Strictly align SGLang's log probs and training engine's log probs to bit-wise equal. This parameter is only used for FSDP right now. [Ref](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/mismatch/blog-en.md#truly-on-policy-training)",
            )
            parser.add_argument(
                "--train-env-vars",
                type=json.loads,
                default="{}",
                help="Extra environment variables for training process, e.g., PyTorch memory management ones.",
            )
            parser.add_argument(
                "--train-memory-margin-bytes",
                type=int,
                default=1024**3,
                help="Reserved memory margin for training in bytes. Defaults to 1GB.",
            )
            parser.add_argument(
                "--disable-weights-backuper",
                action="store_false",
                dest="enable_weights_backuper",
                help=(
                    "Applies to `megatron` training backend only. Disables the system that backs up model weights (Actor, Ref, Old Actor) to CPU RAM. Disabling saves significant host memory but prevents features that rely on weight-swapping, such as computing the KL-divergence against a reference model. **Note**: do not set `--ref-load` and `--keep-old-actor` if disable weights backuper."
                ),
            )
            parser.add_argument(
                "--megatron-to-hf-mode",
                choices=["raw", "bridge"],
                default="raw",
                help="Method to convert Megatron weights to HuggingFace format for SGLang integration.",
            )
            parser.add_argument(
                "--custom-model-provider-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom function that replaces the default model provider. [Ref](../get_started/customization.md#20-model-provider---custom-model-provider-path)"
                ),
            )
            parser.add_argument(
                "--recompute-loss-function",
                action="store_true",
                help="Enable recomputing the loss function to save memory during training.",
            )
            parser.add_argument(
                "--log-probs-chunk-size",
                type=int,
                default=-1,
                help="Specifies the chunk size for logprobs computation to reduce peak memory usage. Processing logits in smaller batches, it prevents CUDA OOM errors during long-context prefilling or re-computation. Set to `-1` to disable chunking. [Ref](https://github.com/sgl-project/sglang/pull/6318)",
            )

            return parser

        # rollout
        def add_rollout_arguments(parser):
            parser.add_argument(
                "--hf-checkpoint",
                type=str,
                default=None,
                help=("Path to the HuggingFace checkpoint used to initialize SGLang and provide the tokenizer."),
            )
            parser.add_argument(
                "--model-name",
                type=str,
                default=None,
                help=(
                    "The name of the model that is used to convert the Megatron weights into HuggingFace format. If not set, we will use `type(AutoConfig.from_pretrained(args.hf_checkpoint)).__name__.lower()` as `model_name`. Providing this argument can also help in cases where transformers cannot find certain models."
                ),
            )
            parser.add_argument(
                "--rollout-function-path",
                type=str,
                default=(
                    "miles.rollout.inference_rollout.inference_rollout_common.InferenceRolloutFn"
                    if enable_experimental_rollout_refactor()
                    else "miles.rollout.sglang_rollout.generate_rollout"
                ),
                help=(
                    "Path to the rollout generation function. Use this to inject custom logic (e.g., for multi-turn or tool use). [Ref](../get_started/customization.md#1-rollout-function---rollout-function-path)"
                ),
            )
            parser.add_argument(
                "--rollout-temperature",
                type=float,
                default=1.0,
                help="Sampling temperature for the inference engine during rollout.",
            )
            parser.add_argument(
                "--rollout-top-p", type=float, default=1.0, help="Top-p (nucleus) sampling threshold during rollout."
            )
            parser.add_argument(
                "--rollout-top-k",
                type=int,
                default=-1,
                help="Top-k sampling threshold during rollout. `-1` means disabled.",
            )
            parser.add_argument(
                "--rollout-max-context-len",
                type=int,
                default=None,
                help=(
                    "The maximum context size for the inference engine during rollout. It should not exceed the `max_position_embeddings` in the HuggingFace model's `config.json`. **Note:** This acts as a hard cap for the total tokens (Prompt + Response)."
                ),
            )
            parser.add_argument(
                "--rollout-max-prompt-len",
                type=int,
                default=None,
                help=(
                    "Maximum length of the prompt. Longer prompts are filtered during dataset initialization. This is not recommended if the dataset is large. **Note:** Defaults to `rollout-max-context-len - 1` if not set, ensuring at least one token can be generated."
                ),
            )
            parser.add_argument(
                "--rollout-max-response-len",
                type=int,
                default=None,
                help=(
                    "Maximum length of the response (`max_tokens` in SGLang). **Note:** Generation will stop when either this limit is reached or the total session length hits `rollout-max-context-len`."
                ),
            )
            parser.add_argument(
                "--rollout-skip-special-tokens",
                action="store_true",
                default=False,
                help=(
                    "Skip special tokens (e.g., `<\\|im_end\\|>`, `<\\|endoftext\\|>`) in the decoded response string. **Critical for Multi-Turn RL:** Ensures that when a response is appended to the conversation history for the next turn, it doesn't include terminal special tokens that would interfere with chat template formatting or cause early termination in subsequent turns."
                ),
            )
            parser.add_argument(
                "--rollout-stop",
                type=str,
                nargs="+",
                default=None,
                help=(
                    'A list of strings that trigger termination of generation if they appear in the output (e.g., `"\\nUser:"`).'
                ),
            )
            parser.add_argument(
                "--rollout-stop-token-ids",
                type=int,
                nargs="+",
                default=None,
                help=(
                    "A list of numerical token IDs that trigger termination. This is the token-level equivalent of `--rollout-stop` and is preferred for special control tokens that are difficult to input as strings."
                ),
            )
            parser.add_argument(
                "--rollout-shuffle",
                action="store_true",
                default=False,
                help=("Shuffle the prompts during rollout."),
            )
            parser.add_argument(
                "--rollout-seed",
                type=int,
                default=42,
                help=("Seed for the random number generator during rollout (used for shuffling and sampling)."),
            )

            # sampling
            parser.add_argument(
                "--over-sampling-batch-size",
                type=int,
                default=None,
                help=(
                    "Number of prompts requested in each **oversampling** round when **dynamic sampling** is enabled. Miles samples `over_sampling_batch_size` prompts, generates `--n-samples-per-prompt` responses per prompt asynchronously, and then keeps/discards each prompt group via `--dynamic-sampling-filter-path`. If filtering is strict and the remaining accepted batch size drops below the target `--rollout-batch-size`, Miles automatically triggers another oversampling round of the same size. If unset, defaults to `--rollout-batch-size`. See [Dynamic Sampling](../get_started/quick_start.md#dynamic-sampling)."
                ),
            )
            parser.add_argument(
                "--dynamic-sampling-filter-path",
                type=str,
                default=None,
                help=(
                    "Path to the filter function for dynamic sampling. [Ref](../get_started/customization.md#4-dynamic-sampling-filter---dynamic-sampling-filter-path)"
                ),
            )

            # partial rollout
            parser.add_argument(
                "--partial-rollout",
                action="store_true",
                default=False,
                help=(
                    "Enable partial rollout for **dynamic sampling**: cache partially generated (aborted/unfinished) samples and resume generation in later rollout steps, reducing wasted compute for long responses. Cached samples are stored in the rollout buffer and can be prioritized/selected via `--buffer-filter-path` (default FIFO behavior). See [Partial Rollout](../get_started/quick_start.md#partial-rollout)."
                ),
            )
            parser.add_argument(
                "--mask-offpolicy-in-partial-rollout",
                action="store_true",
                default=False,
                help=(
                    "When using partial rollout, mask the previously generated (cached) response tokens so they do not contribute to the loss; only tokens generated after resuming are used for training. This helps avoid training on a cached prefix produced by an older policy version. See [Partial Rollout](../get_started/quick_start.md#partial-rollout)."
                ),
            )
            parser.add_argument(
                "--custom-generate-function-path",
                type=str,
                default=None,
                help=(
                    "Path to override only the `generate` step within the default rollout function. If your custom `generate` returns `list[Sample]` (multi-sample), make sure your rollout pipeline can handle it; the default rollout expects a flat `list[Sample]` of length `--n-samples-per-prompt` for each prompt group. [Ref](../get_started/customization.md#2-custom-generate-function---custom-generate-function-path)"
                ),
            )
            parser.add_argument(
                "--custom-rollout-log-function-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom function for logging training rollout data. [Ref](../get_started/customization.md#14-logging-functions)"
                ),
            )
            parser.add_argument(
                "--custom-eval-rollout-log-function-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom function for logging evaluation rollout data. [Ref](../get_started/customization.md#14-logging-functions)"
                ),
            )

            parser.add_argument(
                "--buffer-filter-path",
                type=str,
                default=None,
                help=(
                    "Path to the function to filter or sort samples in the rollout buffer before training. [Ref](../get_started/customization.md#5-buffer-filter---buffer-filter-path)"
                ),
            )
            # update weight
            parser.add_argument(
                "--update-weight-buffer-size",
                type=int,
                default=512 * 1024**2,
                help=(
                    "Buffer size for updating weights, in bytes. [Ref](https://hebiao064.github.io/rl-weight-sync#42-optimizing-sglang-server-calls-with-tensor-bucketing-from-50s-to-30s)"
                ),
            )
            parser.add_argument(
                "--update-weights-interval",
                type=int,
                default=1,
                help="Interval (in rollout rounds) for syncing weights to inference engines. Set to `>1` for async RL.",
            )
            parser.add_argument(
                "--keep-old-actor",
                action="store_true",
                help='Maintains a "Model Queue" (Actor, Rollout Actor, Old Actor) to ensure importance sampling ratios are calculated against the exact policy version that generated the data. Essential for asynchronous RL where training and inference are decoupled, preventing mathematical incorrectness due to model staleness. It consumes additional Host Memory (extra ~1x model size for `update_weights_interval > 1` or 2x for `update_weights_interval == 1`) depending on update interval.',
            )

            parser.add_argument(
                "--rollout-data-postprocess-path",
                type=str,
                default=None,
                help=(
                    "Path to a function called after all rollout data (including log probs) is ready. [Ref](../get_started/customization.md#8-rollout-data-postprocess---rollout-data-postprocess-path)"
                ),
            )
            parser.add_argument(
                "--rollout-external",
                action="store_true",
                default=False,
                help="Use external SGLang instances instead of launching them inside the framework.",
            )
            parser.add_argument(
                "--rollout-external-engine-addrs",
                type=str,
                default=None,
                nargs="+",
                help="Addresses and ports of the external engines.",
            )
            return parser

        def add_fault_tolerance_arguments(parser):
            parser.add_argument(
                "--use-fault-tolerance",
                action="store_true",
                default=False,
                help="Enable fault tolerance for rollout engines. Periodically sends `/health_generate` heartbeats.",
            )
            parser.add_argument(
                "--rollout-health-check-interval",
                type=float,
                default=30.0,
                help="Interval in seconds between rollout engine `/health_generate` checks during generate/eval.",
            )
            parser.add_argument(
                "--rollout-health-check-timeout",
                type=float,
                default=30.0,
                help="Timeout in seconds to wait for a rollout engine `/health_generate` response before killing it.",
            )
            parser.add_argument(
                "--rollout-health-check-first-wait",
                type=float,
                default=0,
                help="Initial grace period (in seconds) before starting health checks. This allows time for model compilation and initialization. Increase this value significantly when using deepgemm.",
            )
            return parser

        # data
        def add_data_arguments(parser):
            # dataset
            # TODO: maybe add an num_epoch and calculate the num_rollout from buffer
            parser.add_argument(
                "--num-rollout",
                type=int,
                default=None,
                help="Number of rollout steps. If not set, Miles will calculate the number of rollout steps from the dataset size. **Note:** This value will be overwritten if `--num-epoch` is also set.",
            )
            parser.add_argument(
                "--num-epoch",
                type=int,
                default=None,
                help=(
                    "Number of epochs for the training. If set, `num_rollout` is calculated as `(num_epoch * dataset_size) // rollout_batch_size`. **Note:** This argument takes precedence and will overwrite `--num-rollout` if both are specified."
                ),
            )

            parser.add_argument(
                "--disable-rollout-global-dataset",
                action="store_false",
                dest="rollout_global_dataset",
                help=(
                    "Disable the global dataset for rollout. By default, Miles loads `--prompt-data` into a global dataset and samples from it for rollout. Setting this flag turns off this behavior. Use this flag only when providing a custom `--rollout-function-path` (and usually a custom `--data-source-path`) that handles data loading independently."
                ),
            )

            parser.add_argument(
                "--data-source-path",
                type=str,
                default="miles.rollout.data_source.RolloutDataSourceWithBuffer",
                help="Path to a custom Python class for the rollout data source. [Ref](../get_started/customization.md#15-data-source---data-source-path)",
            )
            parser.add_argument(
                "--prompt-data",
                type=str,
                default=None,
                help=(
                    "Path to the prompt dataset (JSONL format), and each line should contain `--input-key` and `--label-key`, which will be used as the prompt and the label, respectively."
                ),
            )
            parser.add_argument(
                "--apply-chat-template",
                action="store_true",
                default=False,
                help="Whether to apply the chat template to the input prompt. The input should be the same structure as an OpenAI message, e.g., `[{'role': 'user', 'content': 'blabla'}]`.",
            )
            # Temporarily be JSON-serialized str, will be a real dict after using Omegaconf
            parser.add_argument(
                "--apply-chat-template-kwargs",
                type=json.loads,
                default="{}",
                help="Extra arguments for the chat template processing (JSON string).",
            )
            parser.add_argument(
                "--input-key",
                type=str,
                default="input",
                help="Key in the JSONL data representing the user input/prompt.",
            )
            parser.add_argument(
                "--label-key",
                type=str,
                default=None,
                help="Key in the JSONL data representing the label/ground truth.",
            )
            parser.add_argument(
                "--multimodal-keys",
                type=json.loads,
                default=None,
                help=(
                    'JSON string for multimodal data mapping media types to data keys. Example: `\'{"image": "image_file"}\'`'
                ),
            )
            parser.add_argument(
                "--metadata-key",
                type=str,
                default="metadata",
                help="When adding tools during `apply_chat_template`, provide the key for the tools to the prompt dataset.",
            )
            parser.add_argument(
                "--tool-key",
                type=str,
                default="tools",
                help=("JSON key for tool definitions in the prompt dataset (used when applying chat templates)."),
            )

            parser.add_argument(
                "--start-rollout-id",
                type=int,
                default=None,
                help=(
                    "The starting rollout step. If not set, it is inferred from the --load checkpoint when resuming training. Otherwise, if training is not continuous, Miles will start training from scratch"
                ),
            )

            # batch sizes
            parser.add_argument(
                "--rollout-batch-size",
                type=int,
                required=True,
                help=(
                    "Number of prompts per rollout batch. The total data returned should be `rollout_batch_size` * `n_samples_per_prompt`."
                ),
            )
            parser.add_argument(
                "--n-samples-per-prompt",
                type=int,
                default=1,
                help="Number of responses to generate for each prompt, e.g., the group size of GRPO. The default rollout pipeline expects each prompt group to contain exactly `n_samples_per_prompt` samples.",
            )

            # gbs of the training, note that the gbs is of sample, not of prompts,
            # so if you hope to train 1 step for each rollout, the global_bach_size should be set as
            # `rollout_batch_size * n_samples_per_prompt`.
            reset_arg(
                parser,
                "--global-batch-size",
                type=int,
                default=None,
                help="Total samples per optimizer step. Automatically calculated or **overridden** if `num_steps_per_rollout` is set.",
            )
            parser.add_argument(
                "--num-steps-per-rollout",
                type=int,
                default=None,
                help=(
                    "The number of training steps to perform using the data collected in a single rollout round. Setting this to `n` means the policy model will be updated `n` times using the same batch of rollout data. Miles ensures that `(rollout-batch-size * n-samples-per-prompt) = (global-batch-size * num-steps-per-rollout)`. If this value is not provided, you have to set `--global-batch-size` explicitly. If both are provided, `--num-steps-per-rollout`  will **override** the global batch size with `num_steps_per_rollout = (rollout_batch_size * n_samples_per_prompt) // num_steps_per_rollout`."
                ),
            )
            # mbs for the training, will be ignored if `use_dynamic_batch_size` is set.
            reset_arg(
                parser,
                "--micro-batch-size",
                type=int,
                default=1,
                help="Micro batch size per GPU. Ignored when `--use-dynamic-batch-size` is enabled. Works for both `--qkv-format thd` and `--qkv-format bshd` (and is required for `bshd` because dynamic batch size is unsupported).",
            )
            parser.add_argument(
                "--balance-data",
                action="store_true",
                default=False,
                help=(
                    "Repartition each rollout batch so each data-parallel rank gets a similar total token count via the Karmarkar-Karp method. It may be beneficial for training speed, but changes per-rank sample grouping and adds a small CPU scheduling overhead."
                ),
            )

            parser.add_argument(
                "--use-dynamic-batch-size",
                action="store_true",
                default=False,
                help=(
                    "Dynamically packs variable-length samples into micro-batches to maximize GPU utilization, ensuring the total token count per batch does not exceed `--max-tokens-per-gpu`. For example, with a 300-token limit, samples of lengths 100, 200, and 300 would be packed into two batches: `[100, 200]` and `[300]`. **Note:** Miles ensures that enabling this optimization does not affect the mathematical correctness of per-sample or per-token loss calculation. It is **strongly recommended** to enable this for maximum efficiency. **Compatibility:** only supported when `--qkv-format` is `thd` (does not work for `bshd`)."
                ),
            )
            parser.add_argument(
                "--max-tokens-per-gpu",
                type=int,
                default=None,
                help=(
                    "The maximum number of tokens (Prompt + Response combined) per GPU for dynamic batch size. This parameter defines the total sequence length budget for packing samples into micro-batches during training. Note that when enabling context parallel (CP), the effective capacity is shared, so the value should be approximately `(Total_Sequence_Length) // cp_size`."
                ),
            )
            parser.add_argument(
                "--log-probs-max-tokens-per-gpu",
                type=int,
                default=None,
                help=(
                    "The maximum number of tokens per GPU for calculating log probs. This is used to calculate the log probs of the responses during rollout, and should be set to a larger value than `max_tokens_per_gpu` if you want better performance."
                ),
            )
            return parser

        def add_eval_arguments(parser):
            parser.add_argument(
                "--eval-function-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom evaluation function. [Ref](../get_started/customization.md#16-evaluation-function---eval-function-path)"
                ),
            )

            # change the default value of eval_interval from Megatron to None
            reset_arg(
                parser,
                "--eval-interval",
                type=int,
                default=None,
                help="Interval (in rollout steps) between evaluations.",
            )

            parser.add_argument(
                "--eval-prompt-data",
                type=str,
                default=None,
                nargs="+",
                help=("List of name and path pairs for evaluation datasets (e.g., `aime /path/to/aime.jsonl`)."),
            )
            parser.add_argument(
                "--eval-config",
                type=str,
                default=None,
                help=(
                    "Path to an OmegaConf YAML/JSON file describing evaluation datasets (overrides `--eval-prompt-data`)."
                ),
            )
            parser.add_argument(
                "--skip-eval-before-train",
                action="store_true",
                default=False,
                help="Skip the evaluation step before training starts.",
            )

            # The following keys are used to override the rollout version during eval.
            parser.add_argument(
                "--eval-input-key", type=str, default=None, help="JSON key for input text in evaluation datasets."
            )
            parser.add_argument(
                "--eval-label-key",
                type=str,
                default=None,
                help="JSON key for ground truth labels in evaluation datasets.",
            )
            parser.add_argument(
                "--eval-tool-key", type=str, default=None, help="JSON key for tool definitions in evaluation datasets."
            )
            parser.add_argument(
                "--n-samples-per-eval-prompt",
                type=int,
                default=1,
                help="Number of responses for each prompt in generation.",
            )
            parser.add_argument(
                "--eval-temperature",
                type=float,
                default=None,
                help="Temperature for evaluation (defaults to rollout temperature if not set).",
            )
            parser.add_argument(
                "--eval-top-p",
                type=float,
                default=None,
                help="Top-p sampling threshold for evaluation (defaults to rollout top-p if not set).",
            )
            parser.add_argument(
                "--eval-top-k",
                type=int,
                default=None,
                help="Top-k sampling threshold for evaluation (defaults to rollout top-k if not set).",
            )
            parser.add_argument(
                "--eval-max-response-len",
                type=int,
                default=None,
                help="Maximum response length for evaluation (defaults to rollout max response length if not set).",
            )
            parser.add_argument(
                "--eval-max-prompt-len", type=int, default=None, help="Maximum prompt length for evaluation."
            )
            parser.add_argument(
                "--eval-min-new-tokens",
                type=int,
                default=None,
                help="Minimum tokens to generate for evaluation responses (Not used).",
            )
            parser.add_argument(
                "--eval-max-context-len",
                type=int,
                default=None,
                help="Maximum context length for evaluation (defaults to rollout max context length if not set).",
            )

            return parser

        def add_algo_arguments(parser):
            parser.add_argument(
                "--ref-load",
                type=str,
                default=None,
                help=("Path to the reference model checkpoint. Used as an initial checkpoint if `--load` is not set."),
            )
            parser.add_argument(
                "--ref-ckpt-step", type=int, default=None, help="The checkpoint step for the reference model."
            )
            reset_arg(parser, "--load", type=str, default=None, help="Path to the training model checkpoint to load.")
            reset_arg(parser, "--save", type=str, default=None, help="Path to save checkpoints.")
            reset_arg(
                parser,
                "--save-interval",
                type=int,
                default=None,
                help="Interval (in rollout steps) to save checkpoints. Requires `--save` to be set.",
            )
            reset_arg(
                parser,
                "--async-save",
                action="store_true",
                help="Enable asynchronous checkpoint saving (Megatron backend only).",
            )
            reset_arg(
                parser,
                "--no-save-optim",
                action="store_true",
                default=False,
                help=(
                    "If set, optimizer state is not saved with checkpoints to reduce size, but prevents resumption of training."
                ),
            )
            parser.add_argument(
                "--save-hf",
                type=str,
                default=None,
                help=(
                    "Path to save the model in HuggingFace format when using Megatron backend. The model will be saved to `save_hf.format(rollout_id)`."
                ),
            )
            reset_arg(
                parser,
                "--seed",
                type=int,
                default=1234,
                help="Random seed for the training process. **Also passed to SGLang servers as `random_seed`** (Miles uses `seed + engine_rank` so each engine has a distinct but reproducible seed).",
            )
            reset_arg(
                parser, "--clip-grad", type=float, default=1.0, help="Maximum gradient norm for gradient clipping."
            )
            reset_arg(
                parser, "--calculate-per-token-loss", action="store_true", help="Calculate loss on a per-token basis."
            )
            reset_arg(parser, "--lr", type=float, default=1e-6, help="Learning rate for the Actor.")

            parser.add_argument(
                "--num-critic-only-steps",
                type=int,
                default=0,
                help="Number of initial steps dedicated to training only the Critic.",
            )
            parser.add_argument(
                "--critic-load", type=str, default=None, help="Checkpoint to load for the critic model."
            )
            parser.add_argument("--critic-save", type=str, default=None, help="Path to save the critic model.")
            parser.add_argument(
                "--critic-lr", type=float, default=None, help="Learning rate for the Critic. Defaults to `--lr`."
            )
            parser.add_argument(
                "--critic-lr-warmup-iters",
                type=int,
                default=0,
                help="Number of iterations for Critic learning rate linear warmup.",
            )

            parser.add_argument("--eps-clip", type=float, default=0.2, help="PPO clip range.")
            parser.add_argument(
                "--eps-clip-high",
                type=float,
                default=None,
                help="PPO clip upper range (defaults to `--eps-clip` if not set).",
            )
            parser.add_argument(
                "--eps-clip-c",
                type=float,
                default=None,
                help="Lower bound for [Dual-clip PPO](https://arxiv.org/pdf/1912.09729).",
            )
            parser.add_argument("--value-clip", type=float, default=0.2, help="Clip range for value loss.")
            parser.add_argument(
                "--kl-coef",
                type=float,
                default=0.00,
                help="KL penalty coefficient for reward shaping. This is applied to the reward signal before advantage calculation for PPO and REINFORCE-style estimator.",
            )
            parser.add_argument(
                "--loss-type",
                type=str,
                choices=["policy_loss", "sft_loss", "custom_loss"],
                default="policy_loss",
                help=("Type of loss function to use."),
            )
            parser.add_argument(
                "--custom-loss-function-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom loss calculation function (requires `--loss-type custom_loss`). [Ref](../get_started/customization.md#9-custom-loss-function---custom-loss-function-path)"
                ),
            )
            parser.add_argument(
                "--kl-loss-type",
                type=str,
                choices=["k1", "k2", "k3", "low_var_kl"],
                default="k1",
                help="Selection of the KL loss implementation. See [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html) for more details.",
            )
            parser.add_argument(
                "--advantage-estimator",
                type=str,
                choices=[
                    "grpo",
                    "gspo",
                    "reinforce_plus_plus",
                    "reinforce_plus_plus_baseline",
                    "ppo",
                    "on_policy_distillation",
                ],
                default="grpo",
                help="Advantage estimator to use.",
            )
            parser.add_argument(
                "--disable-compute-advantages-and-returns",
                action="store_false",
                dest="compute_advantages_and_returns",
                help=(
                    "Disables the calculation of advantages and returns. This is typically used for SFT or custom loss functions where value estimation is not required."
                ),
            )
            parser.add_argument(
                "--use-kl-loss",
                action="store_true",
                default=False,
                help="Enable KL loss term in the final objective (as in GRPO).",
            )
            parser.add_argument(
                "--kl-loss-coef",
                type=float,
                default=0.0,
                help="Weight of the KL loss term in the final objective.",
            )
            parser.add_argument(
                "--use-unbiased-kl",
                action="store_true",
                default=False,
                help="Apply Importance Sampling (IS) correction to the KL estimator. Reduces bias from distribution shift.",
            )
            parser.add_argument(
                "--ref-update-interval",
                type=int,
                default=None,
                help="Interval (in rollout steps) to update ref model from actor. If `None`, ref model is not updated.",
            )
            parser.add_argument(
                "--entropy-coef",
                type=float,
                default=0.0,
                help="Coefficient for entropy regularization term. Penalizes low entropy to encourage exploration and prevent premature convergence.",
            )
            parser.add_argument(
                "--gamma",
                type=float,
                default=1.0,
                help="Discount factor for future rewards. Used in PPO (GAE) and REINFORCE++.",
            )
            parser.add_argument("--lambd", type=float, default=1.0, help="PPO GAE lambda.")
            parser.add_argument(
                "--normalize-advantages",
                action="store_true",
                default=False,
                help="Performs distributed masked whitening of advantages. Normalization statistics are computed globally across the Data-Parallel group, ignoring padding tokens.",
            )
            parser.add_argument(
                "--disable-grpo-std-normalization",
                action="store_false",
                dest="grpo_std_normalization",
                help="Disable standard deviation normalization for GRPO. From [Dr.GRPO](https://arxiv.org/pdf/2503.20783)",
            )
            parser.add_argument(
                "--disable-rewards-normalization",
                action="store_false",
                dest="rewards_normalization",
                help="Disable the default group-wise reward normalization for GRPO, GSPO, and REINFORCE++. This effectively skips the baseline subtraction step.",
            )
            parser.add_argument(
                "--use-rollout-entropy",
                action="store_true",
                default=False,
                help=(
                    "Enable entropy calculation when calculating the logprobs from actor and reference model. This is useful for implementing custom entropy-based loss masking."
                ),
            )
            parser.add_argument(
                "--get-mismatch-metrics",
                action="store_true",
                default=False,
                help="Calculate mismatch metrics. If it is set, you need to provide a custom TIS function via `--custom-tis-function-path`.",
            )
            parser.add_argument(
                "--reset-optimizer-states",
                action="store_true",
                default=False,
                help=(
                    "Resets the optimizer state after each rollout round. This clears the optimization history, which can improve stability or satisfy specific experimental requirements."
                ),
            )
            parser.add_argument(
                "--use-rollout-logprobs",
                action="store_true",
                default=False,
                help=(
                    "Use rollout logprobs as the old-policy logprobs when computing importance sampling ratios / PPO-style KL in GRPO/GSPO/PPO. If not set, Miles recomputes old-policy logprobs with the training actor (e.g., `old_actor` or `actor`, depending on configuration). If `--get-mismatch-metrics` is set, the log probs will still be recomputed by the training engine (one more forward pass will be applied)."
                ),
            )
            # Off-Policy Correction using Importance Sampling: https://fengyao.notion.site/off-policy-rl
            parser.add_argument(
                "--use-tis",
                action="store_true",
                default=False,
                help="Enable Token-level Importance Sampling (TIS) from this [blog](https://fengyao.notion.site/off-policy-rl#279721e3f6c48092bbe2fcfe0e9c6b33).",
            )
            parser.add_argument(
                "--tis-clip",
                type=float,
                default=2.0,
                help="Clipping threshold C for importance sampling ratios to control variance.",
            )
            parser.add_argument(
                "--tis-clip-low",
                type=float,
                default=0,
                help="Lower bound clipping threshold C for importance sampling ratios to control variance.",
            )
            parser.add_argument(
                "--custom-tis-function-path",
                type=str,
                default=None,
                help="Path to a custom TIS or MIS function. [Ref](../get_started/customization.md#10-custom-tisrs-function---custom-tis-function-path)",
            )
            parser.add_argument(
                "--custom-pg-loss-reducer-function-path",
                type=str,
                default=None,
                help="Custom reducer function for policy gradient loss. [Ref](../get_started/customization.md#11-custom-pg-loss-reducer---custom-pg-loss-reducer-function-path)",
            )

            parser.add_argument(
                "--use-routing-replay",
                action="store_true",
                default=False,
                help="Enable R2 (Routing Replay) for MoE: record expert routing decisions during forward and replay them during backward. [Paper](https://arxiv.org/abs/2507.18071) **Note:** automatically set to `True` when `--use-rollout-routing-replay` is enabled.",
            )
            parser.add_argument(
                "--use-rollout-routing-replay",
                action="store_true",
                default=False,
                help="Enable R3 (Rollout Routing Replay) for MoE: record expert routing decisions during rollout and replay them during training. **Requires `--use-miles-router`**. [Paper](https://arxiv.org/abs/2510.11370) [Ref](miles-router.md#22-rollout-routing-replay-r3-for-moe)",
            )
            parser.add_argument(
                "--use-opsm",
                action="store_true",
                default=False,
                help="Enable Off-Policy Sequence Masking (OPSM). Filters sequences that have **BOTH** negative advantages (bad results) AND high KL divergence (stale data). This stabilizes training by preventing updates from unreliable, highly off-policy samples.",
            )
            parser.add_argument(
                "--opsm-delta",
                type=float,
                default=1e-4,
                help="The threshold for Off-Policy Sequence Masking (OPSM).",
            )
            return parser

        def add_router_arguments(parser):
            parser.add_argument(
                "--use-miles-router",
                action="store_true",
                default=False,
                help="Use Miles Router (FastAPI passthrough proxy) instead of SGLang Model Gateway for rollout routing. Required for features that depend on preserving extra rollout metadata (e.g., R3). [Ref](miles-router.md)",
            )
            parser.add_argument(
                "--miles-router-middleware-paths",
                type=str,
                nargs="+",
                default="",
                help="Paths to custom MilesRouter middleware functions. [Ref](../get_started/customization.md#18-miles-router-middleware---miles-router-middleware-paths)",
            )
            parser.add_argument(
                "--miles-router-timeout",
                type=float,
                default=None,
                help="Timeout for router HTTP requests in seconds.",
            )
            parser.add_argument(
                "--miles-router-max-connections",
                type=int,
                default=None,
                help="Max connections for MilesRouter HTTP client.",
            )
            parser.add_argument(
                "--miles-router-health-check-failure-threshold",
                type=int,
                default=3,
                help="Number of consecutive failures before marking a worker as unhealthy.",
            )
            parser.add_argument(
                "--miles-router-enable-token-input-for-chat-completions",
                action="store_true",
                default=False,
                help=(
                    "This is an experimental feature, and only supports for text model."
                    "Whether to enable token input for chat completions. If set, we will calculate "
                    "the input_ids for the prompt part inside miles and add it to the request body."
                    "This is reserved for cross turn token in under OAI format."
                ),
            )
            RouterArgs.add_cli_args(parser, use_router_prefix=True, exclude_host_port=True)
            return parser

        # wandb
        def add_wandb_arguments(parser):
            # wandb parameters
            parser.add_argument("--use-wandb", action="store_true", default=False, help="Enable WandB logging.")
            parser.add_argument(
                "--wandb-mode",
                type=str,
                default=None,
                choices=["online", "offline", "disabled"],
                help="WandB operating mode. Overrides `WANDB_MODE`.",
            )
            parser.add_argument(
                "--wandb-dir",
                type=str,
                default=None,
                help="Directory to store WandB logs. Default is `./wandb` in current directory.",
            )
            parser.add_argument("--wandb-key", type=str, default=None, help="WandB API key.")
            parser.add_argument("--wandb-host", type=str, default=None, help="WandB host address.")
            parser.add_argument("--wandb-team", type=str, default=None, help="WandB team name.")
            parser.add_argument("--wandb-group", type=str, default=None, help="WandB group name.")
            reset_arg(parser, "--wandb-project", type=str, default=None, help="WandB project name.")
            parser.add_argument(
                "--disable-wandb-random-suffix",
                action="store_false",
                dest="wandb_random_suffix",
                default=True,
                help=(
                    "Disable adding a random suffix to the WandB run name. By default, we will add a random 6 length string with characters to the run name."
                ),
            )
            parser.add_argument(
                "--wandb-always-use-train-step",
                action="store_true",
                default=False,
                help=("Use training steps instead of rollout steps for the x-axis."),
            )
            parser.add_argument(
                "--log-multi-turn",
                action="store_true",
                default=False,
                help="Log detailed information for multi-turn conversations.",
            )
            parser.add_argument(
                "--log-passrate",
                action="store_true",
                default=False,
                help="Enable logging of `pass@n` metrics.",
            )
            parser.add_argument(
                "--log-reward-category",
                type=str,
                default=None,
                help=(
                    "Log reward-category statistics (e.g., why the reward function marked a failure). Use this argument to specify the key in the reward dict."
                ),
            )
            parser.add_argument(
                "--log-correct-samples",
                action="store_true",
                default=False,
                help="Explicitly log metrics for correct samples.",
            )
            parser.add_argument("--wandb-run-id", type=str, default=None, help="Specific WandB run ID to resume.")
            return parser

        # tensorboard
        def add_tensorboard_arguments(parser):
            # tb_project_name, tb_experiment_name
            parser.add_argument(
                "--use-tensorboard", action="store_true", default=False, help="Enable Tensorboard logging."
            )
            parser.add_argument(
                "--tb-project-name",
                type=str,
                default=None,
                help="Tensorboard project directory.",
            )
            parser.add_argument("--tb-experiment-name", type=str, default=None, help="Tensorboard experiment name.")

            return parser

        # debug
        def add_debug_arguments(parser):
            parser.add_argument(
                "--save-debug-rollout-data",
                type=str,
                default=None,
                help=("Path to save rollout data for offline analysis. [Ref](../developer_guide/debug.md)"),
            )
            parser.add_argument(
                "--load-debug-rollout-data",
                type=str,
                default=None,
                help=("Path to load debug rollout data (bypasses SGLang). [Ref](../developer_guide/debug.md)"),
            )
            parser.add_argument(
                "--load-debug-rollout-data-subsample",
                type=float,
                default=None,
                help="Percentage of debug data to load (0.0 to 1.0). [Ref](../developer_guide/debug.md)",
            )
            parser.add_argument(
                "--debug-rollout-only",
                action="store_true",
                default=False,
                help=("Run the rollout phase only without training. [Ref](../developer_guide/debug.md)"),
            )
            parser.add_argument(
                "--debug-train-only",
                action="store_true",
                default=False,
                help=(
                    "Run the training phase only without launching SGLang servers. [Ref](../developer_guide/debug.md)"
                ),
            )
            parser.add_argument(
                "--save-debug-train-data",
                type=str,
                default=None,
                help=("Path to save training batches for offline math debugging."),
            )
            parser.add_argument(
                "--dump-details",
                type=str,
                default=None,
                help=("Dump exhaustive training details for post-hoc visualization."),
            )
            # use together with --record-memory-history and --memory-snapshot-path (defined in Megatron)
            parser.add_argument(
                "--memory-snapshot-dir", type=str, default=".", help="Directory for PyTorch memory snapshots."
            )
            parser.add_argument(
                "--memory-snapshot-num-steps",
                type=int,
                default=None,
                help="Number of steps to record before saving snapshot.",
            )
            parser.add_argument(
                "--profile-target",
                type=str,
                choices=["train_overall", "train_actor", "train_log_probs"],
                default=["train_overall"],
                nargs="+",
                help="Training components to profile (accepts multiple).",
            )
            parser.add_argument(
                "--memory-recorder",
                type=str,
                choices=["torch", "memray"],
                default="torch",
                help="Selection of the memory recording backend.",
            )
            parser.add_argument(
                "--check-weight-update-equal",
                action="store_true",
                help="Use SGLang's weight checker to check and ensure that the loaded weight from HF checkpoint and received from Megatron are bit-wise equal.",
            )
            return parser

        def add_network_arguments(parser):
            parser.add_argument(
                "--http-proxy", type=str, default=None, help="HTTP proxy server for remote reward model calls."
            )
            parser.add_argument(
                "--use-distributed-post",
                action="store_true",
                default=False,
                help="Use distributed POST requests for remote reward models.",
            )
            return parser

        def add_reward_model_arguments(parser):
            parser.add_argument(
                "--rm-type",
                type=str,
                default=None,
                help="Built-in reward model selection.",
            )
            parser.add_argument(
                "--reward-key",
                type=str,
                default=None,
                help=(
                    "JSON key to extract the numerical reward from a returned dictionary if reward model returns a dict instead of a value."
                ),
            )
            parser.add_argument(
                "--eval-reward-key",
                type=str,
                default=None,
                help="Evaluation variant for `--reward-key`.",
            )
            parser.add_argument(
                "--group-rm",
                action="store_true",
                default=False,
                help="Defer reward computation to process the entire group of samples (per-prompt) at once. Essential for comparative/ranking reward models and improves throughput. **Not supported in eval**.",
            )
            parser.add_argument(
                "--rm-url",
                type=str,
                default=None,
                help="URL for the reward model service (used with `--rm-type remote_rm`).",
            )
            parser.add_argument(
                "--custom-rm-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom Python reward function. [Ref](../get_started/customization.md#3-reward-model---custom-rm-path)"
                ),
            )
            parser.add_argument(
                "--custom-reward-post-process-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom reward post-processor. [Ref](../get_started/customization.md#12-reward-post-processing---custom-reward-post-process-path)"
                ),
            )
            parser.add_argument(
                "--custom-convert-samples-to-train-data-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom data format converter. [Ref](../get_started/customization.md#13-samples-to-train-data-conversion---custom-convert-samples-to-train-data-path)"
                ),
            )
            return parser

        def add_rollout_buffer_arguments(parser):
            parser.add_argument(
                "--rollout-buffer-url",
                type=str,
                default=None,
                help="URL for the rollout buffer service.",
            )

            parser.add_argument(
                "--fetch-trajectory-retry-times",
                type=int,
                default=-1,
                help="Number of times to retry fetching trajectory, -1 means unlimited retry.",
            )
            parser.add_argument(
                "--min-batch-collection-ratio",
                type=float,
                default=1,
                help="Minimum batch collection ratio before proceeding.",
            )
            parser.add_argument("--rollout-task-type", type=str, default="math", help="Type of task being performed.")
            parser.add_argument(
                "--loss-mask-type",
                type=str,
                default="qwen",
                choices=["qwen", "qwen3", "distill_qwen"],
                help="Selection of the token masking logic.",
            )
            parser.add_argument(
                "--data-pad-size-multiplier",
                type=int,
                default=128,
                help="Multiplier used to calculate the sequence padding boundary. Miles rounds sequence lengths up to a multiple of `tensor_parallel_size * data_pad_size_multiplier`. This optimization ensures that matrix dimensions are aligned with NVIDIA Tensor Core requirements, maximizing throughput and reducing VRAM fragmentation. **Note:** better not change this; values `<128` may trigger accuracy loss under `--qkv-format thd` when `TP>=4`.",
            )
            parser.add_argument(
                "--rollout-sample-filter-path",
                type=str,
                default=None,
                help=(
                    "Path to the function that marks individual samples to be excluded from loss calculation. [Ref](../get_started/customization.md#6-rollout-sample-filter---rollout-sample-filter-path)"
                ),
            )
            parser.add_argument(
                "--rollout-all-samples-process-path",
                type=str,
                default=None,
                help=(
                    "Path to the function to process all samples (including filtered ones) after rollout. [Ref](../get_started/customization.md#7-rollout-all-samples-process---rollout-all-samples-process-path)"
                ),
            )
            parser.add_argument(
                "--disable-rollout-trim-samples",
                action="store_true",
                default=False,
                help="Disable trim samples in rollout buffer when converting samples to train data.",
            )
            parser.add_argument(
                "--use-dynamic-global-batch-size",
                action="store_true",
                default=False,
                help="Enable dynamic global batch size, disable trim samples in rollout buffer when converting samples to train data.",
            )
            return parser

        def add_custom_megatron_plugins_arguments(parser):
            """
            Add custom Megatron plugins arguments.
            This is a placeholder for any additional arguments that might be needed.
            """
            # Custom arguments can be added here
            parser.add_argument(
                "--custom-megatron-init-path",
                type=str,
                default=None,
                help="Path to custom Megatron initialization logic. [Ref](../get_started/customization.md#17-megatron-hooks)",
            )
            parser.add_argument(
                "--custom-megatron-before-log-prob-hook-path",
                type=str,
                default=None,
                help="Hook called before calculating log probabilities. [Ref](../get_started/customization.md#17-megatron-hooks)",
            )
            parser.add_argument(
                "--custom-megatron-before-train-step-hook-path",
                type=str,
                default=None,
                help="Hook called before each training step. [Ref](../get_started/customization.md#17-megatron-hooks)",
            )
            return parser

        def add_mtp_training_arguments(parser):
            """Add MTP training specific arguments."""
            reset_arg(parser, "--mtp-num-layers", type=int, default=None, help="Number of MTP layers to include.")
            reset_arg(
                parser,
                "--mtp-loss-scaling-factor",
                type=float,
                default=0.2,
                help="Scaling factor applied to the MTP loss.",
            )
            parser.add_argument(
                "--enable-mtp-training",
                action="store_true",
                default=False,
                help="Enable MTP layer parameter updates during training.",
            )

            return parser

        def add_prefill_decode_disaggregation_arguments(parser):
            parser.add_argument(
                "--prefill-num-servers",
                type=int,
                default=None,
                help="Number of dedicated prefill servers for PD disaggregation.",
            )
            return parser

        def add_ci_arguments(parser):
            parser.add_argument("--ci-test", action="store_true", help="Enable Continuous Integration testing mode.")
            parser.add_argument(
                "--ci-disable-kl-checker", action="store_true", help="Disable KL divergence sanity checks in CI."
            )
            parser.add_argument(
                "--ci-metric-checker-key", type=str, default=None, help="Metric key to monitor for pass/fail in CI."
            )
            parser.add_argument(
                "--ci-metric-checker-threshold",
                type=float,
                default=None,
                help="Pass/fail threshold (minimum value) for the monitored metric.",
            )
            parser.add_argument(
                "--ci-save-grad-norm", type=str, default=None, help="Path to save gradient norms for CI comparison."
            )
            parser.add_argument(
                "--ci-load-grad-norm", type=str, default=None, help="Path to load gradient norms for CI verification."
            )
            return parser

        def add_user_provided_function_arguments(parser):
            args_partial, _ = parser.parse_known_args()
            for path in [
                args_partial.rollout_function_path,
                args_partial.custom_generate_function_path,
            ]:
                try:
                    fn = load_function(path)
                except (ModuleNotFoundError, ValueError):
                    continue
                if fn is not None and callable(getattr(fn, "add_arguments", None)):
                    fn.add_arguments(parser)
            return parser

        def add_sglang_tp_size():
            temp_parser = argparse.ArgumentParser(add_help=False)
            temp_parser.add_argument(
                "--rollout-num-gpus-per-engine",
                type=int,
                default=1,
                help="Number of GPUs per inference engine, same as `tp_size` in SGLang. For multi-node serving, this should be the total GPU count / `tp_size` for each SGLang instance.",
            )
            temp_args, _ = temp_parser.parse_known_args()
            sglang_tp_size = temp_args.rollout_num_gpus_per_engine
            return sglang_tp_size

        # Add custom arguments in front to prevent overwritten some miles arguments.
        if add_custom_arguments is not None:
            parser = add_custom_arguments(parser)

        parser = add_cluster_arguments(parser)
        parser = add_train_arguments(parser)
        parser = add_rollout_arguments(parser)
        parser = add_fault_tolerance_arguments(parser)
        parser = add_data_arguments(parser)
        parser = add_eval_arguments(parser)
        parser = add_algo_arguments(parser)
        parser = add_wandb_arguments(parser)
        parser = add_tensorboard_arguments(parser)
        parser = add_router_arguments(parser)
        parser = add_debug_arguments(parser)
        parser = add_sglang_arguments(parser)
        parser = add_network_arguments(parser)
        parser = add_reward_model_arguments(parser)
        parser = add_rollout_buffer_arguments(parser)
        parser = add_mtp_training_arguments(parser)
        parser = add_prefill_decode_disaggregation_arguments(parser)
        parser = add_ci_arguments(parser)
        parser = add_custom_megatron_plugins_arguments(parser)
        if enable_experimental_rollout_refactor():
            parser = add_user_provided_function_arguments(parser)
        reset_arg(
            parser,
            "--custom-config-path",
            type=str,
            default=None,
            help="Path to the YAML config for custom function arguments.",
        )
        reset_arg(
            parser, "--padded-vocab-size", type=int, default=None, help="Manually specify the vocab size for padding."
        )

        parser.set_defaults(sglang_tensor_parallel_size=add_sglang_tp_size())
        return parser

    return add_miles_arguments


def parse_args(add_custom_arguments=None):
    # Users may call `parse_args` very early, thus we ensure logger is configured here
    configure_logger()

    add_miles_arguments = get_miles_extra_args_provider(add_custom_arguments)

    backend = parse_args_train_backend()
    if backend == "megatron":
        from miles.backends.megatron_utils.arguments import parse_args as megatron_parse_args
        from miles.backends.megatron_utils.arguments import set_default_megatron_args
        from miles.backends.megatron_utils.arguments import validate_args as megatron_validate_args

        args = megatron_parse_args(extra_args_provider=add_miles_arguments)
        if args.hf_checkpoint:
            hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
            hf_validate_args(args, hf_config)

        args.rank = 0
        args.world_size = args.actor_num_nodes * args.actor_num_gpus_per_node
        args = set_default_megatron_args(args)
    else:
        from miles.backends.fsdp_utils.arguments import load_fsdp_args

        args = load_fsdp_args(extra_args_provider=add_miles_arguments)
        args.rank = 0  # Primary process rank for wandb initialization
        args.world_size = args.actor_num_nodes * args.actor_num_gpus_per_node

        assert args.context_parallel_size == 1, "Context parallelism is not supported for FSDP backend."

    miles_validate_args(args)

    if backend == "megatron":
        megatron_validate_args(args)

        # always use varlen
        args.variable_seq_lengths = True
        if getattr(args, "moe_token_dispatcher_type", None) == "allgather":
            logger.info(
                "--moe-token-dispatcher-type allgather does not support variable sequence length, "
                "please use alltoall dispatcher instead."
            )
            args.moe_token_dispatcher_type = "alltoall"

    sglang_validate_args(args)

    return args


def parse_args_train_backend():
    if os.environ.get("MILES_BACKEND") is not None:
        raise Exception("`MILES_BACKEND` is deprecated, please use --train-backend directly.")

    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    args_partial, _ = parser.parse_known_args()
    return args_partial.train_backend


def _resolve_eval_datasets(args) -> list[EvalDatasetConfig]:
    """
    Build evaluation dataset configurations from either --eval-config or --eval-prompt-data.
    """
    datasets_config = []
    defaults: dict[str, Any] = {}

    if args.eval_config:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(args.eval_config)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(cfg_dict, dict):
            raise ValueError("--eval-config must contain a mapping at the root.")

        eval_cfg = cfg_dict.get("eval", cfg_dict)
        if not isinstance(eval_cfg, dict):
            raise ValueError("--eval-config must define an `eval` mapping or be a mapping itself.")

        defaults = dict(eval_cfg.get("defaults") or {})
        datasets_config = ensure_dataset_list(eval_cfg.get("datasets"))
        if not datasets_config:
            raise ValueError("--eval-config does not define any datasets under `eval.datasets`.")
    elif args.eval_prompt_data:
        values = list(args.eval_prompt_data)
        if len(values) == 1:
            logger.info("[legacy] only one eval_prompt_data detected, will assume it is data for aime")
            values = ["aime", values[0]]
        if len(values) % 2 != 0:
            raise ValueError("eval prompt data must be provided as name/path pairs.")
        datasets_config = [{"name": values[i], "path": values[i + 1]} for i in range(0, len(values), 2)]
    else:
        datasets_config = []

    eval_datasets = build_eval_dataset_configs(args, datasets_config, defaults)
    if eval_datasets:
        args.eval_prompt_data = [item for dataset in eval_datasets for item in (dataset.name, dataset.path)]
    else:
        args.eval_prompt_data = None

    return eval_datasets


def miles_validate_args(args):
    args.eval_datasets = _resolve_eval_datasets(args)

    if args.kl_coef != 0 or args.use_kl_loss:
        if not os.path.exists(args.ref_load):
            raise FileNotFoundError(f"ref_load {args.ref_load} does not exist, please check the path.")

        if not os.path.exists(os.path.join(args.ref_load, "latest_checkpointed_iteration.txt")):
            logger.info(
                f"ref_load {args.ref_load} does not have latest_checkpointed_iteration.txt, "
                "please make sure it is a valid megatron checkpoint directory."
            )

    # TODO: During loading, we need to set the start_rollout_id here.
    if args.megatron_to_hf_mode == "bridge":
        if args.load is None:
            args.load = args.ref_load or args.hf_checkpoint
        args.start_rollout_id = 0
    else:
        if (
            args.load is None
            or not os.path.exists(args.load)
            or not os.path.exists(os.path.join(args.load, "latest_checkpointed_iteration.txt"))
        ):
            args.no_load_optim = True
            args.no_load_rng = True
            args.finetune = True
            args.load = args.ref_load
            if args.ref_ckpt_step is not None:
                args.ckpt_step = args.ref_ckpt_step
            args.start_rollout_id = 0

    if args.eval_interval is not None:
        assert args.eval_datasets, "Evaluation datasets must be configured when eval_interval is set."

    if args.save_interval is not None:
        assert args.save is not None, "'--save' is required when save_interval is set."

    assert not (args.kl_coef != 0 and args.kl_loss_coef != 0), "Only one of kl_coef and kl_loss_coef can be set"

    if args.advantage_estimator in ["reinforce_plus_plus", "reinforce_plus_plus_baseline"]:
        assert args.normalize_advantages, (
            "The 'reinforce_plus_plus' and 'reinforce_plus_plus_baseline' advantage estimators "
            "require advantage normalization. Please add `--normalize-advantages` to your command."
        )

    if args.use_rollout_logprobs:
        assert not args.use_tis, "use_rollout_logprobs and use_tis cannot be set at the same time."

    if args.get_mismatch_metrics:
        assert (
            args.custom_tis_function_path is not None
        ), "custom_tis_function_path must be set when get_mismatch_metrics is set"

        if args.use_rollout_logprobs:
            logger.info(
                "get_mismatch_metrics is set; For metrics calculation, the log probs will still be recomputed by training engine. One more forward pass will be applied."
            )

    if args.use_dynamic_batch_size:
        assert args.max_tokens_per_gpu is not None, "max_tokens_per_gpu must be set when use_dynamic_batch_size is set"
        if args.log_probs_max_tokens_per_gpu is None:
            args.log_probs_max_tokens_per_gpu = args.max_tokens_per_gpu

    if args.eps_clip_high is None:
        args.eps_clip_high = args.eps_clip

    if args.eval_reward_key is None:
        args.eval_reward_key = args.reward_key

    if args.dump_details is not None:
        args.save_debug_rollout_data = f"{args.dump_details}/rollout_data/{{rollout_id}}.pt"
        args.save_debug_train_data = f"{args.dump_details}/train_data/{{rollout_id}}_{{rank}}.pt"

    if args.load_debug_rollout_data is not None:
        logger.info(
            f"load_debug_rollout_data {args.load_debug_rollout_data} is set, "
            "will not instantiate sglang servers and will only run the training process."
        )
        args.debug_train_only = True

    args.use_critic = args.advantage_estimator == "ppo"
    if args.critic_num_gpus_per_node is None:
        args.critic_num_gpus_per_node = args.actor_num_gpus_per_node
    if args.critic_num_nodes is None:
        args.critic_num_nodes = args.actor_num_nodes
    if args.critic_load is None:
        args.critic_load = args.load
    if args.critic_lr is None:
        args.critic_lr = args.lr

    if args.offload:
        args.offload_train = True
        args.offload_rollout = True
    del args.offload

    if args.debug_rollout_only:
        if args.colocate and (not args.rollout_num_gpus):
            args.rollout_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
        else:
            args.actor_num_gpus_per_node = min(8, args.rollout_num_gpus)
            args.actor_num_nodes = args.rollout_num_gpus // args.actor_num_gpus_per_node
        args.colocate = False
        args.offload_train = args.offload_rollout = False
        if args.train_memory_margin_bytes > 0:
            logger.warning("Force train_memory_margin_bytes=0 since debug_rollout_only does not support it")
            args.train_memory_margin_bytes = 0

    assert not (args.debug_rollout_only and args.debug_train_only), (
        "debug_rollout_only and debug_train_only cannot be set at the same time, " "please set only one of them."
    )

    # always true on offload for colocate at the moment.
    if args.colocate:
        if args.offload_train is None:
            args.offload_train = True
        if args.offload_rollout is None:
            args.offload_rollout = True
        if args.rollout_num_gpus != args.actor_num_gpus_per_node * args.actor_num_nodes:
            logger.info(
                f"rollout_num_gpus {args.rollout_num_gpus} != actor_num_gpus_per_node {args.actor_num_gpus_per_node} "
                f"* actor_num_nodes {args.actor_num_nodes}, overriding rollout_num_gpus to match actor_num_gpus_per_node * actor_num_nodes."
            )
            args.rollout_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
            if args.use_critic:
                args.rollout_num_gpus += args.critic_num_gpus_per_node * args.critic_num_nodes

    if args.offload_train is None:
        args.offload_train = False
    if args.offload_rollout is None:
        args.offload_rollout = False

    if args.eval_function_path is None:
        args.eval_function_path = args.rollout_function_path

    if args.num_steps_per_rollout is not None:
        global_batch_size = args.rollout_batch_size * args.n_samples_per_prompt // args.num_steps_per_rollout
        if args.global_batch_size is not None:
            assert args.global_batch_size == global_batch_size, (
                f"global_batch_size {args.global_batch_size} is not equal to "
                f"rollout_batch_size {args.rollout_batch_size} * n_samples_per_prompt {args.n_samples_per_prompt} "
                f"// num_steps_per_rollout {args.num_steps_per_rollout}"
            )
        args.global_batch_size = global_batch_size

    if args.n_samples_per_prompt == 1:
        args.grpo_std_normalization = False
        logger.info("n_samples_per_prompt is set to 1, grpo_std_normalization will be set to False.")

    if args.over_sampling_batch_size is None:
        args.over_sampling_batch_size = args.rollout_batch_size

    assert args.over_sampling_batch_size >= args.rollout_batch_size, (
        f"over_sampling_batch_size {args.over_sampling_batch_size} should be greater than or equal to "
        f"rollout_batch_size {args.rollout_batch_size}"
    )

    if args.num_epoch is not None:
        if args.num_rollout is not None:
            logger.info("Both num_epoch and num_rollout are set, num_epoch will be ignored.")
        else:
            assert args.rollout_global_dataset, (
                "num_epoch is set, but rollout_global_dataset is not set, "
                "please remove --disable-rollout-global-dataset to use num_epoch"
            )
    else:
        # if num_epoch is not set, we should set num_rollout
        assert args.num_rollout is not None, (
            "num_epoch is not set, but num_rollout is not set, " "please set --num-rollout or --num-epoch"
        )

    if args.enable_mtp_training:
        assert args.mtp_num_layers, "mtp_num_layers must be set when enable_mtp_training is set"

    if args.use_rollout_routing_replay:
        args.use_routing_replay = True

    if args.custom_config_path:
        with open(args.custom_config_path) as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if hasattr(args, k):
                logger.info(f"Warning: Argument {k} is already set to {getattr(args, k)}, will override with {v}.")
            setattr(args, k, v)

    if args.eval_max_context_len is None:
        logger.info(
            f"args.eval_max_context_len is not set. Use args.rollout_max_context_len {args.rollout_max_context_len} as default value."
        )
        args.eval_max_context_len = args.rollout_max_context_len

    if args.rollout_max_context_len is not None:
        if args.rollout_max_prompt_len is None:
            args.rollout_max_prompt_len = args.rollout_max_context_len - 1
            logger.info(
                f"args.rollout_max_prompt_len is not set. Use args.rollout_max_context_len - 1 ({args.rollout_max_context_len} - 1) as default value so that there is at least one generated token to compute loss."
            )
        assert (
            args.rollout_max_prompt_len <= args.rollout_max_context_len - 1
        ), f"args.rollout_max_prompt_len ({args.rollout_max_prompt_len}) must be smaller than args.rollout_max_context_len ({args.rollout_max_context_len}) so that there is at least one generated token to compute loss."

    assert not (
        args.prefill_num_servers is not None and args.rollout_external
    ), "prefill_num_servers cannot be set when rollout_external is set."

    if args.qkv_format == "bshd":
        assert args.train_backend == "megatron", "bshd format is only supported for megatron backend."
        assert (
            args.use_dynamic_batch_size is False
        ), "Dynamic batch size is not supported for bshd format. Please specify --micro-batch-size instead."


def hf_validate_args(args, hf_config):
    def equal(x, y):
        return x == y

    errors = []

    # multimodal models have different config structure
    if hasattr(hf_config, "text_config"):
        hf_config = hf_config.text_config

    for hf_config_name, megatron_config_name, compare_fn in [
        ("hidden_size", "hidden_size", equal),
        ("num_attention_heads", "num_attention_heads", equal),
        ("num_hidden_layers", "num_layers", equal),
        ("intermediate_size", "ffn_hidden_size", equal),
        ("tie_word_embeddings", "untie_embeddings_and_output_weights", lambda x, y: not x == y),
        ("rms_norm_eps", "norm_epsilon", equal),
        ("rope_theta", "rotary_base", equal),
    ]:
        if hasattr(hf_config, hf_config_name):
            if not compare_fn(getattr(hf_config, hf_config_name), getattr(args, megatron_config_name)):
                errors.append(
                    f"{hf_config_name} in hf config {getattr(hf_config, hf_config_name)} is not equal to "
                    f"{megatron_config_name} {getattr(args, megatron_config_name)}, please check the config."
                )

    if len(errors) > 0:
        raise AssertionError("hf_validate_args failed: " + "; ".join(errors))
