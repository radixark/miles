import json
from typing import Any, ClassVar

from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.args.schema import A, Arg, BaseConfig


# data
class DataConfig(BaseConfig):
    _mutable_fields: ClassVar[frozenset[str]] = frozenset({"start_rollout_id", "num_rollout"})

    # dataset
    # TODO: maybe add an num_epoch and calculate the num_rollout from buffer
    num_rollout: A[
        int | None,
        Arg(
            help="Number of rollout steps. If not set, we will calculate the number of rollout steps from the dataset size."
        ),
    ] = None
    debug_exit_after_rollout: A[
        int | None,
        Arg(
            help="Exit training after this many rollouts (for testing checkpoint resume with consistent scheduler params)."
        ),
    ] = None
    num_epoch: A[
        int | None,
        Arg(
            help=(
                "Number of epochs for the training. "
                "This is used to calculate the number of rollout steps from the dataset size. "
                "If set, we will calculate the number of rollout steps as `num_rollout = num_epoch * dataset_size // rollout_batch_size`."
                "If both `--num-epoch` and `--num-rollout` are set, `--num-epoch` will be ignored."
            )
        ),
    ] = None
    rollout_global_dataset: A[
        bool,
        Arg(
            cli_name="--disable-rollout-global-dataset",
            action="store_false",
            help=(
                "Disable the global dataset for rollout. By default, Miles loads `--prompt-data` into a global dataset and samples from it for rollout. "
                "Setting this flag turns off this behavior, Use this flag only when providing a custom `--rollout-function-path` (and usually a custom `--data-source-path`) that handles data loading independently."
            ),
        ),
    ] = True
    data_source_path: A[
        CustomFunctionConfig,
        Arg(help="The data source class for rollout data."),
    ] = CustomFunctionConfig(path="miles.rollout.data_source.RolloutDataSource")
    prompt_data: A[
        str | None,
        Arg(
            help=(
                "The path to the prompt data. "
                "Currently we only support jsonl format, and each line should contains --input-key and --label-key, "
                "which will be used as the prompt and the label respectively."
                "If you want to use a custom template, you can set --apply-chat-template to true, in that case, "
                "the input should be the same structure as an openai message, e.g. [{'role': 'user', 'content': 'blabla'}]. "
            )
        ),
    ] = None
    apply_chat_template: A[bool, Arg()] = False
    # Temporarily be JSON-serialized str, will be a real dict after using Omegaconf
    apply_chat_template_kwargs: A[Any, Arg(type_parser=json.loads)] = "{}"
    chat_template_path: A[
        str | None,
        Arg(
            help=(
                "Path to an explicit custom Jinja chat template file (.jinja). "
                "Sets tokenizer.chat_template when loading via load_tokenizer, "
                "and also sets --sglang-chat-template so the sglang server uses the same template. "
                "For Miles-maintained fixed templates, leave this unset and pass "
                "--tito-model so Miles can auto-resolve the registered template. "
                "The literal value 'autofix' is kept only as a "
                "deprecated compatibility alias for that auto-resolve path. "
                "The path must be accessible on all Ray worker nodes "
                "(e.g. a path inside the miles repo, or a shared filesystem like NFS)."
            )
        ),
    ] = None
    input_key: A[str, Arg(help="JSON dataset key")] = "input"
    label_key: A[str | None, Arg(help="JSON dataset key")] = None
    multimodal_keys: A[
        Any,
        Arg(
            type_parser=json.loads,
            help=(
                'JSON string for multimodal data mapping media types to data keys. Example: \'{"image": "image_file"}\''
            ),
        ),
    ] = None
    metadata_key: A[str, Arg(help="JSON dataset key")] = "metadata"
    tool_key: A[
        str,
        Arg(
            help="When need to add tools during apply_chat_template, you should provide the key for the tools in the prompt dataset."
        ),
    ] = "tools"
    start_rollout_id: A[
        int | None,
        Arg(
            help=(
                "The starting rollout step, if not set, will try to load the step from --load when doing continue training, "
                "otherwise will be set to 0, meaning training from start."
            )
        ),
    ] = None

    # batch sizes
    rollout_batch_size: A[
        int | None,
        Arg(
            help=(
                "The number of prompts in each rollout step. "
                "The total data returned should be rollout_batch_size * n_samples_per_prompt. "
                "Required for the train entry. "
            ),
        ),
    ] = None
    n_samples_per_prompt: A[int, Arg(help="Number of responses for each prompt in generation")] = 1

    num_steps_per_rollout: A[
        int | None,
        Arg(
            help=(
                "Number of steps per rollout, e.g. It is equivalent to setting gbs as "
                "`rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout`."
            )
        ),
    ] = None
    balance_data: A[
        bool,
        Arg(
            help=(
                "Repartition each rollout batch so each data-parallel rank gets a similar total token count via Karmarkar-Karp method. "
                "It may be beneficial for training speed but changes per-rank sample grouping and adds a small CPU scheduling overhead."
            )
        ),
    ] = False
    balance_by_flops: A[
        bool,
        Arg(
            help=(
                "Use FLOPs-based workload estimation for micro-batch partitioning via "
                "Karmarkar-Karp instead of first-fit token packing, and distribute mbs "
                "across DP ranks by FLOPs. Captures the quadratic attention cost when "
                "sequence lengths vary widely. Requires --use-dynamic-batch-size. NOTE: "
                "FLOPs balancing does not enforce the per-mbs token cap."
            )
        ),
    ] = False
    allow_partial_train_step: A[
        bool,
        Arg(
            help=(
                "Train the trailing rollouts that don't fill a whole global_batch_size step as one "
                "smaller final step instead of dropping them (rollout-side schedule + dynamic batch "
                "size only). Loss normalization and the LR scheduler use the true per-step count."
            )
        ),
    ] = False
    use_dynamic_batch_size: A[
        bool,
        Arg(
            help=(
                "Because the sample length varies, to maximize the GPU utilization, "
                "we will use the dynamic batch size to adjust the micro batch size according to the maximum number of tokens each gpu can run. "
                "For example, if we have 3 samples, with the length of 100, 200, and 300, and the max_tokens_per_gpu is 300, when enabling "
                "dynamic batch size, miles will make 2 micro batches, i.e. [100, 200], [300]."
            )
        ),
    ] = False
    max_tokens_per_gpu: A[
        int | None,
        Arg(
            help=(
                "The maximum number of tokens per GPU for dynamic batch size. "
                "Note that when enabling context parallel (CP), the max tokens per gpu should be around "
                "`max_response_len // cp_size` instead of `max_response_len`."
            )
        ),
    ] = None
    log_probs_max_tokens_per_gpu: A[
        int | None,
        Arg(
            help=(
                "The maximum number of tokens per GPU for calculating log probs. "
                "This is used to calculate the log probs of the responses during rollout, "
                "and should be set to a larger value than `max_tokens_per_gpu` if you want better performance. "
            )
        ),
    ] = None
