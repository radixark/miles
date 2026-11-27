from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.args.schema import A, Arg, BaseConfig


class RewardModelConfig(BaseConfig):
    rm_type: A[str | None, Arg(help="Type of the reward model")] = None
    reward_key: A[
        str | None,
        Arg(
            help=(
                "Some reward model may return a dict instead of a value, "
                "this is the key to extract the reward value from the dict. "
            )
        ),
    ] = None
    custom_rm_path: A[
        CustomFunctionConfig | None,
        Arg(
            help=(
                "Path to the custom reward model function. "
                "If set, we will use this function to calculate the reward instead of the default one. "
                "The function should have the signature `def custom_rm(args, sample) -> float`."
            ),
        ),
    ] = None


class RewardModelRolloutOnlyConfig(BaseConfig):
    length_penalty_coeff: A[
        float,
        Arg(help="Non-negative exponent for correct-trajectory length penalties; zero disables shaping."),
    ] = 0.0
    length_penalty_min_tokens: A[
        int,
        Arg(help="Positive floor on the shortest correct trajectory's generated-token count."),
    ] = 1
    eval_reward_key: A[str | None, Arg(help="The eval variant for --reward-key")] = None
    group_rm: A[bool, Arg(help="Whether to do rm on a whole group.")] = False
    reward_pool_config: A[
        str | None,
        Arg(
            help=(
                "YAML list of reward pools to start: each entry sets a pool's `name`, `scorer_path`, `scorer_kwargs`, "
                "`num_workers`, `uses_gpu`, `batch_size` and `max_wait_ms`. A GPU pool's workers each take one GPU "
                "after the rollout and eval GPUs, and a reward function reaches the pool as its `<name>_pool` keyword "
                "argument, `await pool.score(outputs, prompts)`."
            )
        ),
    ] = None
    rm_url: A[
        str | None,
        Arg(help="URL for the reward model service for --rm-type remote_rm, e.g. http://localhost:8000"),
    ] = None
    custom_reward_post_process_path: A[
        CustomFunctionConfig | None,
        Arg(
            help="Path to the custom function that will post process reward, by default it will be the normalization for grpo. ",
        ),
    ] = None
    custom_convert_samples_to_train_data_path: A[
        CustomFunctionConfig | None,
        Arg(
            help=(
                "Path to a custom function that converts samples to training data. "
                "If set, this function will replace the default _convert_samples_to_train_data. "
                "The function should have the signature `def convert_samples_to_train_data(args, samples) -> dict`."
            ),
        ),
    ] = None
