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
    eval_reward_key: A[str | None, Arg(help="The eval variant for --reward-key")] = None
    group_rm: A[bool, Arg(help="Whether to do rm on a whole group.")] = False
    rm_url: A[
        str | None,
        Arg(help="URL for the reward model service for --rm-type remote_rm, e.g. http://localhost:8000"),
    ] = None
    custom_rm_path: A[
        str | None,
        Arg(
            help=(
                "Path to the custom reward model function. "
                "If set, we will use this function to calculate the reward instead of the default one. "
                "The function should have the signature `def custom_rm(args, sample) -> float`."
            )
        ),
    ] = None
    custom_reward_post_process_path: A[
        str | None,
        Arg(
            help="Path to the custom function that will post process reward, by default it will be the normalization for grpo. "
        ),
    ] = None
    custom_convert_samples_to_train_data_path: A[
        str | None,
        Arg(
            help=(
                "Path to a custom function that converts samples to training data. "
                "If set, this function will replace the default _convert_samples_to_train_data. "
                "The function should have the signature `def convert_samples_to_train_data(args, samples) -> dict`."
            )
        ),
    ] = None
