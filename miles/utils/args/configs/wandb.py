from typing import ClassVar

from miles.utils.args.schema import A, Arg, BaseConfig


# wandb
class WandbConfig(BaseConfig):
    _mutable_fields: ClassVar[frozenset[str]] = frozenset({"wandb_run_id"})

    # wandb parameters
    use_wandb: A[bool, Arg()] = False
    wandb_mode: A[
        str | None,
        Arg(
            choices=["online", "offline", "disabled"],
            help="W&B mode: online (default), offline (local only), or disabled. Overrides WANDB_MODE env var.",
        ),
    ] = None
    wandb_dir: A[
        str | None,
        Arg(help="Directory to store wandb logs. Default is ./wandb in current directory."),
    ] = None
    wandb_key: A[str | None, Arg()] = None
    wandb_host: A[str | None, Arg()] = None
    wandb_team: A[str | None, Arg()] = None
    wandb_group: A[str | None, Arg()] = None
    wandb_random_suffix: A[
        bool,
        Arg(
            cli_name="--disable-wandb-random-suffix",
            action="store_false",
            help=(
                "Whether to add a random suffix to the wandb run name. "
                "By default, we will add a random 6 length string with characters to the run name."
            ),
        ),
    ] = True
    wandb_always_use_train_step: A[
        bool,
        Arg(
            help=(
                "Whether to always use train step as the step metric in wandb. "
                "If set, we will always use the train steps for wandb logging, "
                "otherwise, will use rollout step for most info other than train/*. "
            )
        ),
    ] = False
    log_multi_turn: A[
        bool,
        Arg(help="Whether to log information for multi-turn rollout."),
    ] = False
    log_reward_category: A[
        str | None,
        Arg(
            help=(
                "Log statistics of the category of reward, such as why the reward function considers it as failed. "
                "Specify the key in the reward dict using this argument."
            )
        ),
    ] = None
    log_correct_samples: A[
        bool,
        Arg(help="Explicitly log metrics for correct samples."),
    ] = False
    wandb_run_id: A[str | None, Arg()] = None


class WandbRolloutOnlyConfig(BaseConfig):
    log_passrate: A[
        bool,
        Arg(help="Whether to turn on passrate logging, which will log the pass@n of the responses in the rollout."),
    ] = False
