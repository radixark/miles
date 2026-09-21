import argparse

from miles.backends.fsdp_utils.config import FsdpArgsNamespace
from miles.backends.megatron_utils.megatron_config import MegatronConfig
from miles.utils.args.schema import A, Arg, BaseConfig, reset_arg


# TODO: Unify trainer descriptions after zhichen's training backend refactor
class RawTrainerBackendConfig(BaseConfig):
    raw_megatron: MegatronConfig
    raw_fsdp: FsdpArgsNamespace | None


class TrainerBackendTraitConfig(BaseConfig):
    # from AlgoConfig
    ckpt_step: int | None
    load: A[str | None, Arg(reset=True)] = None
    save: A[str | None, Arg(reset=True)] = None
    save_interval: A[int | None, Arg(reset=True)] = None
    async_save: A[bool, Arg(reset=True)]
    seed: A[int, Arg(reset=True)] = 1234

    # from TrainConfig
    num_layers: int | None

    # from ClusterConfig
    distributed_backend: A[str, Arg(reset=True)] = "nccl"
    distributed_timeout_minutes: A[int, Arg(reset=True)] = 10

    # from EvalConfig
    # change the default value of eval_interval from Megatron to None
    eval_interval: A[int | None, Arg(reset=True)] = None

    # from DataConfig
    # gbs of the training, note that the gbs is of sample, not of prompts,
    # so if you hope to train 1 step for each rollout, the global_bach_size should be set as
    # `rollout_batch_size * n_samples_per_prompt`.
    global_batch_size: A[int | None, Arg(reset=True)] = None
    # mbs for the training, will be ignored if `use_dynamic_batch_size` is set.
    micro_batch_size: A[int, Arg(reset=True)] = 1

    # from WandbConfig
    wandb_project: A[str | None, Arg(reset=True)] = None

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser=parser)
        reset_arg(parser=parser, name="--lr", type=float, default=1e-6)
        reset_arg(parser=parser, name="--clip-grad", type=float, default=1.0)
        reset_arg(parser=parser, name="--calculate-per-token-loss", action="store_true")
        reset_arg(
            parser=parser,
            name="--no-save-optim",
            action="store_true",
            default=False,
            help=(
                "If set, do not save the optimizer state when saving checkpoints. "
                "This reduces checkpoint size but disables training resumption from the saved checkpoint."
            ),
        )
