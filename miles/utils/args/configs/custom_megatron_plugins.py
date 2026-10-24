from argparse import ArgumentParser

from miles.utils.args.schema import A, Arg, BaseConfig


class CustomMegatronPluginsConfig(BaseConfig):
    """
    Add custom Megatron plugins arguments.
    This is a placeholder for any additional arguments that might be needed.
    """

    custom_megatron_init_path: A[str | None, Arg()] = None
    custom_megatron_before_log_prob_hook_path: A[str | None, Arg()] = None
    custom_megatron_before_train_step_hook_path: A[str | None, Arg()] = None

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> None:
        from miles_plugins.models.deepseek_v4.arguments import add_dsv4_arguments

        add_dsv4_arguments(parser)
        super().add_arguments(parser=parser)


class DsaMegatronPluginsConfig(BaseConfig):
    dsa_impl: A[str, Arg(choices=["miles", "megatron"])] = "miles"
    miles_dsa_topk_backend: A[
        str,
        Arg(
            choices=["canonical", "torch", "flashinfer"],
            help=(
                "DSA indexer top-k backend for both raw DSA implementations. 'canonical' is the exact top-k with "
                "ties at the k-th score broken toward the smaller key, like the rollout's top-k, returned in "
                "ascending key order; native Megatron DSA runs it as its exact 'torch' top-k."
            ),
        ),
    ] = "canonical"

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> None:
        super().add_arguments(parser=parser)
        if any(action.dest == "cp_comm_type" for action in parser._actions):
            parser.set_defaults(cp_comm_type=None)
