import argparse

from miles.utils.args.schema import A, Arg, BaseConfig


class CustomMegatronPluginsConfig(BaseConfig):
    """
    Add custom Megatron plugins arguments.
    This is a placeholder for any additional arguments that might be needed.
    """

    freeze_indexer: A[bool, Arg()] = False
    custom_megatron_init_path: A[str | None, Arg()] = None
    custom_megatron_before_log_prob_hook_path: A[str | None, Arg()] = None
    custom_megatron_before_train_step_hook_path: A[str | None, Arg()] = None

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        from miles_plugins.models.deepseek_v4.arguments import add_dsv4_arguments

        add_dsv4_arguments(parser)
        super().add_arguments(parser=parser)
