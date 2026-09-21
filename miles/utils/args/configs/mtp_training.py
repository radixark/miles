import argparse

from miles.utils.args.schema import A, Arg, BaseConfig, reset_arg


class MtpTrainingConfig(BaseConfig):
    """Add MTP training specific arguments."""

    enable_mtp_training: A[bool, Arg(help="Enable MTP layer parameter updates during training")] = False

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser=parser)
        reset_arg(parser=parser, name="--mtp-num-layers", type=int, default=None)
        reset_arg(parser=parser, name="--mtp-loss-scaling-factor", type=float, default=0.2)
