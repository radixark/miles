from miles.utils.args.schema import A, Arg, BaseConfig


class MtpTrainingConfig(BaseConfig):
    """Add MTP training specific arguments."""

    enable_mtp_training: A[bool, Arg(help="Enable MTP layer parameter updates during training")] = False
