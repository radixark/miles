from miles.utils.args.schema import A, Arg, BaseConfig


class MtpTrainingConfig(BaseConfig):
    """Add MTP training specific arguments."""

    enable_mtp_training: A[
        bool,
        Arg(
            help=(
                "Build and train the model's --mtp-num-layers MTP layers alongside the policy (megatron only). "
                "Without it no MTP layer is built, so exported checkpoints carry no MTP weights."
            )
        ),
    ] = False
