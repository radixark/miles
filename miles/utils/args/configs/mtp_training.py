import argparse

from miles.utils.args.schema import A, Arg, BaseConfig, reset_arg


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

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser=parser)
        # The model's MTP layers; built only under --enable-mtp-training, since Megatron trains any it builds.
        reset_arg(parser=parser, name="--mtp-num-layers", type=int, default=None)
        reset_arg(parser=parser, name="--mtp-loss-scaling-factor", type=float, default=0.2)
        # Megatron's own TransformerConfig flag, defaulted on so MTP loss trains only the MTP layers.
        reset_arg(parser, "--mtp-detach-heads", action="store_true", default=True)
        parser.add_argument(
            "--no-mtp-detach-heads",
            dest="mtp_detach_heads",
            action="store_false",
            help="Let MTP loss gradients flow back into the main model's embedding, output layer and "
            "hidden states.",
        )
