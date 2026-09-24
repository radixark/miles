import argparse

from miles.utils.args.schema import A, Arg, BaseConfig


class TinkerConfig(BaseConfig):
    tinker_server_host: A[str, Arg()] = "0.0.0.0"
    tinker_server_port: A[int, Arg()] = 10613
    tinker_base_model: A[
        str | None,
        Arg(help="Model name advertised by the gateway (default: --hf-checkpoint)"),
    ] = None
    tinker_checkpoint_root: A[
        str | None,
        Arg(help="Directory for tinker:// checkpoints (default: <save>/tinker)"),
    ] = None
    tinker_train_attn: A[bool, Arg(action=argparse.BooleanOptionalAction)] = True
    tinker_train_mlp: A[bool, Arg(action=argparse.BooleanOptionalAction)] = True
    tinker_train_unembed: A[bool, Arg(action=argparse.BooleanOptionalAction)] = True
