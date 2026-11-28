import argparse

from miles.utils.args.schema import A, Arg, BaseConfig


class NetworkConfig(BaseConfig):
    http_proxy: A[str | None, Arg()] = None
    use_distributed_post: A[
        bool | None,
        Arg(
            action=argparse.BooleanOptionalAction,
            help=(
                "Send the rollout's HTTP requests through --num-gpus-per-node Ray actors per node, each with its own "
                "connection pool, instead of one pool in the rollout executor. One pool holding every in-flight "
                "generation stream is rescanned by httpcore on every response (O(streams) per event), which "
                "serializes the executor once a step has about a thousand concurrent samples. Defaults to true under "
                "--cluster-backend ray; an explicit --no-use-distributed-post keeps the single local pool. "
                "--cluster-backend kubernetes has no Ray cluster to place the actors on and runs without them."
            ),
        ),
    ] = None
