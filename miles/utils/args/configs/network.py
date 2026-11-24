from miles.utils.args.schema import A, Arg, BaseConfig


class NetworkConfig(BaseConfig):
    http_proxy: A[str | None, Arg()] = None
    use_distributed_post: A[bool, Arg()] = False
