from miles.utils.args.schema import A, Arg, BaseConfig


class SglangClientConfig(BaseConfig):
    sglang_server_concurrency: A[int, Arg()] = 512
