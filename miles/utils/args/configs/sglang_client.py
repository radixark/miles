from miles.utils.args.schema import A, Arg, BaseConfig


class SglangClientConfig(BaseConfig):
    sglang_server_concurrency: A[int, Arg()] = 512
    sglang_pp_layer_partition: A[
        str | None,
        Arg(
            help=(
                "Layers per pipeline stage of each SGLang engine, comma-separated (e.g. 10,8,12,8): one entry per "
                "--sglang-pp-size stage, summing to the model's layer count. Unset, SGLang splits the layers evenly. "
                "The engines receive it as SGLANG_PP_LAYER_PARTITION."
            )
        ),
    ] = None
