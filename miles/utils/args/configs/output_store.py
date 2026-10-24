from miles.utils.args.schema import A, Arg, BaseConfig


class OutputStoreConfig(BaseConfig):
    use_output_store: A[
        bool,
        Arg(
            help=(
                "Have SGLang put the routed experts, indexer top-k and sampling mask of training responses in the "
                "run's Mooncake and return a ref instead of the arrays; the trainer reads them back, and the "
                "objects are deleted once nothing can read them. Requires --object-store-backend mooncake."
            )
        ),
    ] = False
    # Fits a 32k-token DeepSeek-V3.2 response with indexer top-k (about 15.3 GiB).
    output_store_local_buffer_size: A[
        str,
        Arg(
            help=(
                "Local buffer of every SGLang engine and train worker writing or reading the output store, in "
                "bytes or as a string such as '16gb'. An engine puts all stored arrays of a response through it at "
                "once, so it must hold one response's routed experts, indexer top-k and sampling mask together. "
                "With RDMA, each of these clients pins the whole buffer in host memory when it starts."
            )
        ),
    ] = "16gb"
