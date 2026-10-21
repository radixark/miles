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
    # Arbitrary until measured.
    output_store_local_buffer_size: A[
        str,
        Arg(
            help=(
                "Local buffer of every output store client Miles creates (SGLang writers, trainer readers, the "
                "deleting clients), in bytes or as a string such as '1gb'. An engine puts all stored arrays of a "
                "response through it at once, so it must hold one response's routed experts, indexer top-k and "
                "sampling mask together."
            )
        ),
    ] = "1gb"
