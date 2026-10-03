from miles.utils.args.schema import A, Arg, BaseConfig


class PrefillDecodeDisaggregationConfig(BaseConfig):
    prefill_num_servers: A[int | None, Arg(help="Number of prefill servers for disaggregation.")] = None
