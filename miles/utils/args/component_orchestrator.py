from miles.utils.args.schema import A, Arg, BaseConfig


class OrchestratorOnlyConfig(BaseConfig):
    load: A[str | None, Arg(reset=True)] = None
    requested_load: str | None
