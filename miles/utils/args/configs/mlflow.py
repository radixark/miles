from typing import ClassVar

from miles.utils.args.schema import A, Arg, BaseConfig


# mlflow
class MlflowConfig(BaseConfig):
    _mutable_fields: ClassVar[frozenset[str]] = frozenset({"mlflow_run_id"})

    use_mlflow: A[bool, Arg()] = False
    mlflow_tracking_uri: A[
        str | None,
        Arg(help="MLflow tracking server URI. Defaults to MLFLOW_TRACKING_URI env var, or local mlruns/ directory."),
    ] = None
    mlflow_experiment_name: A[str, Arg(help="MLflow experiment name.")] = "miles"
    mlflow_run_name: A[str | None, Arg(help="MLflow run name. Defaults to --wandb-group if not set.")] = None
    mlflow_run_id: A[str | None, Arg()] = None
