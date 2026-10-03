import os

from pydantic import Field

from miles.utils.args.schema import A, Arg, BaseConfig


# prometheus
class PrometheusConfig(BaseConfig):
    use_prometheus: A[bool, Arg()] = False
    prometheus_port: A[
        int,
        Arg(
            help=(
                "Port for the Prometheus metrics HTTP server. "
                "Prometheus scrapes /metrics on this port. "
                "Defaults to PROMETHEUS_PORT env var or 9090."
            )
        ),
    ] = Field(default_factory=lambda: int(os.environ.get("PROMETHEUS_PORT", "9090")))
    prometheus_run_name: A[
        str | None,
        Arg(
            help=(
                "Human-readable run name attached as a 'run_name' label to all "
                "Prometheus metrics. Used to distinguish runs in Grafana. "
                "Defaults to --wandb-group if set."
            )
        ),
    ] = None
