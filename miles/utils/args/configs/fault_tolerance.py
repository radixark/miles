import argparse

from miles.utils.args.schema import A, Arg, BaseConfig
from miles.utils.ft_utils.health_checker import SimpleHealthCheckerConfig

_FT_CHOICES = ["rollout", "train"]
_DEFAULT_FT_API_SERVER_PORT = 18080


class FaultToleranceConfig(BaseConfig):
    trainer_heartbeat_checker_interval: float
    trainer_heartbeat_checker_timeout: float
    trainer_heartbeat_checker_first_wait: float
    trainer_heartbeat_checker_failure_threshold: int
    rollout_health_check_interval: float
    rollout_health_check_timeout: float
    rollout_health_check_first_wait: float
    rollout_health_check_failure_threshold: int

    use_fault_tolerance: A[
        bool,
        Arg(help="Enable fault tolerance. Use --ft-components to select which components."),
    ] = False
    ft_components: A[
        list[str] | None,
        Arg(
            type_parser=None,
            nargs="+",
            choices=_FT_CHOICES,
            help="FT components to enable (requires --use-fault-tolerance). "
            "Choices: rollout, train. Default when omitted: rollout.",
        ),
    ] = None
    api_server_host: A[
        str,
        Arg(
            help=(
                "Host the HTTP api server binds to. The default only serves the local mini "
                "fault-tolerance controller; set 0.0.0.0 to accept remote controllers."
            )
        ),
    ] = "127.0.0.1"
    api_server_port: A[
        int | None,
        Arg(
            help=(
                f"Port for HTTP api server. 0 = disabled. Left unset it is "
                f"{_DEFAULT_FT_API_SERVER_PORT} under --use-fault-tolerance and 0 otherwise, "
                f"because the mini fault-tolerance controller drives cells over this port."
            )
        ),
    ] = None
    mini_ft_controller_enable: A[
        bool | None,
        Arg(
            action=argparse.BooleanOptionalAction,
            help="Enable the mini fault-tolerance controller that auto-heals Fatal cells. "
            "Left unset it follows --ft-components and --api-server-port, which is what makes "
            "--use-fault-tolerance heal on its own.",
        ),
    ] = None
    mini_ft_controller_poll_interval: A[
        float,
        Arg(help="Interval in seconds between cell health polls."),
    ] = 10.0
    mini_ft_controller_resume_delay: A[
        float,
        Arg(help="Delay in seconds between suspending and resuming a cell during heal."),
    ] = 10.0

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser=parser)
        SimpleHealthCheckerConfig.add_arguments(
            parser,
            prefix="rollout-health-check",
            interval_default=30.0,
            timeout_default=30.0,
            first_wait_default=0.0,
            failure_threshold_default=1,
        )
        SimpleHealthCheckerConfig.add_arguments(parser, prefix="trainer-heartbeat-checker")
