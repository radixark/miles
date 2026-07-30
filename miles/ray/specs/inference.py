import logging
import shlex
import sys

from miles.backends.sglang_utils.router_args_utils import compute_sglang_router_args, router_args_to_argv
from miles.backends.sglang_utils.sglang_config import ModelConfig, resolve_sglang_config
from miles.router.config import compute_miles_router_config
from miles.utils.workers.argv_utils import config_to_argv
from miles.utils.workers.worker_spec import CommandWorkerSpec, LaunchCommandContext, PortInfo, SchedulingSpec

logger = logging.getLogger(__name__)


def specs_router(args) -> list[CommandWorkerSpec]:
    config = resolve_sglang_config(args)  # TODO avoid resolve repeatedly
    return [
        _compute_spec_router(args, model_idx=model_idx, model_cfg=model_cfg)
        for model_idx, model_cfg in enumerate(config.models)
    ]


def compute_router_pool_id(model_idx: int) -> str:
    return f"inference-router-{model_idx}"


def _compute_spec_router(args, model_idx: int, model_cfg: ModelConfig) -> CommandWorkerSpec:
    def _compute_launch_command(ctx: LaunchCommandContext) -> str:
        router_port = ctx.ports["primary"]

        if args.use_miles_router:
            assert not model_cfg.has_pd_disaggregation, "miles router does not support PD disaggregation."
            router_config = compute_miles_router_config(args, host=ctx.host, port=router_port)
            launch_argv = [sys.executable, "-m", "miles.router.router", *config_to_argv(router_config)]
        else:
            router_args = compute_sglang_router_args(
                args,
                host=ctx.host,
                port=router_port,
                prometheus_port=ctx.ports["prometheus"],
                has_pd_disaggregation=model_cfg.has_pd_disaggregation,
            )
            logger.info(f"Launch router with args: {router_args}")
            launch_argv = [sys.executable, "-m", "sglang_router.launch_router", *router_args_to_argv(router_args)]

        return shlex.join(launch_argv)

    return CommandWorkerSpec(
        name=compute_router_pool_id(model_idx),
        port_infos=[
            _compute_router_primary_port_info(args, model_idx=model_idx),
            PortInfo(name="prometheus", static_port=9000, allow_dynamic=True),
        ],
        env_var=lambda: {},
        scheduling=SchedulingSpec.single(num_gpus_per_worker=0),
        launch_command=_compute_launch_command,
    )


def _compute_router_primary_port_info(args, model_idx: int) -> PortInfo:
    if args.sglang_router_port is None:
        return PortInfo(name="primary", static_port=8000, allow_dynamic=True)
    return PortInfo(name="primary", static_port=args.sglang_router_port + model_idx)


def specs_session_server(args) -> list[CommandWorkerSpec]:
    _config = resolve_sglang_config(args)  # TODO avoid resolve repeatedly
    return None  # TODO return real objects


def specs_inference_engine(args) -> list[CommandWorkerSpec]:
    _config = resolve_sglang_config(args)  # TODO avoid resolve repeatedly
    return None  # TODO return real objects
