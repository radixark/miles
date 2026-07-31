import logging
import os
import shlex
import sys

from miles.backends.sglang_utils.router_args_utils import compute_sglang_router_args, router_args_to_argv
from miles.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig, resolve_sglang_config
from miles.backends.sglang_utils.sglang_engine import compute_engine_launch_cmd
from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from miles.rollout.session.config import compute_session_server_config
from miles.router.config import compute_miles_router_config
from miles.utils import dumper_utils
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
        primary = ctx.self_addrs["primary"]

        if args.use_miles_router:
            assert not model_cfg.has_pd_disaggregation, "miles router does not support PD disaggregation."
            router_config = compute_miles_router_config(args, host=primary.host, port=primary.port)
            launch_argv = [sys.executable, "-m", "miles.router.router", *config_to_argv(router_config)]
        else:
            router_args = compute_sglang_router_args(
                args,
                host=primary.host,
                port=primary.port,
                prometheus_port=ctx.self_addrs["prometheus"].port,
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
        scheduling=SchedulingSpec.single(
            num_gpus_per_worker=0,
            # TODO: refactor the flag
            pin_to_head=args.pin_rollout_manager_to_head,
        ),
        launch_command=_compute_launch_command,
    )


def _compute_router_primary_port_info(args, model_idx: int) -> PortInfo:
    if args.sglang_router_port is None:
        return PortInfo(name="primary", static_port=8000, allow_dynamic=True)
    return PortInfo(name="primary", static_port=args.sglang_router_port + model_idx)


def spec_session_server(args) -> CommandWorkerSpec:
    _config = resolve_sglang_config(args)  # TODO avoid resolve repeatedly

    def _compute_launch_command(ctx: LaunchCommandContext) -> str:
        config = compute_session_server_config(
            args,
            host=args.session_server_ip or ctx.self_addrs["primary"].host,
            port=ctx.self_addrs["primary"].port,
            # TODO: make the indexing it k8s native compatible
            instance_id=compute_session_server_instance_id(args, ctx.cell_index),
            backend_url=ctx.spec_addrs[compute_router_pool_id(0)][0]["primary"].addr,
        )
        launch_argv = [sys.executable, "-m", "miles.rollout.session.server", *config_to_argv(config)]
        return shlex.join(launch_argv)

    return CommandWorkerSpec(
        name="session-server",
        port_infos=[
            _compute_session_server_primary_port_info(args),
        ],
        env_var=lambda: {},
        scheduling=SchedulingSpec(
            num_cells=args.num_session_servers if args.use_session_server else 0,
            num_workers_per_cell=1,
            num_gpus_per_worker=0,
            pin_to_head=args.pin_rollout_manager_to_head,
        ),
        launch_command=_compute_launch_command,
    )


def _compute_session_server_primary_port_info(args) -> PortInfo:
    if args.session_server_port is None:
        return PortInfo(name="primary", static_port=8000, allow_dynamic=True)
    return PortInfo(name="primary", static_port=args.session_server_port, offset_by_cell=True)


def compute_session_server_instance_id(args, instance_index: int) -> str:
    return f"{args.run_uuid}-{instance_index}"


def specs_inference_engine(args) -> list[CommandWorkerSpec]:
    config = resolve_sglang_config(args)  # TODO avoid resolve repeatedly
    return [
        _compute_spec_inference_engine(args, model_idx=model_idx, server_group_config=server_group_config)
        for model_idx, model_cfg in enumerate(config.models)
        for server_group_config in model_cfg.server_groups
    ]


def _compute_spec_inference_engine(
    args,
    model_idx: int,
    server_group_config: ServerGroupConfig,
) -> CommandWorkerSpec:
    def _compute_launch_command(ctx: LaunchCommandContext) -> str:
        dist_init = ctx.self_addrs["dist_init"]
        return compute_engine_launch_cmd(
            args=args,
            # TODO: make the indexing it k8s native compatible
            node_rank=ctx.worker_in_cell_index,
            worker_type=server_group_config.worker_type,
            base_gpu_id=ctx.gpu_ids[0],
            sglang_overrides=server_group_config.overrides,
            num_gpus_per_engine=server_group_config.num_gpus_per_engine,
            dist_init_addr=f"{dist_init.host}:{dist_init.port}",
            nccl_port=ctx.self_addrs["nccl"].port,
            host=ctx.self_addrs["primary"].host,
            port=ctx.self_addrs["primary"].port,
            disaggregation_bootstrap_port=d.port if (d := ctx.self_addrs.get("disaggregation_bootstrap")) else None,
            engine_info_bootstrap_port=ctx.self_addrs["engine_info_bootstrap"].port,
        )

    envs = compute_inference_engine_env_vars(args)
    return CommandWorkerSpec(
        name=f"inference-engine-{model_idx}",
        port_infos=[
            PortInfo(name="primary", static_port=8000, allow_dynamic=True),
            PortInfo(name="dist_init", static_port=9000, mode="master", allow_dynamic=True),
            PortInfo(name="nccl", static_port=10000, allow_dynamic=True),
            PortInfo(name="disaggregation_bootstrap", static_port=11000, allow_dynamic=True),
            PortInfo(name="engine_info_bootstrap", static_port=12000, allow_dynamic=True),
        ],
        env_var=lambda: envs,
        scheduling=SchedulingSpec(
            num_cells=server_group_config.num_gpus // server_group_config.num_gpus_per_engine,
            num_workers_per_cell=max(1, server_group_config.num_gpus_per_engine // args.num_gpus_per_node),
            # TODO: may need real num for k8s native mode
            num_gpus_per_worker=0.2,
        ),
        launch_command=_compute_launch_command,
    )


def compute_inference_engine_env_vars(args) -> dict[str, str]:
    env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
        key: os.environ.get(key, default_val)
        for key, default_val in {
            # DeepEP/NVSHMEM's internal NCCL conflicts with our NCCL and hangs under CUDA graphs.
            "NVSHMEM_DISABLE_NCCL": "1",
            "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
            "SGLANG_DG_CACHE_DIR_PER_PROCESS": "1",
            "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
            "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
            "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2": (
                "0" if args.colocate and args.rollout_num_gpus_per_engine > 1 else "1"
            ),
            "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
            "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
            "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
        }.items()
    }
    env_vars.update(dumper_utils.get_sglang_env(args))
    return env_vars
