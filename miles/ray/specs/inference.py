import logging
import os
import shlex

from miles.backends.sglang_utils.router_args_utils import compute_sglang_router_args, router_args_to_argv
from miles.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig, resolve_sglang_config
from miles.backends.sglang_utils.sglang_engine import compute_engine_launch_cmd
from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from miles.rollout.session.config import compute_session_server_config
from miles.router.config import compute_miles_router_config
from miles.utils import dumper_utils
from miles.utils.workers.argv_utils import config_to_argv, python_argv_prefix
from miles.utils.workers.launch_gate import GATE_PORT_NAME
from miles.utils.workers.naming import compute_worker_name
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_spec import (
    CommandWorkerSpec,
    LaunchCommandContext,
    PortInfo,
    SchedulingSpec,
    ServeWorkerSpec,
)

logger = logging.getLogger(__name__)

INFERENCE_CONTROLLER_POOL_ID = "inference-controller"
INFERENCE_CONTROLLER_WORKER_CLASS = "miles.ray.rollout.inference_controller.InferenceController"


def spec_inference_controller(args) -> ServeWorkerSpec:
    return ServeWorkerSpec(
        name=INFERENCE_CONTROLLER_POOL_ID,
        port_infos=[],
        env_var=lambda _ctx: {},
        scheduling=SchedulingSpec(
            num_cells=1,
            num_workers_per_cell=1,
            num_gpus_per_worker=0,
            num_cpus_per_worker=1,
            pin_to_head=args.pin_rollout_manager_to_head,
        ),
        worker_class=INFERENCE_CONTROLLER_WORKER_CLASS,
        ctor_kwargs=lambda _ctx: dict(args=args),
    )


def create_inference_controller_handle() -> BaseWorkerHandle:
    from miles.utils.workers.worker_provider.ray import RayWorkerProvider

    provider = RayWorkerProvider.create()  # TODO inject instance
    return provider.get_handle(inference_controller_worker_name())


def inference_controller_worker_name() -> str:
    return compute_worker_name(pool_id=INFERENCE_CONTROLLER_POOL_ID)


def specs_router(args) -> list[CommandWorkerSpec]:
    if args.debug_train_only:
        return []

    config = resolve_sglang_config(args)  # TODO avoid resolve repeatedly
    return [
        _compute_spec_router(args, model_idx=model_idx, model_cfg=model_cfg)
        for model_idx, model_cfg in enumerate(config.models)
    ]


def compute_router_pool_id(model_idx: int) -> str:
    return f"inference-router-{model_idx}"


def compute_router_worker_name(model_idx: int) -> str:
    return compute_worker_name(pool_id=compute_router_pool_id(model_idx))


def _compute_spec_router(args, model_idx: int, model_cfg: ModelConfig) -> CommandWorkerSpec:
    interpreter_prefix = python_argv_prefix()

    def _compute_launch_command(ctx: LaunchCommandContext) -> str:
        primary = ctx.self_addrs["primary"]

        if args.use_miles_router:
            assert not model_cfg.has_pd_disaggregation, "miles router does not support PD disaggregation."
            router_config = compute_miles_router_config(args, host=primary.host, port=primary.port)
            launch_argv = [*interpreter_prefix, "-m", "miles.router.router", *config_to_argv(router_config)]
        else:
            router_args = compute_sglang_router_args(
                args,
                host=primary.host,
                port=primary.port,
                prometheus_port=ctx.self_addrs["prometheus"].port,
                has_pd_disaggregation=model_cfg.has_pd_disaggregation,
            )
            logger.info(f"Launch router with args: {router_args}")
            launch_argv = [
                *interpreter_prefix,
                "-m",
                "sglang_router.launch_router",
                *router_args_to_argv(router_args),
            ]

        return shlex.join(launch_argv)

    return CommandWorkerSpec(
        name=compute_router_pool_id(model_idx),
        port_infos=[
            _compute_router_primary_port_info(args, model_idx=model_idx),
            PortInfo(name="prometheus", static_port=9000, allow_dynamic=True),
        ],
        env_var=lambda _ctx: {},
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
    config = resolve_sglang_config(args)  # TODO avoid resolve repeatedly
    interpreter_prefix = python_argv_prefix()

    def _compute_launch_command(ctx: LaunchCommandContext) -> str:
        config = compute_session_server_config(
            args,
            host=args.session_server_ip or ctx.self_addrs["primary"].host,
            port=ctx.self_addrs["primary"].port,
            # TODO: make the indexing it k8s native compatible
            instance_id=compute_session_server_instance_id(args, ctx.cell_index),
            backend_url=ctx.pool_addrs[compute_router_pool_id(0)][0]["primary"].addr,
        )
        launch_argv = [*interpreter_prefix, "-m", "miles.rollout.session.server", *config_to_argv(config)]
        return shlex.join(launch_argv)

    return CommandWorkerSpec(
        name="session-server",
        port_infos=[
            _compute_session_server_primary_port_info(args),
        ],
        env_var=lambda _ctx: {},
        scheduling=SchedulingSpec(
            num_cells=(args.session_server_workers if args.use_session_server and config.models else 0),
            num_workers_per_cell=1,
            num_gpus_per_worker=0,
            num_cpus_per_worker=0,
            pin_to_head=True,
        ),
        launch_command=_compute_launch_command,
    )


def _compute_session_server_primary_port_info(args) -> PortInfo:
    if args.session_server_port is None:
        return PortInfo(name="primary", static_port=8000, allow_dynamic=True)
    return PortInfo(name="primary", static_port=args.session_server_port, offset_by_cell=True)


def compute_session_server_instance_id(args, instance_index: int) -> str:
    return f"{args.run_uuid}-{instance_index}"


def compute_engine_pool_id(model_idx: int, group_index: int) -> str:
    return f"inference-engine-{model_idx}-{group_index}"


def specs_inference_engine(args) -> list[CommandWorkerSpec]:
    if args.debug_train_only:
        return []

    config = resolve_sglang_config(args)  # TODO avoid resolve repeatedly

    return [
        _compute_spec_inference_engine(
            args,
            model_idx=model_idx,
            group_index=group_index,
            model_cfg=model_cfg,
            server_group_config=server_group_config,
        )
        for model_idx, model_cfg in enumerate(config.models)
        for group_index, server_group_config in enumerate(model_cfg.server_groups)
        if server_group_config.worker_type != "placeholder"
    ]


def compute_engine_pool_ids(args) -> list[str]:
    return [spec.name for spec in specs_inference_engine(args)]


def _compute_spec_inference_engine(
    args,
    model_idx: int,
    group_index: int,
    model_cfg: ModelConfig,
    server_group_config: ServerGroupConfig,
) -> CommandWorkerSpec:
    num_workers_per_cell = max(1, server_group_config.num_gpus_per_engine // args.num_gpus_per_node)

    def _compute_launch_command(ctx: LaunchCommandContext) -> str:
        dist_init = ctx.self_addrs["dist_init"]
        # TODO: only node 0's seed is used by sglang; node != 0 should get node 0's number
        random_seed = (
            args.seed
            + server_group_config.engine_offset
            + ctx.cell_index * num_workers_per_cell
            + ctx.worker_in_cell_index
        )
        return compute_engine_launch_cmd(
            args=args,
            # TODO: make the indexing it k8s native compatible
            node_rank=ctx.worker_in_cell_index,
            worker_type=server_group_config.worker_type,
            base_gpu_id=ctx.local_gpu_ids[0],
            sglang_overrides=server_group_config.overrides,
            num_gpus_per_engine=server_group_config.num_gpus_per_engine,
            dist_init_addr=f"{dist_init.host}:{dist_init.port}",
            nccl_port=ctx.self_addrs["nccl"].port,
            host=ctx.self_addrs["primary"].host,
            port=ctx.self_addrs["primary"].port,
            disaggregation_bootstrap_port=d.port if (d := ctx.self_addrs.get("disaggregation_bootstrap")) else None,
            engine_info_bootstrap_port=ctx.self_addrs["engine_info_bootstrap"].port,
            gated_launch_port=ctx.self_addrs[GATE_PORT_NAME].port,
            random_seed=random_seed,
        )

    num_gpus_per_engine = server_group_config.num_gpus_per_engine
    assert num_gpus_per_engine <= args.num_gpus_per_node or num_gpus_per_engine % args.num_gpus_per_node == 0, (
        f"group '{server_group_config.worker_type}' wants {num_gpus_per_engine=} which neither fits in one node of "
        f"{args.num_gpus_per_node} gpus nor tiles whole nodes, so its ranks would never all be launched"
    )

    envs = compute_inference_engine_env_vars(args)
    scheduling = SchedulingSpec(
        num_cells=server_group_config.num_gpus // server_group_config.num_gpus_per_engine,
        num_workers_per_cell=num_workers_per_cell,
        # TODO: may need real num for k8s native mode
        num_gpus_per_worker=0.2,
        num_gpu_slots_per_worker=min(server_group_config.num_gpus_per_engine, args.num_gpus_per_node),
        pg_name="rollout",
        pg_slot_offset=server_group_config.gpu_offset,
    )

    num_workers_total = server_group_config.num_gpus // scheduling.num_gpu_slots_per_worker
    assert num_workers_total % scheduling.num_workers_per_cell == 0, (
        f"group '{server_group_config.worker_type}' has {num_workers_total=} which is not a whole number of "
        f"{scheduling.num_workers_per_cell}-worker engines; the trailing engine would have no node to run its "
        f"remaining ranks"
    )

    return CommandWorkerSpec(
        name=compute_engine_pool_id(model_idx=model_idx, group_index=group_index),
        port_infos=[
            PortInfo(name="primary", static_port=8000, allow_dynamic=True),
            PortInfo(
                name="dist_init",
                static_port=9000,
                mode="master",
                allow_dynamic=True,
                num_consecutive=30 + args.sglang_dp_size,
            ),
            PortInfo(name="nccl", static_port=10000, allow_dynamic=True),
            *(
                [PortInfo(name="disaggregation_bootstrap", static_port=11000, allow_dynamic=True)]
                if server_group_config.worker_type == "prefill"
                else []
            ),
            PortInfo(name="engine_info_bootstrap", static_port=12000, allow_dynamic=True),
            PortInfo(name=GATE_PORT_NAME, static_port=13000, mode="master", allow_dynamic=True),
        ],
        env_var=lambda _ctx: envs,
        scheduling=scheduling,
        launch_command=_compute_launch_command,
        # TODO: reduce complexity around passing around configs later during arguments refactor
        meta=lambda ctx: dict(
            model_id=model_cfg.name,
            worker_type=server_group_config.worker_type,
            num_gpus_per_engine=server_group_config.num_gpus_per_engine,
            gpu_offset=server_group_config.gpu_offset
            + ctx.cell_index * scheduling.num_workers_per_cell * scheduling.num_gpu_slots_per_worker,
            sglang_api_key=server_group_config.overrides.get("api_key", args.sglang_api_key),
            needs_offload=server_group_config.needs_offload,
            update_weights=model_cfg.update_weights,
        ),
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
