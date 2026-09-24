import logging
import os
import shlex
from typing import Any, ClassVar, Self

from miles.backends.sglang_utils.router_args_utils import compute_sglang_router_args, router_args_to_argv
from miles.backends.sglang_utils.sglang_api_client import WorkerType
from miles.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig
from miles.backends.sglang_utils.sglang_engine import compute_engine_launch_cmd
from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from miles.rollout.session.config import compute_session_server_config
from miles.router.config import compute_miles_router_config
from miles.utils.args.custom_view import compute_custom_function_config
from miles.utils.args.runtime import InferenceControllerConfig
from miles.utils.function_registry import load_function
from miles.utils.http_utils import resolve_ip
from miles.utils.workers.argv_utils import config_to_argv, python_argv_prefix
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.launch_gate import GATE_PORT_NAME
from miles.utils.workers.naming import compute_worker_name
from miles.utils.workers.registration.hub import RegistrationHub
from miles.utils.workers.registration.reporter import RegistrationReporter
from miles.utils.workers.types import DeployComponent, PlatformAccess
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_provider.static import StaticWorkerProvider, parse_host_and_port
from miles.utils.workers.worker_spec import (
    BaseCommandSpec,
    BaseServeSpec,
    LaunchCommandContext,
    PortInfo,
    SchedulingSpec,
    WorkerCtorContext,
    WorkerLaunchContext,
    WorkerMetaContext,
)

logger = logging.getLogger(__name__)

POOL_CATEGORY_INFERENCE_ENGINE = "inference_engine"

ENGINE_POOL_ID_PREFIX = "inference-engine"
INFERENCE_CONTROLLER_ADDR_FLAG = "--inference-controller-addr"
INFERENCE_CONTROLLER_POOL_ID = "inference-controller"
SESSION_SERVER_POOL_ID = "session-server"
INFERENCE_CONTROLLER_WORKER_CLASS = "miles.ray.rollout.inference_controller.InferenceController"
INFERENCE_REGISTRATION_REPORTER_POOL_ID = "inference-registration-reporter"
INFERENCE_REGISTRATION_REPORTER_WORKER_CLASS = "miles.utils.workers.registration.reporter.RegistrationReporterWorker"


class InferenceControllerSpec(BaseServeSpec):
    worker_type: ClassVar[str] = "inference_controller"
    config_class = InferenceControllerConfig
    args: InferenceControllerConfig
    name: str = INFERENCE_CONTROLLER_POOL_ID
    platform_access: PlatformAccess = PlatformAccess.READ
    worker_class: str = INFERENCE_CONTROLLER_WORKER_CLASS

    @classmethod
    def create(cls, config: InferenceControllerConfig) -> Self:
        return cls(
            args=config,
            scheduling=SchedulingSpec(
                num_cells=1,
                num_workers_per_cell=1,
                num_gpus_per_worker=0,
                num_cpus_per_worker=1,
                pin_to_head=config.pin_rollout_manager_to_head,
            ),
        )

    def ctor_kwargs(self, ctx: WorkerCtorContext) -> dict[str, Any]:
        return dict(
            args=ctx.args,
            engine_provider=_compute_controller_engine_provider(ctx.args, capability=ctx.capability),
            router_providers=compute_router_providers(ctx.args, capability=ctx.capability),
        )


class InferenceRegistrationReporterSpec(BaseServeSpec):
    worker_type: ClassVar[str] = "inference_registration_reporter"
    config_class = InferenceControllerConfig
    args: InferenceControllerConfig
    name: str = INFERENCE_REGISTRATION_REPORTER_POOL_ID
    deploy_component: DeployComponent = DeployComponent.INFERENCE
    platform_access: PlatformAccess = PlatformAccess.READ
    worker_class: str = INFERENCE_REGISTRATION_REPORTER_WORKER_CLASS

    @classmethod
    def slice_configs(cls, args: Any) -> list[InferenceControllerConfig]:
        if DeployComponent(args.deploy_component) is not DeployComponent.INFERENCE:
            return []
        return [InferenceControllerConfig.slice_from(args)]

    @classmethod
    def create(cls, config: InferenceControllerConfig) -> Self:
        return cls(
            args=config,
            scheduling=SchedulingSpec(
                num_cells=1,
                num_workers_per_cell=1,
                num_gpus_per_worker=0,
                num_cpus_per_worker=1,
                pin_to_head=config.pin_rollout_manager_to_head,
            ),
        )

    def ctor_kwargs(self, ctx: WorkerCtorContext) -> dict[str, Any]:
        return dict(
            args=ctx.args,
            reporter=_create_inference_registration_reporter(ctx.args, capability=ctx.capability),
        )


def _compute_controller_engine_provider(args, *, capability: BackendCapability) -> BaseWorkerProvider:
    if DeployComponent(args.deploy_component).deploys_own_inference_engines():
        return compute_engine_provider(args, capability=capability)
    return RegistrationHub(run_uuid=args.run_uuid)


def _create_inference_registration_reporter(args, *, capability: BackendCapability) -> RegistrationReporter:
    controller_provider = compute_inference_controller_provider(args, capability=capability)
    return RegistrationReporter(
        run_uuid=args.run_uuid,
        reporter_id=args.deploy_instance_id,
        hub_endpoint=controller_provider.get_handle(inference_controller_worker_name()),
        worker_provider=compute_engine_provider(args, capability=capability),
    )


def compute_engine_provider(args, *, capability: BackendCapability) -> BaseWorkerProvider:
    path = args.custom_inference_engine_provider_path
    fn = load_function(path)
    fn_args = compute_custom_function_config(args, path)
    return fn(fn_args, capability=capability)


def backend_inference_engine_provider(args, *, capability: BackendCapability) -> BaseWorkerProvider:
    return capability.dynamic_worker_provider(pool_ids=compute_engine_pool_ids(args))


def compute_router_providers(args, *, capability: BackendCapability) -> list[BaseWorkerProvider]:
    config = args.sglang
    return [
        capability.static_worker_provider(pool_id=compute_router_pool_id(model_idx))
        for model_idx in range(len(config.models))
    ]


def create_inference_controller_handle(*, capability: BackendCapability) -> BaseWorkerHandle:
    worker_name = inference_controller_worker_name()
    provider = capability.static_worker_provider(pool_id=INFERENCE_CONTROLLER_POOL_ID)
    return provider.get_handle(worker_name)


def compute_inference_controller_provider(args, *, capability: BackendCapability) -> BaseWorkerProvider:
    if (entry := args.inference_controller_addr) is not None:
        return StaticWorkerProvider.of_rpc_addrs(
            pool_id=INFERENCE_CONTROLLER_POOL_ID,
            addrs=[parse_host_and_port(entry)],
            worker_class=INFERENCE_CONTROLLER_WORKER_CLASS,
        )
    return capability.static_worker_provider(pool_id=INFERENCE_CONTROLLER_POOL_ID)


def session_server_worker_name(cell_index: int) -> str:
    return compute_worker_name(pool_id=SESSION_SERVER_POOL_ID, cell_index=cell_index)


def inference_controller_worker_name() -> str:
    return compute_worker_name(pool_id=INFERENCE_CONTROLLER_POOL_ID)


class RouterSpec(BaseCommandSpec):
    model_cfg: ModelConfig

    @classmethod
    def create(cls, config: Any) -> list[Self]:
        return [
            _compute_spec_router(config, model_idx=model_idx, model_cfg=model_cfg)
            for model_idx, model_cfg in enumerate(config.sglang.models)
        ]

    def launch_command(self, ctx: LaunchCommandContext) -> str:
        args = ctx.args
        model_cfg = self.model_cfg
        interpreter_prefix = python_argv_prefix()
        primary = ctx.self_addrs["primary"]

        has_pd_disaggregation = model_cfg.has_pd_disaggregation or args.rollout_external_router_pd

        if args.use_miles_router:
            assert not has_pd_disaggregation, "miles router does not support PD disaggregation."
            router_config = compute_miles_router_config(
                args, host=primary.host, port=primary.port, num_engines=model_cfg.num_server_cells
            )
            launch_argv = [*interpreter_prefix, "-m", "miles.router.router", *config_to_argv(router_config)]
        else:
            router_args = compute_sglang_router_args(
                args,
                host=resolve_ip(primary.host),
                port=primary.port,
                prometheus_port=ctx.self_addrs["prometheus"].port,
                has_pd_disaggregation=has_pd_disaggregation,
            )
            logger.info(f"Launch router with args: {router_args}")
            launch_argv = [
                *interpreter_prefix,
                "-m",
                "sglang_router.launch_router",
                *router_args_to_argv(router_args),
            ]

        return shlex.join(launch_argv)


def compute_router_pool_id(model_idx: int) -> str:
    return f"inference-router-{model_idx}"


def compute_router_worker_name(model_idx: int) -> str:
    return compute_worker_name(pool_id=compute_router_pool_id(model_idx))


def _compute_spec_router(args, model_idx: int, model_cfg: ModelConfig) -> RouterSpec:
    return RouterSpec(
        args=args,
        model_cfg=model_cfg,
        name=compute_router_pool_id(model_idx),
        port_infos=[
            _compute_router_primary_port_info(args, model_idx=model_idx),
            PortInfo(name="prometheus", static_port=9000, allow_dynamic=True),
        ],
        scheduling=SchedulingSpec.single(
            num_gpus_per_worker=0,
            # TODO: refactor the flag
            pin_to_head=args.pin_rollout_manager_to_head,
        ),
    )


def _compute_router_primary_port_info(args, model_idx: int) -> PortInfo:
    if args.sglang_router_port is None:
        return PortInfo(name="primary", static_port=8000, allow_dynamic=True)
    return PortInfo(name="primary", static_port=args.sglang_router_port + model_idx)


class SessionServerSpec(BaseCommandSpec):
    @classmethod
    def create(cls, config: Any) -> Self:
        return _compute_spec_session_server(config)

    def launch_command(self, ctx: LaunchCommandContext) -> str:
        args = ctx.args
        interpreter_prefix = python_argv_prefix()
        (router_addrs,) = ctx.pool_addrs[compute_router_pool_id(0)]
        session_config = compute_session_server_config(
            args,
            host=args.session_server_ip or ctx.self_addrs["primary"].host,
            port=ctx.self_addrs["primary"].port,
            # TODO: make the indexing it k8s native compatible
            instance_id=compute_session_server_instance_id(args, ctx.cell_index),
            backend_url=router_addrs["primary"].addr,
        )
        launch_argv = [*interpreter_prefix, "-m", "miles.rollout.session.server", *config_to_argv(session_config)]
        return shlex.join(launch_argv)


def _compute_spec_session_server(args: Any) -> SessionServerSpec:
    config = args.sglang  # TODO avoid resolve repeatedly

    return SessionServerSpec(
        args=args,
        name=SESSION_SERVER_POOL_ID,
        port_infos=[
            _compute_session_server_primary_port_info(args),
        ],
        scheduling=SchedulingSpec(
            num_cells=(args.session_server_workers if args.use_session_server and config.models else 0),
            num_workers_per_cell=1,
            num_gpus_per_worker=0,
            num_cpus_per_worker=0,
            pin_to_head=True,
        ),
    )


def _compute_session_server_primary_port_info(args) -> PortInfo:
    if args.session_server_port is None:
        return PortInfo(name="primary", static_port=8000, allow_dynamic=True)
    return PortInfo(name="primary", static_port=args.session_server_port, offset_by_cell=True)


def compute_session_server_instance_id(args, instance_index: int) -> str:
    return f"{args.run_uuid}-{instance_index}"


def compute_engine_pool_id(args, *, model_idx: int, group_index: int) -> str:
    segment = args.deploy_instance_id or DeployComponent(args.deploy_component).value
    return f"{ENGINE_POOL_ID_PREFIX}-{segment}-{model_idx}-{group_index}"


class InferenceEngineSpec(BaseCommandSpec):
    category: str = POOL_CATEGORY_INFERENCE_ENGINE
    deploy_component: DeployComponent = DeployComponent.INFERENCE
    model_cfg: ModelConfig
    server_group_config: ServerGroupConfig

    # TODO: reduce complexity around passing around configs later during arguments refactor
    def meta(self, ctx: WorkerMetaContext) -> dict[str, Any]:
        scheduling = self.scheduling
        server_group_config = self.server_group_config
        return dict(
            model_id=self.model_cfg.name,
            worker_type=server_group_config.worker_type.value,
            num_gpus_per_engine=server_group_config.num_gpus_per_engine,
            gpu_offset=server_group_config.gpu_offset
            + ctx.cell_index * scheduling.num_workers_per_cell * scheduling.num_gpu_slots_per_worker,
            sglang_api_key=self.args.sglang.get_value("api_key", group=server_group_config),
            needs_offload=server_group_config.needs_offload,
            update_weights=self.model_cfg.update_weights,
        )

    @classmethod
    def create(cls, config: Any) -> list[Self]:
        if config.rollout_external:
            return []
        return [
            _compute_spec_inference_engine(
                config,
                model_idx=model_idx,
                group_index=group_index,
                model_cfg=model_cfg,
                server_group_config=server_group_config,
            )
            for model_idx, model_cfg in enumerate(config.sglang.models)
            for group_index, server_group_config in enumerate(model_cfg.server_groups)
            if server_group_config.worker_type != WorkerType.PLACEHOLDER
        ]

    def env_var(self, ctx: WorkerLaunchContext) -> dict[str, str]:
        return compute_inference_engine_env_vars(ctx.args)

    def launch_command(self, ctx: LaunchCommandContext) -> str:
        args = ctx.args
        server_group_config = self.server_group_config
        num_workers_per_cell = self.scheduling.num_workers_per_cell
        interpreter_prefix = python_argv_prefix()
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
            interpreter_prefix=interpreter_prefix,
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


def compute_engine_pool_ids(args) -> list[str]:
    return [spec.name for spec in InferenceEngineSpec.create(args)]


def _compute_spec_inference_engine(
    args,
    model_idx: int,
    group_index: int,
    model_cfg: ModelConfig,
    server_group_config: ServerGroupConfig,
) -> InferenceEngineSpec:
    num_workers_per_cell = max(1, server_group_config.num_gpus_per_engine // args.num_gpus_per_node)

    num_gpus_per_engine = server_group_config.num_gpus_per_engine
    assert num_gpus_per_engine <= args.num_gpus_per_node or num_gpus_per_engine % args.num_gpus_per_node == 0, (
        f"group '{server_group_config.worker_type.value}' wants {num_gpus_per_engine=} which neither fits in one node of "
        f"{args.num_gpus_per_node} gpus nor tiles whole nodes, so its ranks would never all be launched"
    )

    scheduling = SchedulingSpec(
        num_cells=server_group_config.num_gpus // server_group_config.num_gpus_per_engine,
        num_workers_per_cell=num_workers_per_cell,
        # TODO: may need real num for k8s native mode
        num_gpus_per_worker=0.2,
        num_gpu_slots_per_worker=min(server_group_config.num_gpus_per_engine, args.num_gpus_per_node),
        num_gpus_per_node=args.num_gpus_per_node,
        pg_name="rollout",
        pg_slot_offset=server_group_config.gpu_offset,
    )

    num_workers_total = server_group_config.num_gpus // scheduling.num_gpu_slots_per_worker
    assert num_workers_total % scheduling.num_workers_per_cell == 0, (
        f"group '{server_group_config.worker_type.value}' has {num_workers_total=} which is not a whole number of "
        f"{scheduling.num_workers_per_cell}-worker engines; the trailing engine would have no node to run its "
        f"remaining ranks"
    )

    return InferenceEngineSpec(
        args=args,
        model_cfg=model_cfg,
        server_group_config=server_group_config,
        name=compute_engine_pool_id(args, model_idx=model_idx, group_index=group_index),
        category=POOL_CATEGORY_INFERENCE_ENGINE,
        deploy_component=DeployComponent.INFERENCE,
        port_infos=[
            PortInfo(name="primary", static_port=8000, allow_dynamic=True),
            PortInfo(
                name="dist_init",
                static_port=9000,
                mode="master",
                allow_dynamic=True,
                num_consecutive=30 + args.sglang.get_value("dp_size", group=server_group_config),
            ),
            PortInfo(name="nccl", static_port=10000, allow_dynamic=True),
            *(
                [PortInfo(name="disaggregation_bootstrap", static_port=11000, allow_dynamic=True)]
                if server_group_config.worker_type == WorkerType.PREFILL
                else []
            ),
            PortInfo(name="engine_info_bootstrap", static_port=12000, allow_dynamic=True),
            PortInfo(name=GATE_PORT_NAME, static_port=13000, mode="master", allow_dynamic=True),
        ],
        scheduling=scheduling,
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
            "SGLANG_EXPOSE_OWN_ENV_VARS": "1",
        }.items()
    }
    if args.dumper_enable or args.dumper_inference:
        from miles.utils import dumper_utils

        env_vars.update(dumper_utils.get_sglang_env(args))
    return env_vars
