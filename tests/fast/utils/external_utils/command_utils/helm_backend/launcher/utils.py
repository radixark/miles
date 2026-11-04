from miles.backends.sglang_utils.sglang_scaling_config import SglangScalingConfig
from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.workers.argv_utils import ORCHESTRATOR_CONFIG_FLAG
from miles.utils.workers.connection_config import StaticConnConfig
from miles.utils.workers.serving.worker_config import OrchestratorWorkerConfig


class LauncherArgs(ScalingConfig):
    sglang_scaling: SglangScalingConfig = SglangScalingConfig(groups={})
    cluster_backend: str = "ray"
    colocate: bool = False
    deploy_component: str = "all"
    deploy_instance_id: str | None = None
    argv: list[str] = []
    train_env_vars: dict[str, str] = {}
    use_wandb: bool = False
    wandb_run_id: str | None = None


def launcher_args_orchestrator_command(
    train_script: str, *, args: LauncherArgs, static_connections: StaticConnConfig
) -> list[str]:
    payload = OrchestratorWorkerConfig(args=args.model_dump(mode="json"), static_connections=static_connections)
    return ["python", train_script, ORCHESTRATOR_CONFIG_FLAG, payload.model_dump_json()]
