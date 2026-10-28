import os
import shlex

from miles.utils.external_utils.command_utils.base_backend import (
    CLUSTER_BACKEND_FLAG,
    BaseCommandBackend,
    ExecuteTrainConfig,
    ExecuteTrainRequest,
    LaunchGuard,
)
from miles.utils.external_utils.command_utils.common import (
    ArgvManipulator,
    _pythonpath_with_sources,
    get_mooncake_master_port,
    run_shell_command,
    train_env_vars,
)
from miles.utils.external_utils.command_utils.ray_backend.command import start_mooncake_master
from miles.utils.external_utils.model_args_utils import shell_safe_model_args
from miles.utils.workers.types import ClusterBackend


class MonarchCommandBackend(BaseCommandBackend):
    """Runs the driver directly, as the Monarch client that spawns the run's worker processes itself.

    Unlike the Ray backend, nothing has to run before the driver: there is no ``ray start`` to bring up a cluster and
    no ``ray job submit`` to hand it the driver. The driver's own process starts Monarch on its host (``this_host()``)
    and spawns every worker there.
    """

    def _execute_train_inner(
        self, *, request: ExecuteTrainRequest, config: ExecuteTrainConfig, guard: LaunchGuard | None
    ) -> None:
        """Clean up an earlier run on this host, start the Mooncake master, run the prepare commands, then run the
        driver (``train.py``) in the foreground with ``--cluster-backend monarch``."""
        assert not request.extra_manifests, (
            "extra_manifests are objects a helm release installs beside the run, and a monarch launch installs no "
            "release; launch onto kubernetes, or start what they describe yourself"
        )
        assert config.num_nodes == 1, f"a monarch run spawns its workers on one host, not {config.num_nodes} nodes"
        master_addr = os.environ.get("MASTER_ADDR") or "127.0.0.1"
        mooncake_master_port = get_mooncake_master_port(shlex.split(request.train_args))

        self._clean_up_previous_run()
        if not config.external_mooncake:
            start_mooncake_master(rpc_port=mooncake_master_port, run_command=self.exec_command_cpu)
        # A prepare command readies each node for the trainer, e.g. copies a checkpoint to node-local disk, so it runs
        # on every node of the run, as under the Ray backend.
        for cmd in request.prepare_cmd.values():
            self.exec_command_multi_node(cmd)
        if (f := request.before_ray_job_submit) is not None:
            f()

        backend_env_vars = {"no_proxy": f"127.0.0.1,{master_addr}", "MASTER_ADDR": master_addr}
        env_vars = train_env_vars(request, backend_env_vars, config=config)
        env_vars["PYTHONPATH"] = _pythonpath_with_sources(request.megatron_path, env_vars.get("PYTHONPATH"))
        model_args = shell_safe_model_args(request.megatron_model_type)
        train_args = request.train_args
        # train.py defaults to --cluster-backend ray, whose cluster this launch never starts
        if not ArgvManipulator.is_defined(shlex.split(train_args), CLUSTER_BACKEND_FLAG):
            train_args = f"{train_args} {CLUSTER_BACKEND_FLAG} {ClusterBackend.MONARCH.value}"
        exports = " ".join(f"{name}={shlex.quote(value)}" for name, value in env_vars.items())
        self.exec_command_cpu(f"env {exports} python3 {shlex.quote(request.train_script)} {model_args} {train_args}")

    def _exec_command_gpu_inner(
        self, cmd: str, capture_output: bool = False, num_gpus_per_node: int | None = None
    ) -> str | None:
        # num_gpus_per_node is a resource request, which only a backend that schedules the command somewhere honours
        # (kubernetes sizes its command pod by it). Here, as under Ray, the command runs on a host the run already
        # holds and sees all of its gpus, so there is nothing to request.
        return run_shell_command(cmd, capture_output=capture_output)

    def _exec_command_multi_node_inner(
        self,
        cmd: str,
        capture_output: bool = False,
        num_nodes: int | None = None,
        num_gpus_per_node: int | None = None,
    ) -> list[str | None]:
        # The run has the driver's host only, which the command runs on with all of its gpus; num_gpus_per_node
        # requests nothing here, as in _exec_command_gpu_inner.
        return [run_shell_command(cmd, capture_output=capture_output)]

    def _clean_up_previous_run(self) -> None:
        """Best effort: kill what an earlier run on this host may have left behind, so this run can take its gpus and
        its Mooncake master port.

        It kills by name the SGLang engines, Miles' own processes and the Mooncake master, as the Ray backend's sweep
        does (which also stops ray). Anything else an earlier run started, or a process under another name, is left
        running.
        """
        self.exec_command_cpu("pkill -9 sglang; sleep 3; pkill -9 miles; pkill -9 mooncake_master; true")
