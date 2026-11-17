import os
import shlex
import socket
from concurrent.futures import ThreadPoolExecutor

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
    substitute_placeholders,
    train_env_vars,
)
from miles.utils.external_utils.command_utils.ray_backend.command import start_mooncake_master
from miles.utils.external_utils.model_args_utils import shell_safe_model_args
from miles.utils.http_utils import is_port_available
from miles.utils.misc import get_current_node_ip
from miles.utils.object_store_config import MOONCAKE_MASTER_ENV_VAR
from miles.utils.workers.monarch.env_vars import MONARCH_JOB_STATE_ENV_VAR, MONARCH_RUN_ENV_VARS_ENV_VAR
from miles.utils.workers.types import ClusterBackend


class MonarchCommandBackend(BaseCommandBackend):
    """Runs the driver directly, as the Monarch client that spawns the run's worker processes itself.

    Unlike the Ray backend, nothing is started or submitted here: there is no ``ray start`` to bring up a cluster and
    no ``ray job submit`` to hand it the driver.

    - A single-node run needs nothing beforehand: the driver's own process starts Monarch on its host
      (``this_host()``) and spawns every worker there.
    - A multi-node run expects to run inside a Monarch job already: a SLURM allocation that
      ``python -m miles.utils.workers.monarch.slurm`` submitted, which started a Monarch worker loop (the process that
      spawns processes on the driver's behalf) on every node, and runs this launcher as its batch script on the head
      node. The driver attaches to those worker loops as its worker hosts.

    The job has to exist before this launcher runs, because the launcher runs inside it, as its batch script; were
    the submission done here, the launcher would have to wait outside the allocation and then start itself again
    inside it.
    """

    def _execute_train_inner(
        self, *, request: ExecuteTrainRequest, config: ExecuteTrainConfig, guard: LaunchGuard | None
    ) -> None:
        """Check the run fits where it runs, clean up an earlier run on this host (in a SLURM job: check the Mooncake
        master port is free instead), start the Mooncake master, run the prepare commands on every node, then run
        the driver (``train.py``) in the foreground on this node, with ``--cluster-backend monarch``."""
        assert not request.extra_manifests, (
            "extra_manifests are objects a helm release installs beside the run, and a monarch launch installs no "
            "release; launch onto kubernetes, or start what they describe yourself"
        )
        assert config.num_nodes == 1 or _in_slurm_job(), (
            f"a {config.num_nodes}-node monarch run needs the hosts of a monarch job; submit it with "
            f"`python -m miles.utils.workers.monarch.slurm`"
        )
        if _in_slurm_job():
            job_num_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
            assert config.num_nodes == job_num_nodes, (
                f"the run asks for {config.num_nodes} nodes, but the monarch job it runs in holds {job_num_nodes}; "
                f"drop --num-nodes, or submit the job with --nodes {config.num_nodes}"
            )
        multi_node = config.num_nodes > 1
        # workers on other hosts reach the head node by its address, not by loopback
        master_addr = _head_node_addr() if multi_node else (os.environ.get("MASTER_ADDR") or "127.0.0.1")
        mooncake_master_port = get_mooncake_master_port(shlex.split(request.train_args))

        if _in_slurm_job():
            # A fresh allocation holds no earlier run of this launcher, only, on a shared node, other jobs' runs,
            # which neither the cleanup sweep may kill nor this run's Mooncake master may take over.
            # a multi-node run cannot name another port: its master address must be the head node's, which the
            # submission does not know yet, and a loopback one is refused
            assert config.external_mooncake or is_port_available(mooncake_master_port), (
                f"port {mooncake_master_port} of the job's head node is taken, likely by another job sharing the node; "
                + (
                    "submit without --no-exclusive"
                    if multi_node
                    else "pick another master port in --mooncake-store-init-kwargs"
                )
            )
        else:
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
        if multi_node and not config.external_mooncake and MOONCAKE_MASTER_ENV_VAR not in os.environ:
            # The default master address is loopback, which only the head node reaches; an exported MOONCAKE_MASTER
            # or an explicit --mooncake-store-init-kwargs still wins over this one.
            backend_env_vars[MOONCAKE_MASTER_ENV_VAR] = f"{master_addr}:{mooncake_master_port}"
        env_vars = train_env_vars(request, backend_env_vars, config=config)
        env_vars["PYTHONPATH"] = _pythonpath_with_sources(request.megatron_path, env_vars.get("PYTHONPATH"))
        env_vars[MONARCH_RUN_ENV_VARS_ENV_VAR] = ",".join(sorted(env_vars))
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
        # num_gpus_per_node requests nothing here, as in _exec_command_gpu_inner: in a job, every node's gpus are the
        # job's already, and srun --overlap shares them; outside one, the run has the driver's host only.
        if _in_slurm_job():
            return _exec_command_on_slurm_nodes(cmd, capture_output=capture_output, num_nodes=num_nodes)
        return [run_shell_command(cmd, capture_output=capture_output)]

    def _clean_up_previous_run(self) -> None:
        """Best effort: kill what an earlier run on this host may have left behind, so this run can take its gpus and
        its Mooncake master port.

        It kills by name the SGLang engines, Miles' own processes and the Mooncake master, as the Ray backend's sweep
        does (which also stops ray). Anything else an earlier run started, or a process under another name, is left
        running.
        """
        self.exec_command_cpu("pkill -9 sglang; sleep 3; pkill -9 miles; pkill -9 mooncake_master; true")


def _in_slurm_job() -> bool:
    """Whether this launcher runs on the head node of a monarch job that the slurm submitter started."""
    return bool(os.environ.get(MONARCH_JOB_STATE_ENV_VAR)) and bool(os.environ.get("SLURM_JOB_ID"))


def _head_node_addr() -> str:
    """The address the job's other nodes reach this one, its head node, by."""
    # TODO: get_current_node_ip asks Ray's private lookup, which off a Ray node probes the default route;
    # Monarch could offer its own node address instead, but that lookup already works, so no need yet.
    return os.environ.get("MASTER_ADDR") or get_current_node_ip()


def _exec_command_on_slurm_nodes(cmd: str, *, capture_output: bool, num_nodes: int | None) -> list[str | None]:
    """Run ``cmd`` on the job's nodes at once, one ``srun`` step each, and answer their outputs in node-rank order.

    The head node, this one, is rank 0. The command's ``{{node_rank}}``, ``{{nnodes}}``, ``{{master_addr}}`` and
    ``{{node_ip}}`` placeholders are filled in per node, as the Ray backend does. E.g. in a job on nodes
    [gpu-a, gpu-b] launched from gpu-b, ``echo {{node_rank}}`` runs as ``echo 0`` on gpu-b and ``echo 1`` on gpu-a.
    """
    nodelist = run_shell_command('scontrol show hostnames "$SLURM_JOB_NODELIST"', capture_output=True)
    assert nodelist is not None
    # rank 0 is this node, the job's head node, whatever the order slurm lists the job's nodes in
    head = os.environ.get("SLURMD_NODENAME")
    hostnames = sorted(nodelist.split(), key=lambda hostname: hostname != head)
    if num_nodes is not None:
        assert num_nodes <= len(hostnames), f"Requested {num_nodes} nodes but the job holds {len(hostnames)}."
        hostnames = hostnames[:num_nodes]
    node_ips = [_head_node_addr(), *(socket.gethostbyname(hostname) for hostname in hostnames[1:])]

    def run_on(rank: int) -> str | None:
        node_cmd = substitute_placeholders(
            cmd, node_rank=str(rank), nnodes=str(len(hostnames)), master_addr=node_ips[0], node_ip=node_ips[rank]
        )
        # --overlap shares the cpus and gpus the job's worker loops already hold
        return run_shell_command(
            f"srun --overlap --nodes=1 --ntasks=1 -w {shlex.quote(hostnames[rank])} "
            f"bash -c {shlex.quote(f'unset CUDA_VISIBLE_DEVICES; {node_cmd}')}",
            capture_output=capture_output,
        )

    with ThreadPoolExecutor(max_workers=len(hostnames)) as pool:
        return list(pool.map(run_on, range(len(hostnames))))
