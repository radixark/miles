import json
import os
import shlex

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from miles.utils.external_utils.command_utils import base_backend
from miles.utils.external_utils.command_utils.base_backend import BaseCommandBackend, ExecuteTrainRequest
from miles.utils.external_utils.command_utils.common import (
    _pythonpath_with_sources,
    build_train_env_vars,
    check_has_nvlink,
    get_bool_env_var,
    run_shell_command,
    substitute_placeholders,
)
from miles.utils.external_utils.model_args_utils import shell_safe_model_args
from miles.utils.misc import get_current_node_ip


class RayCommandBackend(BaseCommandBackend):
    def execute_train(self, request: ExecuteTrainRequest) -> None:
        external_ray = get_bool_env_var("MILES_SCRIPT_EXTERNAL_RAY")
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")

        self._clean_up_previous_run(external_ray=external_ray)

        if not external_ray:
            base_backend.exec_command_cpu(
                # will prevent ray from buffering stdout/stderr
                f"export PYTHONUNBUFFERED=1 && "
                f"ray start --head --node-ip-address {master_addr} --num-gpus {request.num_gpus_per_node} --disable-usage-stats"
            )

        if (f := request.before_ray_job_submit) is not None:
            f()

        runtime_env_vars = build_train_env_vars(request, self._ray_env_vars(master_addr=master_addr))
        runtime_env_vars["PYTHONPATH"] = _pythonpath_with_sources(
            request.megatron_path, runtime_env_vars.get("PYTHONPATH")
        )
        runtime_env_json = json.dumps({"env_vars": runtime_env_vars})

        if get_bool_env_var("MILES_SCRIPT_ENABLE_RAY_SUBMIT", "1"):
            model_args = shell_safe_model_args(request.megatron_model_type)
            base_backend.exec_command_cpu(
                f"export no_proxy=127.0.0.1 && export PYTHONUNBUFFERED=1 && "
                f"""ray job submit {'' if 'RAY_ADDRESS' in os.environ else '--address="http://127.0.0.1:8265" '}"""
                f"--runtime-env-json={shlex.quote(runtime_env_json)} "
                f"-- python3 {request.train_script} "
                f"{model_args} "
                f"{request.train_args}"
            )

    def exec_command_cpu(self, cmd: str, capture_output: bool = False) -> str | None:
        return run_shell_command(cmd, capture_output=capture_output)

    def exec_command_gpu(
        self, cmd: str, capture_output: bool = False, num_gpus_per_node: int | None = None
    ) -> str | None:
        return run_shell_command(cmd, capture_output=capture_output)

    def exec_command_multi_node(
        self,
        cmd: str,
        capture_output: bool = False,
        num_nodes: int | None = None,
        num_gpus_per_node: int | None = None,
    ) -> list[str | None]:
        return exec_command_all_ray_nodes(cmd, capture_output=capture_output, num_nodes=num_nodes)

    def _clean_up_previous_run(self, external_ray: bool) -> None:
        base_backend.exec_command_cpu(
            "pkill -9 sglang; "
            "sleep 3; "
            f"{'' if external_ray else 'ray stop --force; '}"
            f"{'' if external_ray else 'pkill -9 ray; '}"
            # cannot be run in CI, o/w kill the parent script
            # TODO: do we really need this kill? (or can we instead kill miles)
            # "pkill -9 python; "
            "pkill -9 miles; "
            "sleep 3; "
            f"{'' if external_ray else 'pkill -9 ray; '}"
            # "pkill -9 python; "
            "pkill -9 miles; "
            "pkill -9 redis; "
            "true; "
        )

    def _ray_env_vars(self, master_addr: str) -> dict[str, str]:
        return {
            "NCCL_NVLS_ENABLE": os.environ.get("NCCL_NVLS_ENABLE", str(int(check_has_nvlink()))),
            **{
                k: os.environ[k]
                for k in ("NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "NCCL_DEBUG", "NCCL_DEBUG_FILE")
                if k in os.environ
            },
            "no_proxy": f"127.0.0.1,{master_addr}",
            # This is needed by megatron / torch distributed in multi-node setup
            "MASTER_ADDR": master_addr,
        }


@ray.remote(num_cpus=0.001)
def _exec_command_on_node(cmd: str, capture_output: bool) -> str | None:
    return run_shell_command(f"unset CUDA_VISIBLE_DEVICES; {cmd}", capture_output=capture_output)


def exec_command_all_ray_nodes(
    cmd: str, capture_output: bool = False, num_nodes: int | None = None
) -> list[str | None]:
    ray.init(address="auto")
    try:
        current_ip = get_current_node_ip()
        nodes = sorted(
            [n for n in ray.nodes() if n.get("Alive")],
            key=lambda n: (n["NodeManagerAddress"] != current_ip, n["NodeManagerAddress"]),
        )
        assert len(nodes) > 0

        if num_nodes is not None:
            assert num_nodes <= len(nodes), f"Requested {num_nodes} nodes but only {len(nodes)} alive nodes available."
            nodes = nodes[:num_nodes]

        master_addr = nodes[0]["NodeManagerAddress"]
        nnodes = str(len(nodes))

        refs = []
        for rank, node in enumerate(nodes):
            node_cmd = substitute_placeholders(
                cmd,
                node_rank=str(rank),
                nnodes=nnodes,
                master_addr=master_addr,
                node_ip=node["NodeManagerAddress"],
            )
            refs.append(
                _exec_command_on_node.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=node["NodeID"],
                        soft=False,
                    ),
                ).remote(node_cmd, capture_output=capture_output)
            )
        return ray.get(refs)
    finally:
        ray.shutdown()
