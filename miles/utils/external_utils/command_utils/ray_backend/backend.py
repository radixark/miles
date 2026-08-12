import json
import os
import shlex

from miles.utils.external_utils.command_utils.base_backend import (
    BaseCommandBackend,
    ExecuteTrainRequest,
    resolve_extra_env_vars,
)
from miles.utils.external_utils.command_utils.common import (
    _pythonpath_with_sources,
    get_bool_env_var,
)
from miles.utils.external_utils.exec_command import exec_command_gpu, exec_command_multi_node
from miles.utils.external_utils.model_args_utils import shell_safe_model_args


class RayCommandBackend(BaseCommandBackend):
    def _execute_train_inner(self, request: ExecuteTrainRequest) -> None:
        external_ray = get_bool_env_var("MILES_SCRIPT_EXTERNAL_RAY")
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")

        self._clean_up_previous_run(external_ray=external_ray)

        if not external_ray:
            self.exec_command_cpu(
                # will prevent ray from buffering stdout/stderr
                f"export PYTHONUNBUFFERED=1 && "
                f"ray start --head --node-ip-address {master_addr} --num-gpus {request.num_gpus_per_node} --disable-usage-stats"
            )

        for cmd in request.prepare_cmd.values():
            self.exec_command_multi_node(cmd)

        if (f := request.before_ray_job_submit) is not None:
            f()

        runtime_env_vars = {
            # exported for the submitting client too, but only the runtime env reaches the ray workers
            "PYTHONUNBUFFERED": "1",
            # If setting this in FSDP, the computation communication overlapping may have issues
            **(
                {}
                if request.train_backend_fsdp
                else {
                    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                }
            ),
            # a get() default is evaluated eagerly, which would probe even when already decided
            "NCCL_NVLS_ENABLE": os.environ.get("NCCL_NVLS_ENABLE") or str(int(self._check_has_nvlink())),
            **{
                k: os.environ[k]
                for k in ("NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "NCCL_DEBUG", "NCCL_DEBUG_FILE")
                if k in os.environ
            },
            "no_proxy": f"127.0.0.1,{master_addr}",
            # This is needed by megatron / torch distributed in multi-node setup
            "MASTER_ADDR": master_addr,
            **(
                {
                    "CUDA_ENABLE_COREDUMP_ON_EXCEPTION": "1",
                    "CUDA_COREDUMP_SHOW_PROGRESS": "1",
                    "CUDA_COREDUMP_GENERATION_FLAGS": "skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory",
                    "CUDA_COREDUMP_FILE": f"{self.config.output_dir}/cuda_coredump_%h.%p.%t",
                }
                if self.config.cuda_core_dump
                else {}
            ),
            **resolve_extra_env_vars(request.extra_env_vars, self.config),
        }
        runtime_env_vars["PYTHONPATH"] = _pythonpath_with_sources(
            request.megatron_path, runtime_env_vars.get("PYTHONPATH")
        )
        runtime_env_json = json.dumps({"env_vars": runtime_env_vars})

        if get_bool_env_var("MILES_SCRIPT_ENABLE_RAY_SUBMIT", "1"):
            model_args = shell_safe_model_args(request.megatron_model_type)
            self.exec_command_cpu(
                f"export no_proxy=127.0.0.1 && export PYTHONUNBUFFERED=1 && "
                f"""ray job submit {'' if 'RAY_ADDRESS' in os.environ else '--address="http://127.0.0.1:8265" '}"""
                f"--runtime-env-json={shlex.quote(runtime_env_json)} "
                f"-- python3 {request.train_script} "
                f"{model_args} "
                f"{request.train_args}"
            )

    def exec_command_gpu(
        self, cmd: str, capture_output: bool = False, num_gpus_per_node: int | None = None
    ) -> str | None:
        return exec_command_gpu(cmd, capture_output=capture_output)

    def exec_command_multi_node(
        self,
        cmd: str,
        capture_output: bool = False,
        num_nodes: int | None = None,
        num_gpus_per_node: int | None = None,
    ) -> list[str | None]:
        return exec_command_multi_node(cmd, capture_output=capture_output, num_nodes=num_nodes)

    def _check_has_nvlink(self) -> bool:
        output = self.exec_command_gpu(
            "nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l", capture_output=True
        )
        return int(output) > 0

    def _clean_up_previous_run(self, external_ray: bool) -> None:
        self.exec_command_cpu(
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
