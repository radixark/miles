import dataclasses
import ipaddress
import logging
import multiprocessing
import os
import time

import requests
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.launch_server import kill_process_tree

from miles.ray.ray_actor import RayActor
from miles.utils.http_utils import get_host_info

logger = logging.getLogger(__name__)


def _to_local_gpu_id(physical_gpu_id: int) -> int:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        return physical_gpu_id  # no remapping
    # CUDA_VISIBLE_DEVICES can be like "4,5,6,7"
    visible = [int(x) for x in cvd.split(",") if x.strip() != ""]
    # In a remapped process, valid torch device indices are 0..len(visible)-1
    if physical_gpu_id in visible:
        return visible.index(physical_gpu_id)
    # If we're already getting local IDs, allow them
    if 0 <= physical_gpu_id < len(visible):
        return physical_gpu_id
    raise RuntimeError(
        f"GPU id {physical_gpu_id} is not valid under CUDA_VISIBLE_DEVICES={cvd}. "
        f"Expected one of {visible} (physical) or 0..{len(visible)-1} (local)."
    )


def _launch_server_target(server_args):
    # addict.Dict used by SGL-D loses its `__frozen` instance attribute across spawn pickle.
    # Reconstruct a fresh one from the unpickled (broken) instance
    import addict

    if server_args.attention_backend_config is not None:
        server_args.attention_backend_config = addict.Dict(server_args.attention_backend_config)

    from sglang.multimodal_gen.runtime.launch_server import launch_server
    launch_server(server_args)


def launch_server_process(server_args: ServerArgs) -> multiprocessing.Process:
    # use spawn to avoid potential risks of fork in terms of subthreads or CUDA.
    multiprocessing.set_start_method("spawn", force=True)
    server_args.host = server_args.host.strip("[]")
    p = multiprocessing.Process(target=_launch_server_target, args=(server_args,))
    p.start()

    _wait_server_healthy(
        base_url=server_args.url(),
        is_process_alive=lambda: p.is_alive(),
    )

    return p


def _wait_server_healthy(base_url, is_process_alive):
    headers = {
        "Content-Type": "application/json; charset=utf-8",
    }

    with requests.Session() as session:
        while True:
            try:
                # SGL-D health_generate
                response = session.get(f"{base_url}/health_generate", headers=headers)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass

            if not is_process_alive():
                raise Exception("Server process terminated unexpectedly.")

            time.sleep(2)

class SGLangDiffusionEngine(RayActor):
    def __init__(self, args, rank: int, base_gpu_id: int | None = None):
        self.args = args
        # rank: the global rank of this engine among all rollout engines
        self.rank = rank
        self.base_gpu_id = base_gpu_id

    def init(self, dist_init_addr, port, nccl_port, host=None):
        # `dist_init_addr` is a multi-node concept from LLM sglang; SGL-D runs
        # single-node per engine. Accept it for caller compat, then drop.
        del dist_init_addr
        self.router_ip = self.args.sglang_router_ip
        self.router_port = self.args.sglang_router_port

        host = host or get_host_info()[1]

        def _format_v6_uri(addr):
            if not addr or addr.startswith("["):
                return addr
            try:
                if ipaddress.ip_address(addr).version == 6:
                    return f"[{addr}]"
            except ValueError:
                pass
            return addr

        host = _format_v6_uri(host)

        server_args_dict, external_engine_need_check_fields = _compute_server_args(
            self.args,
            host=host,
            port=port,
            nccl_port=nccl_port,
        )

        self.node_rank = server_args_dict.get("node_rank", 0)
        self.server_host = server_args_dict["host"]  # with [] if ipv6
        self.server_port = server_args_dict["port"]

        # keep external rollout engine for debug
        if self.args.rollout_external:
            self._init_external(server_args_dict, external_engine_need_check_fields=external_engine_need_check_fields)
        else:
            self._init_normal(server_args_dict)

    def _init_external(self, expect_server_args):
        logger.info(f"Use external SGLang-Diffusion engine (rank={self.rank}, expect_server_args={expect_server_args})")

        # TODO: miles diffusion support server args sanity check
        # Now only do healthy check for generate
        # SGL-D TODO: SGLang-D support get actual server args
        # def _get_actual_server_args():
        #     response = requests.get(f"http://{self.server_host}:{self.server_port}/get_server_info")
        #     response.raise_for_status()
        #     return response.json()

        _wait_server_healthy(
            base_url=f"http://{self.server_host}:{self.server_port}",
            is_process_alive=lambda: True,
        )

    def _init_normal(self, server_args_dict):
        logger.info(f"Launch HttpServerEngineAdapter at: {self.server_host}:{self.server_port}")
        self._pin_to_assigned_gpu()
        self.process = launch_server_process(ServerArgs.from_kwargs(**server_args_dict))

        if self.node_rank == 0 and self.router_ip and self.router_port:
            if self.args.use_miles_router:
                response = requests.post(
                    f"http://{self.router_ip}:{self.router_port}/add_worker?url=http://{self.server_host}:{self.server_port}"
                )
                response.raise_for_status()
            else:
                # SGL-D router TODO: add_worker path for the non-miles router
                logger.warning("Skipping router add_worker: only miles_router is supported for now")

    def _pin_to_assigned_gpu(self):
        if self.base_gpu_id is None:
            return
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not cvd:
            return
        visible = [x.strip() for x in cvd.split(",") if x.strip()]
        local_idx = _to_local_gpu_id(self.base_gpu_id)
        pinned = visible[local_idx]
        os.environ["CUDA_VISIBLE_DEVICES"] = pinned
        logger.info(
            f"Engine rank={self.rank}: pinned CUDA_VISIBLE_DEVICES={pinned} "
            f"(base_gpu_id={self.base_gpu_id}, local_idx={local_idx})"
        )

    def _make_request(self, endpoint: str, payload: dict | None = None):
        """Make a POST request to the specified endpoint with the given payload.

        Args:
            endpoint: The API endpoint to call
            payload: The JSON payload to send (default: empty dict)

        Returns:
            The JSON response from the server
        """
        if self.node_rank != 0:
            return

        url = f"http://{self.server_host}:{self.server_port}/{endpoint}"
        response = requests.post(url, json=payload or {})
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            e.add_note(f"{response.text=}")
            raise
        return response.json()

    def health_generate(self, timeout: float = 5.0) -> bool:
        """Run /health_generate on the underlying SGLang-Diffusion HTTP server.

        Args:
            timeout: Timeout for the health request in seconds.

        Returns:
            True if the server responds with HTTP 200.

        Raises:
            requests.RequestException: If the request fails for any reason, including timeout.
        """
        if self.node_rank != 0:
            return True

        response = requests.get(
            f"http://{self.server_host}:{self.server_port}/health_generate",
            timeout=timeout,
        )
        response.raise_for_status()
        return True

    def update_weights_from_tensor(
        self,
        serialized_named_tensors: list[str],
        load_format: str | None = None,
        weight_version: str | None = None,
    ):
        """
        Update model weights from tensor data. The HTTP server will only post meta data, and the real weights will be copied directly from GPUs.

        Note: The model should be on GPUs rather than CPU for this functionality to work properly.
        If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        """
        # SGL-D TODO: SGLang-Diffusion support update weights from tensor
        payload = {
            "serialized_named_tensors": serialized_named_tensors,
            "load_format": load_format
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return self._make_request(
            "update_weights_from_tensor",
            payload,
        )

    def shutdown(self):
        if self.args.rollout_external:
            return

        logger.info(f"Shutdown engine {self.server_host}:{self.server_port}...")
        if self.node_rank == 0:
            worker_url = f"http://{self.server_host}:{self.server_port}"
            response = None
            if self.args.use_miles_router:
                response = requests.post(
                    f"http://{self.router_ip}:{self.router_port}/remove_worker?url={worker_url}"
                )
            else:
                # SGL-D router TODO: shutdown for sglang-diffusion router
                logger.warning(f"Failed to fetch workers list or remove worker: now only support miles_router")

            if response is not None:
                response.raise_for_status()
        kill_process_tree(self.process.pid)

    def get_weight_version(self):
        if self.node_rank != 0:
            return
        # SGL-D TODO: SGLang-Diffusion support get weight version
        url = f"http://{self.server_host}:{self.server_port}/get_weight_version"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()["weight_version"]

    def release_memory_occupation(self):
        # SGL-D TODO: SGLang-Diffusion support release/resume memory occupation
        # This function is used by offload/recover
        return self._make_request("release_memory_occupation")

    def resume_memory_occupation(self, tags: list[str] | None = None):
        # SGL-D TODO: SGLang-Diffusion support resume memory occupation
        # This function is used by offload/recover
        return self._make_request(
            "resume_memory_occupation",
            {"tags": tags},
        )

    def check_weights(self, action: str):
        # SGL-D TODO: SGLang-Diffusion support weights checker
        # Weight checker supports a series of actions through which user can check weights
        # e.g. snapshot, compare, reset_tensors, etc.
        return self._make_request("weights_checker", {"action": action})

    def init_weights_update_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        # SGL-D TODO: Support weights update group for in-memory weight update
        del master_address, master_port, rank_offset, world_size, group_name, backend
        raise NotImplementedError("init_weights_update_group is not implemented in SGL-D yet")

    def simulate_crash(self):
        if self.args.rollout_external or not getattr(self, "process", None):
            logger.info(
                "simulate_crash called but no local engine process exists (rollout_external=%s); skip kill",
                self.args.rollout_external,
            )
            return

        logger.info(f"Simulating crash on engine {self.server_host}:{self.server_port}...")
        self.shutdown()


def _compute_server_args(args, host, port, nccl_port):
    # Only set fields SGL-D's ServerArgs actually accepts. GPU pinning is done
    # in `_init_normal` via CUDA_VISIBLE_DEVICES — SGL-D has no base_gpu_id arg.
    kwargs = {
        "model_path": args.diffusion_model,
        "trust_remote_code": True,
        "host": host,
        "port": port,
        "nccl_port": nccl_port,
        # Each engine needs a distinct master_port starting hint so that
        # concurrent settle_port() probes don't race on the same default (30005).
        "master_port": nccl_port + 10000 if nccl_port is not None else None,
        # parallel — tp_size must match rollout allocation, not user CLI.
        "tp_size": args.rollout_num_gpus_per_engine,
        # Sequence-parallel degree (None = disabled, SGL-D decides internally).
        "sp_degree": args.sglang_sp_degree,
        # Classifier-free-guidance parallel (splits cond/uncond across GPUs).
        "enable_cfg_parallel": args.sglang_enable_cfg_parallel,
        # Force-skip warmup to prevent warmup timeout during RL rollouts.
        "warmup": False,
    }
    # TODO: map args.fp16 to SGL-D precision control (ServerArgs has no `dtype`).

    # Forward every `args.sglang_<field>` the user set via --sglang-* CLI for
    # ServerArgs fields not already hardcoded above. Picks up ulysses_degree /
    # ring_degree / dp_size / etc. without listing each one.
    for attr in dataclasses.fields(ServerArgs):
        if hasattr(args, f"sglang_{attr.name}") and attr.name not in kwargs:
            kwargs[attr.name] = getattr(args, f"sglang_{attr.name}")

    external_engine_need_check_fields = [
        k for k in kwargs.keys() if k not in _EXTERNAL_ENGINE_SKIP_CHECK_FIELDS
    ]
    return kwargs, external_engine_need_check_fields


# Fields to skip when verifying an external SGLang-Diffusion engine's server args
# against what miles would have computed. Empty for now; add field names here as
# the external-engine sanity check grows (e.g. ports that legitimately differ).
_EXTERNAL_ENGINE_SKIP_CHECK_FIELDS: list[str] = []
