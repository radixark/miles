"""Opt-in, instance-local HDO adapter: keep at most one CPU parameter's gradient.

Only the gradient staging changes. HDO's parameter-group/state synchronization,
AdamW implementations, FP32 masters/moments, and registered copy-back hooks stay
in charge. No MCore class or global torch method is patched.
"""

import logging
from types import MethodType

import torch

logger = logging.getLogger(__name__)
_FLAG = "--optimizer-cpu-streaming-gradients"


def add_optimizer_cpu_streaming_gradients_argument(parser):
    parser.add_argument(
        _FLAG,
        action="store_true",
        help=(
            "Stage one HDO CPU AdamW gradient at a time using blocking, pageable D2H copies. "
            "Requires --optimizer-cpu-offload and --overlap-cpu-optimizer-d2h-h2d; "
            "preserves FP32 master weights and optimizer-state dtypes."
        ),
    )
    return parser


def _params(optimizer):
    for group in optimizer.param_groups:
        yield from group["params"]


def _require(condition, message):
    if not condition:
        raise RuntimeError(f"{_FLAG}: {message}")


def _validate_hdo(hdo):
    """Check live objects, also after HDO load_state_dict rebuilds its children."""
    for name in (
        "cpu_optimizers",
        "gpu_optimizer",
        "cpu_copys_map_gpu_param",
        "gpu_params_map_cpu_copy",
        "param_to_fp32_param",
        "cpu_copy_map_grad",
        "_cpu_optimizer_map_data_event",
        "param_update_in_fp32",
    ):
        _require(hasattr(hdo, name), f"unsupported HDO: missing {name}")
    for name in ("_sync_hdo_param_groups_to_sub_optimizers", "_sync_sub_optimizers_state_to_hdo"):
        _require(callable(getattr(hdo, name, None)), f"unsupported HDO: missing {name}")
    for name in ("_d2h_stream", "_h2d_stream"):
        stream = getattr(hdo, name, None)
        _require(
            callable(getattr(stream, "synchronize", None)) and callable(getattr(stream, "wait_stream", None)),
            f"unsupported HDO stream {name}",
        )
    _require(getattr(hdo, "offload_fraction", 0) > 0, "requires HDO CPU offload")
    _require(getattr(hdo, "overlap_cpu_optimizer_d2h_h2d", False), "requires per-parameter overlap mode")
    _require(not hdo.cpu_copy_map_grad, "cannot enable/re-enter with cached CPU gradients")
    _require(not hdo._cpu_optimizer_map_data_event, "cannot enable/re-enter with pending D2H events")
    seen = set()
    for optimizer in hdo.cpu_optimizers:
        params = list(_params(optimizer))
        _require(len(params) == 1, "each CPU sub-optimizer must own exactly one parameter")
        param = params[0]
        _require(param not in seen, "CPU sub-optimizers must not share parameters")
        seen.add(param)
        _require(param.device.type == "cpu" and param.dtype == torch.float32, "requires FP32 CPU masters")
        _require(param.grad is None, "CPU masters must not retain gradients between streaming steps")
        _require(param in hdo.cpu_copys_map_gpu_param, "missing CPU-to-GPU parameter mapping")
        source = hdo.cpu_copys_map_gpu_param[param]
        _require(hdo.gpu_params_map_cpu_copy.get(source) is param, "inconsistent CPU/GPU parameter mappings")
        _require(isinstance(optimizer, torch.optim.AdamW), "only CPU torch.optim.AdamW is supported")
        _require(
            any(
                getattr(hook, "__qualname__", "").endswith(
                    "HybridDeviceOptimizer._register_param_copy_back_gpu_hook."
                    "<locals>.param_copy_back_gpu_hook_closure.<locals>.param_copy_back_gpu_hook"
                )
                for hook in optimizer._optimizer_step_post_hooks.values()
            ),
            "missing supported HDO CPU parameter copy-back hook",
        )
        for group in optimizer.param_groups:
            _require(not group.get("differentiable", False), "differentiable AdamW is unsupported")
    _require(seen == set(hdo.cpu_copys_map_gpu_param), "CPU sub-optimizer parameter coverage mismatch")


def _sync_gpu_grads(hdo):
    """Retain HDO's non-offloaded FP32 gradient path (including decoupled_grad)."""
    if not hdo.param_update_in_fp32:
        return
    for param, master in hdo.param_to_fp32_param.items():
        if param in hdo.gpu_params_map_cpu_copy:
            continue
        grad = getattr(param, "decoupled_grad", param.grad)
        # AdamW tests grad, not requires_grad: explicitly clear stale None grads.
        master.grad = None if grad is None else grad.to(master.dtype)
        master.requires_grad = grad is not None


@torch.no_grad()
def _streaming_step(self, closure=None):
    _require(closure is None, "closures are unsupported because they can mutate unstaged gradients")
    _require(not self._miles_streaming_gradients_failed, "a previous step failed; restart/load in a fresh optimizer")
    _validate_hdo(self)
    try:
        self._sync_hdo_param_groups_to_sub_optimizers()
        # A CUDA stream wait is NOT a host-side barrier. A prior copy-back must
        # finish before AdamW may overwrite the CPU master that is its source.
        self._h2d_stream.synchronize()
        self._d2h_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._d2h_stream):
            _sync_gpu_grads(self)
        self._d2h_stream.synchronize()

        if self.gpu_optimizer is not None:
            self.gpu_optimizer.step(closure)

        for optimizer in self.cpu_optimizers:
            param = next(_params(optimizer))
            source = self.cpu_copys_map_gpu_param[param]
            grad = getattr(source, "decoupled_grad", source.grad)
            param.requires_grad = False
            try:
                if grad is not None:
                    self._d2h_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(self._d2h_stream):
                        # No pinned allocation and no asynchronous host read.
                        # copy=True also avoids aliasing a CPU input in CPU tests.
                        param.grad = grad.to("cpu", dtype=param.dtype, non_blocking=False, copy=True)
                    self.cpu_copy_map_grad[param] = param.grad
                else:
                    param.grad = None
                # Keep the original optimizer entrypoint: its copy-back and any
                # other step hooks still run in their original registration order.
                optimizer.step(closure)
            finally:
                try:
                    # Also drain work queued by a hook which subsequently raises.
                    self._h2d_stream.synchronize()
                finally:
                    param.grad = None
                    self.cpu_copy_map_grad.pop(param, None)
        self._sync_sub_optimizers_state_to_hdo()
    except BaseException:
        # Never fall back to the eager implementation after a partial Adam step,
        # nor allow a caller that catches the error to silently step it again.
        self._miles_streaming_gradients_failed = True
        raise


def install_optimizer_cpu_streaming_gradients(args, optimizer):
    """Install on all HDO instances under Megatron wrappers, or fail before any install."""
    if not getattr(args, "optimizer_cpu_streaming_gradients", False):
        return
    _require(getattr(args, "train_backend", "megatron") == "megatron", "requires the Megatron backend")
    _require(getattr(args, "optimizer_cpu_offload", False), "requires --optimizer-cpu-offload")
    _require(
        getattr(args, "overlap_cpu_optimizer_d2h_h2d", False),
        "requires --overlap-cpu-optimizer-d2h-h2d",
    )
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer

    hdos = []
    seen = set()

    def visit(node):
        if node is None or id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, HybridDeviceOptimizer):
            hdos.append(node)
        elif hasattr(node, "chained_optimizers"):
            for child in node.chained_optimizers:
                visit(child)
        elif hasattr(node, "optimizer"):
            visit(node.optimizer)
        else:
            _require(False, f"unsupported optimizer leaf {type(node).__name__}; expected HDO")

    visit(optimizer)
    _require(bool(hdos), "no HybridDeviceOptimizer instance found")
    for hdo in hdos:
        _validate_hdo(hdo)
    # Preserve top-level torch optimizer pre/post hooks as well as child hooks.
    step = torch.optim.Optimizer.profile_hook_step(_streaming_step)
    for hdo in hdos:
        if getattr(hdo, "_miles_streaming_gradients_installed", False):
            continue
        hdo.step = MethodType(step, hdo)
        hdo._miles_streaming_gradients_failed = False
        hdo._miles_streaming_gradients_installed = True
    logger.info(
        "Enabled HDO streaming CPU gradients: %d HDO instances, %d CPU sub-optimizers; "
        "blocking pageable copies, unchanged FP32 masters/moments",
        len(hdos),
        sum(len(hdo.cpu_optimizers) for hdo in hdos),
    )
