import logging
from argparse import Namespace
from collections.abc import Callable
from copy import deepcopy

from miles.utils import tracking_utils
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.timer import Timer

logger = logging.getLogger(__name__)

_PEAK_TFLOPS_BY_GPU = {
    "H100": 990.0,
    "H200": 990.0,
    "B200": 2250.0,
    "A100": 312.0,
    "A800": 312.0,
    "H800": 990.0,
}


def _get_peak_gpu_tflops(args: Namespace) -> float | None:
    """Return peak BF16 TFLOPS per GPU, from args or auto-detected."""
    if getattr(args, "peak_gpu_tflops", None) is not None:
        return args.peak_gpu_tflops
    try:
        import torch

        device_name = torch.cuda.get_device_name(0).upper()
        for key, val in _PEAK_TFLOPS_BY_GPU.items():
            if key in device_name:
                return val
    except Exception:
        pass
    return None


def log_perf_data_raw(
    rollout_id: int, args: Namespace, is_primary_rank: bool, compute_total_fwd_flops: Callable
) -> None:
    timer_instance = Timer()
    log_dict_raw = deepcopy(timer_instance.log_dict())
    timer_instance.reset()

    if not is_primary_rank:
        return

    log_dict = {f"perf/{key}_time": val for key, val in log_dict_raw.items()}

    if ("perf/actor_train_time" in log_dict) and (compute_total_fwd_flops is not None):
        total_fwd_flops = compute_total_fwd_flops(seq_lens=timer_instance.seq_lens)

        if "perf/log_probs_time" in log_dict:
            log_dict["perf/log_probs_tflops"] = total_fwd_flops / log_dict["perf/log_probs_time"]

        if "perf/ref_log_probs_time" in log_dict:
            log_dict["perf/ref_log_probs_tflops"] = total_fwd_flops / log_dict["perf/ref_log_probs_time"]

        if log_dict["perf/actor_train_time"] > 0:
            log_dict["perf/actor_train_tflops"] = 3 * total_fwd_flops / log_dict["perf/actor_train_time"]
            log_dict["perf/actor_train_tok_per_s"] = sum(timer_instance.seq_lens) / log_dict["perf/actor_train_time"]

        # MFU: Model FLOPs Utilization (fraction of peak hardware throughput)
        peak_tflops = _get_peak_gpu_tflops(args)
        if peak_tflops is not None:
            if "perf/actor_train_tflops" in log_dict:
                log_dict["perf/train_mfu"] = log_dict["perf/actor_train_tflops"] / peak_tflops
            if "perf/log_probs_tflops" in log_dict:
                log_dict["perf/log_probs_mfu"] = log_dict["perf/log_probs_tflops"] / peak_tflops

    if "perf/train_wait_time" in log_dict and "perf/train_time" in log_dict:
        total_time = log_dict["perf/train_wait_time"] + log_dict["perf/train_time"]
        if total_time > 0:
            log_dict["perf/step_time"] = total_time
            log_dict["perf/wait_time_ratio"] = log_dict["perf/train_wait_time"] / total_time

    # GPU idle fraction: fraction of step_time not spent on compute
    step_time = log_dict.get("perf/step_time", 0)
    if step_time > 0:
        compute_time = sum(
            log_dict.get(k, 0) for k in ("perf/actor_train_time", "perf/log_probs_time", "perf/ref_log_probs_time")
        )
        log_dict["perf/train_gpu_idle_fraction"] = 1.0 - min(compute_time / step_time, 1.0)

        pause_time = log_dict.get("perf/pause_generation_time", 0)
        if pause_time > 0:
            log_dict["perf/rollout_gpu_idle_fraction"] = pause_time / step_time

    logger.info(f"perf {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    tracking_utils.log(args, log_dict, step_key="rollout/step")
