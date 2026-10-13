import json
import logging
import time
import traceback
from collections import defaultdict
from pathlib import Path

import torch

from miles.utils.memory_utils import print_memory

logger = logging.getLogger(__name__)

_LOW_PRECISION_CATEGORIES = (
    "scale_amax",
    "quantization",
    "gemm",
    "dequantization",
    "layout_memory",
)


def categorize_low_precision_kernel(name: str) -> str | None:
    """Map a CUDA kernel name to a stable, user-facing profiling category."""
    normalized = name.casefold()
    patterns = (
        ("dequantization", ("dequant", "cast_from_fp8", "cast_from_fp4")),
        ("quantization", ("quantize", "cast_to_fp8", "cast_to_fp4", "fp8_cast", "fp4_cast")),
        ("gemm", ("gemm", "nvjet_", "cublas", "cutlass", "matmul")),
        ("scale_amax", ("amax", "compute_scale", "scale_inv", "scale_kernel")),
        ("layout_memory", ("swizzle", "transpose", "permute", "memcpy", "memset", "copy_kernel")),
    )
    for category, markers in patterns:
        if any(marker in normalized for marker in markers):
            return category
    return None


def _summarize_low_precision_events(events, *, name: str, rank: int, step: int) -> dict:
    events = list(events)
    category_totals = {category: {"duration_us": 0.0, "kernel_calls": 0} for category in _LOW_PRECISION_CATEGORIES}
    kernels = defaultdict(lambda: {"duration_us": 0.0, "calls": 0, "parents": set(), "input_shapes": set()})
    kernel_metadata = defaultdict(lambda: {"parents": set(), "input_shapes": set()})
    uncategorized_duration = 0.0
    uncategorized_calls = 0

    for event in events:
        for kernel in getattr(event, "kernels", ()):
            metadata = kernel_metadata[kernel.name]
            metadata["parents"].add(event.name)
            if event.input_shapes:
                metadata["input_shapes"].add(json.dumps(event.input_shapes))

    for event in events:
        if getattr(getattr(event, "device_type", None), "name", None) != "CUDA":
            continue
        duration = float(event.device_time)
        category = categorize_low_precision_kernel(event.name)
        if category is None:
            uncategorized_duration += duration
            uncategorized_calls += 1
            continue

        category_totals[category]["duration_us"] += duration
        category_totals[category]["kernel_calls"] += 1
        kernel_summary = kernels[(category, event.name)]
        kernel_summary["duration_us"] += duration
        kernel_summary["calls"] += 1
        metadata = kernel_metadata[event.name]
        kernel_summary["parents"].update(metadata["parents"])
        kernel_summary["input_shapes"].update(metadata["input_shapes"])

    kernel_details = []
    for (category, kernel_name), summary in sorted(kernels.items()):
        kernel_details.append(
            {
                "category": category,
                "name": kernel_name,
                "duration_us": summary["duration_us"],
                "calls": summary["calls"],
                "parents": sorted(summary["parents"]),
                "input_shapes": [json.loads(shapes) for shapes in sorted(summary["input_shapes"])],
            }
        )

    return {
        "schema_version": 1,
        "profile_name": name,
        "rank": rank,
        "step": step,
        "duration_model": "sum_of_unique_cuda_activity_durations; overlapping kernels are not wall time",
        "categories": category_totals,
        "uncategorized": {
            "duration_us": uncategorized_duration,
            "kernel_calls": uncategorized_calls,
        },
        "kernels": kernel_details,
    }


def _create_trace_handler(args, *, name: str):
    rank = torch.distributed.get_rank()
    profile_low_precision = getattr(args, "profile_low_precision", False)
    if profile_low_precision and not args.tensorboard_dir:
        raise ValueError("--profile-low-precision requires --tensorboard-dir")
    tensorboard_handler = torch.profiler.tensorboard_trace_handler(
        args.tensorboard_dir,
        worker_name=f"{name}_rank_{rank}",
        use_gzip=True,
    )
    if not profile_low_precision:
        return tensorboard_handler

    output_dir = Path(args.tensorboard_dir)

    def handle_trace(profiler):
        tensorboard_handler(profiler)
        summary = _summarize_low_precision_events(
            profiler.events(),
            name=name,
            rank=rank,
            step=profiler.step_num,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"low_precision_{name}_rank_{rank}_step_{profiler.step_num}.json"
        output_path.write_text(json.dumps(summary, indent=2) + "\n")
        logger.info(f"Low-precision profile summary written to {output_path}")

    return handle_trace


class TrainProfiler:
    def __init__(self, args):
        self.args = args
        self._torch_profiler_overall = None
        self._memory_profiler_overall = None

        if args.use_pytorch_profiler and ("train_overall" in args.profile_target):
            self._torch_profiler_overall = _create_torch_profiler(args, name="train_overall")

        if args.record_memory_history and ("train_overall" in args.profile_target):
            self._memory_profiler_overall = _BaseMemoryProfiler.create(args)
            self._memory_profiler_overall.start()

    def on_init_end(self):
        if self._torch_profiler_overall is not None:
            self._torch_profiler_overall.start()

    def step(self, rollout_id: int):
        if self._torch_profiler_overall is not None:
            self._torch_profiler_overall.step()

        if (
            self._memory_profiler_overall is not None
            and ((s := self.args.memory_snapshot_num_steps) is not None)
            and (rollout_id == s - 1)
        ):
            self._memory_profiler_overall.stop()

    def iterate_train_actor(self, iterator):
        return _profile_simple_loop(iterator, self.args, name="train_actor")

    def iterate_train_log_probs(self, iterator):
        return _profile_simple_loop(iterator, self.args, name="train_log_probs")


def _profile_simple_loop(iterator, args, name):
    if not (args.use_pytorch_profiler and (name in args.profile_target)):
        yield from iterator
        return

    torch_profiler = _create_torch_profiler(args, name=name)
    torch_profiler.start()
    for item in iterator:
        yield item
        torch_profiler.step()


def _create_torch_profiler(args, name):
    return torch.profiler.profile(
        schedule=torch.profiler.schedule(
            # TODO the train_actor and train_log_probs ones may need to have different args to control step
            wait=max(args.profile_step_start - 1, 0),
            warmup=1 if args.profile_step_start > 0 else 0,
            active=args.profile_step_end - args.profile_step_start,
            repeat=1,
        ),
        on_trace_ready=_create_trace_handler(args, name=name),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
        with_flops=True,
    )


class _BaseMemoryProfiler:
    @staticmethod
    def create(args):
        c = {
            "torch": _TorchMemoryProfiler,
            "memray": _MemrayMemoryProfiler,
        }[args.memory_recorder]
        return c(args)

    def __init__(self, args):
        self._path_dump = (
            Path(args.memory_snapshot_dir)
            / f"memory_snapshot_time{time.time()}_rank{torch.distributed.get_rank()}_{args.memory_snapshot_path}"
        )

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError


class _TorchMemoryProfiler(_BaseMemoryProfiler):
    def start(self):
        logger.info("Attach OOM dump memory history.")

        torch.cuda.memory._record_memory_history(
            max_entries=1000000,
            # record stack information for the trace events
            # trace_alloc_record_context=True,
            stacks="all",
        )

        def oom_observer(device, alloc, device_alloc, device_free):
            logger.info(
                f"Observe OOM, will dump snapshot to {self._path_dump}. ({device=} {alloc=} {device_alloc=} {device_free=}; stacktrace is as follows)"
            )
            traceback.print_stack()
            torch.cuda.memory._dump_snapshot(self._path_dump)
            print_memory("when oom")

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    def stop(self):
        logger.info(f"Dump memory snapshot to: {self._path_dump}")
        torch.cuda.memory._dump_snapshot(self._path_dump)
        torch.cuda.memory._record_memory_history(enabled=None)


class _MemrayMemoryProfiler(_BaseMemoryProfiler):
    def __init__(self, args):
        super().__init__(args)
        assert args.memory_snapshot_num_steps is not None, "In memray, must provide --memory-snapshot-num-steps"

    def start(self):
        logger.info("Memray tracker started.")
        import memray

        self._tracker = memray.Tracker(
            file_name=self._path_dump,
            native_traces=True,
        )
        self._tracker.__enter__()

    def stop(self):
        logger.info(f"Memray tracker stopped and dump snapshot to: {self._path_dump}")
        self._tracker.__exit__(None, None, None)
