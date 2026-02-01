import gzip
import json
import logging
import tempfile
import time
import traceback
from pathlib import Path

import torch
from torch.profiler import record_function

from miles.utils.memory_utils import print_memory

logger = logging.getLogger(__name__)


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
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            args.tensorboard_dir,
            worker_name=f"{name}_rank_{torch.distributed.get_rank()}",
            use_gzip=True,
        ),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
        with_flops=True,
    )


class FunctionStepProfiler:
    """
    Wraps a function to profile each invocation.

    Uses torch.profiler.profile with CUDA activities to capture kernel-level
    details and Python-to-CUDA correlation.
    """

    def __init__(self, args, name: str, label: str = "target_fn", start: int = 0, end: int = 1):
        self.args = args
        self.name = name
        self.label = label
        self.call_count = 0
        self.enabled = True
        self.output_dir = Path(args.tensorboard_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start = start
        self.end = end
        logger.info(f"Profiler initialization: { self.output_dir }")

    def wrap(self, fn):
        def _wrapped(*args, **kwargs):
            if not self.enabled:
                return fn(*args, **kwargs)
            self.call_count += 1
            if not (self.start <= self.call_count < self.end):
                return fn(*args, **kwargs)
            logger.info(f"FunctionStepProfiler: Profiling call {self.call_count} for '{self.label}'")

            try:
                # Determine activities based on CUDA availability
                assert torch.cuda.is_available(), "CUDA must be available for FunctionStepProfiler"
                activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]

                # Use torch.profiler.profile for proper CUDA kernel profiling
                with torch.profiler.profile(
                    activities=activities,
                    record_shapes=True,
                    with_stack=True,
                    profile_memory=True,
                    with_flops=True,
                ) as prof:
                    with record_function(self.label):
                        result = fn(*args, **kwargs)
                        torch.cuda.synchronize()

                # Export the trace to a gzipped file
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                trace_file = self.output_dir / f"{self.name}_call{self.call_count}_rank_{rank}.pt.trace.json.gz"
                with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as tmp:
                    prof.export_chrome_trace(tmp.name)
                    with open(tmp.name, "rb") as f_in, gzip.open(trace_file, "wb") as f_out:
                        f_out.write(f_in.read())
                logger.info(f"FunctionStepProfiler: Call {self.call_count} profiled, trace saved to {trace_file}")
                return result
            except Exception as e:
                raise ValueError(f"FunctionStepProfiler: Profiler error for '{self.label}', details: {e}") from e

        return _wrapped


def merge_traces(name="update_weights", call_end=5, rank=0, output_dir="/root/profiler_logs/"):
    merged = {"traceEvents": []}
    output_file = Path(output_dir) / f"merged_{name}_rank_{rank}_merged.pt.trace.json.gz"
    for call_iter in range(1, call_end):
        f = Path(output_dir) / f"{name}_call{call_iter}_rank_{rank}.pt.trace.json.gz"
        with gzip.open(f, "rt") as fp:
            data = json.load(fp)
            merged["traceEvents"].extend(data.get("traceEvents", []))
    with gzip.open(output_file, "wt") as fp:
        json.dump(merged, fp)


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
