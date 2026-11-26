import logging
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from miles.platforms import current_platform
from miles.utils.memory_utils import print_memory

if TYPE_CHECKING:
    from miles.utils.args.runtime import TrainerConfig

logger = logging.getLogger(__name__)


class TrainProfiler:
    def __init__(self, args: "TrainerConfig") -> None:
        self.args = args
        self._torch_profiler_overall = None
        self._memory_profiler_overall = None

        if _torch_profiler_enabled(args, name="train_overall"):
            self._torch_profiler_overall = _create_torch_profiler(args, name="train_overall")

        if args.backend.record_memory_history and ("train_overall" in args.profile_target):
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


def _torch_profiler_enabled(args: "TrainerConfig", name: str) -> bool:
    if not (args.backend.use_pytorch_profiler and name in args.profile_target):
        return False
    # Megatron's --profile-ranks: an empty list profiles every rank.
    return not args.backend.profile_ranks or torch.distributed.get_rank() in args.backend.profile_ranks


def _profile_simple_loop(iterator, args: "TrainerConfig", name: str):
    if not _torch_profiler_enabled(args, name=name):
        yield from iterator
        return

    torch_profiler = _create_torch_profiler(args, name=name)
    torch_profiler.start()
    for item in iterator:
        yield item
        torch_profiler.step()
    if isinstance(torch_profiler, _XprofProfiler):
        # a loop shorter than the window still gets its trace; libtpu's session must not outlive the loop either
        torch_profiler.stop()


def _create_torch_profiler(args: "TrainerConfig", name: str) -> "torch.profiler.profile | _XprofProfiler":
    if current_platform.trace_backend == "xprof":
        return _XprofProfiler(args, name=name)
    return torch.profiler.profile(
        schedule=torch.profiler.schedule(
            # TODO the train_actor and train_log_probs ones may need to have different args to control step
            wait=max(args.backend.profile_step_start - 1, 0),
            warmup=1 if args.backend.profile_step_start > 0 else 0,
            active=args.backend.profile_step_end - args.backend.profile_step_start,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            args.backend.tensorboard_dir,
            worker_name=f"{name}_rank_{torch.distributed.get_rank()}",
            use_gzip=True,
        ),
        # As in Megatron's own profiler: shapes and stacks only on request, memory and FLOPs never (no Megatron flag).
        # Per-op recording turns one large-model train step into a trace of tens of GB per rank.
        record_shapes=args.backend.pytorch_profiler_collect_shapes,
        with_stack=args.backend.pytorch_profiler_collect_callstack,
    )


class _XprofProfiler:
    """libtpu's tracer through torch_tpu, on torch.profiler's schedule: the trace covers steps
    --profile-step-start .. --profile-step-end - 1 (a step is the work between two step() calls, counted from
    start()) and is written under --tensorboard-dir/<name>_rank_<rank> as an XProf xplane, which TensorBoard's
    profile plugin opens. The device plane is always on; --pytorch-profiler-collect-callstack turns on the Python
    tracer (the expensive one); --pytorch-profiler-collect-shapes has no XProf equivalent.
    """

    def __init__(self, args: "TrainerConfig", name: str) -> None:
        self._first_step = args.backend.profile_step_start
        self._end_step = args.backend.profile_step_end
        self._directory = Path(args.backend.tensorboard_dir) / f"{name}_rank_{torch.distributed.get_rank()}"
        self._python_tracer = args.backend.pytorch_profiler_collect_callstack
        self._steps = 0
        self._active = False

    def start(self) -> None:
        self._steps = 0
        self._on_step_boundary()

    def step(self) -> None:
        self._steps += 1
        self._on_step_boundary()

    def stop(self) -> None:
        if not self._active:
            return
        from torch_tpu._internal.profiler import _impl as tpu_profiler

        # the trace ends with the device's work of the last step, not with the host's dispatch of it
        torch.accelerator.synchronize()
        tpu_profiler.stop_trace()
        self._active = False
        logger.info(f"XProf trace of steps {self._first_step}..{self._steps - 1} written to {self._directory}")

    def _on_step_boundary(self) -> None:
        if self._steps == self._first_step and self._first_step < self._end_step:
            self._start_trace()
        elif self._steps == self._end_step:
            self.stop()

    def _start_trace(self) -> None:
        # start_trace/stop_trace live in _impl; torch_tpu's own profile() context manager takes no options
        from torch_tpu._internal.profiler import _impl as tpu_profiler

        self._directory.mkdir(parents=True, exist_ok=True)
        options = tpu_profiler.ProfileOptions(
            device_tracer_level=1, host_tracer_level=2, python_tracer_level=1 if self._python_tracer else 0
        )
        tpu_profiler.start_trace(str(self._directory), options)
        self._active = True


class _BaseMemoryProfiler:
    @staticmethod
    def create(args):
        c = {
            "torch": _TorchMemoryProfiler,
            "memray": _MemrayMemoryProfiler,
        }[args.memory_recorder]
        return c(args)

    def __init__(self, args: "TrainerConfig") -> None:
        self._path_dump = (
            Path(args.memory_snapshot_dir)
            / f"memory_snapshot_time{time.time()}_rank{torch.distributed.get_rank()}_{args.backend.memory_snapshot_path}"
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
