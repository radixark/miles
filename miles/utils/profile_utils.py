import logging
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import torch

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

    @contextmanager
    def profile_phase(self, name: str, rollout_id: int | None):
        if rollout_id is None or not should_profile(self.args, name, rollout_id):
            yield
            return
        with _create_torch_profiler(
            self.args, name, output_dir=profile_output_dir(self.args, name, rollout_id)
        ):
            yield

    def iterate_train_actor(self, iterator):
        return _profile_simple_loop(iterator, self.args, name="train_actor")

    def iterate_train_log_probs(self, iterator):
        return _profile_simple_loop(iterator, self.args, name="train_log_probs")


def _profile_simple_loop(iterator, args, name):
    if not (args.use_pytorch_profiler and (name in args.profile_target)):
        yield from iterator
        return

    schedule = torch.profiler.schedule(
        # TODO the train_actor and train_log_probs ones may need to have different args to control step
        wait=max(args.profile_step_start - 1, 0),
        warmup=1 if args.profile_step_start > 0 else 0,
        active=args.profile_step_end - args.profile_step_start,
        repeat=1,
    )
    torch_profiler = _create_torch_profiler(args, name, schedule=schedule)
    torch_profiler.start()
    for item in iterator:
        yield item
        torch_profiler.step()


def should_profile(args, stage: str, rollout_id: int) -> bool:
    step = rollout_id - (getattr(args, "start_rollout_id", None) or 0)
    return (
        getattr(args, "use_pytorch_profiler", False)
        and stage in getattr(args, "profile_target", ())
        and getattr(args, "profile_step_start", 0) <= step < getattr(args, "profile_step_end", 0)
    )


def profile_output_dir(args, stage: str, rollout_id: int | None = None, worker: str | None = None) -> str:
    path = Path(args.profile_output_dir) / stage
    if rollout_id is not None:
        path = path / f"rollout_{rollout_id}"
    if worker is not None:
        path = path / worker
    return str(path)


def profile_activities_for_torch(args):
    activities = []
    for activity in getattr(args, "profile_activities", ["cpu", "gpu"]):
        if activity == "cpu":
            activities.append(torch.profiler.ProfilerActivity.CPU)
        elif activity == "gpu":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities


def profile_activities_for_sglang(args) -> list[str]:
    return [
        activity.upper() if activity == "cpu" else "GPU"
        for activity in getattr(args, "profile_activities", ["cpu", "gpu"])
    ]


def _create_torch_profiler(args, name, *, schedule=None, output_dir: str | None = None):
    return torch.profiler.profile(
        activities=profile_activities_for_torch(args),
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            output_dir or profile_output_dir(args, name),
            worker_name=f"{name}_rank_{torch.distributed.get_rank()}",
            use_gzip=True,
        ),
        record_shapes=args.profile_record_shapes,
        with_stack=args.profile_with_stack,
        profile_memory=args.profile_memory,
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
