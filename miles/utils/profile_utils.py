import gzip
import hashlib
import json
import logging
import os
import socket
import threading
import time
import traceback
from contextlib import ExitStack, contextmanager, redirect_stdout
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import torch
from torch.utils._pytree import tree_leaves

from miles.utils.memory_utils import print_memory

if TYPE_CHECKING:
    from miles.utils.args.runtime import TrainerConfig

logger = logging.getLogger(__name__)


_DYNAMIC_CAPTURE_LOCK = threading.Lock()
_ACTIVITIES = {"CPU": torch.profiler.ProfilerActivity.CPU, "CUDA": torch.profiler.ProfilerActivity.CUDA}


class DynamicProfiler:
    """Rank-aware local capture; call from the worker executing the operation."""

    def __init__(self, external=()):
        self.external = external
        self.config = None
        self.last_capture = None
        self.last_error = None
        self._step = 0
        self._captures = 0
        self._state = "stopped"
        self._capturing = False
        self._deadline = None

    @property
    def rank(self):
        return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    def start(self, config):
        if self.config is not None or self._capturing:
            raise ValueError("Profiling is already armed or recording")
        config = dict(config)
        if "num_steps" in config:
            raise ValueError("Use active/repeat for scheduled profiling and call profile_step at update boundaries")
        schedule = {
            key: config.get(key, default)
            for key, default in (("skip_first", 0), ("wait", 0), ("warmup", 0), ("active", 1), ("repeat", 1))
        }
        if any(type(value) is not int or value < 0 for value in schedule.values()) or schedule["active"] < 1:
            raise ValueError("Schedule fields must be nonnegative integers; active must be positive")
        self._schedule = torch.profiler.schedule(**schedule)
        cycle = schedule["wait"] + schedule["warmup"] + schedule["active"]
        self._schedule_end = schedule["skip_first"] + schedule["repeat"] * cycle if schedule["repeat"] else None
        if config.get("max_duration_sec", 7200) <= 0:
            raise ValueError("max_duration_sec must be positive")
        activities = config.get("activities", ["CPU", "CUDA"])
        if not activities or set(activities) - _ACTIVITIES.keys():
            raise ValueError("Activities must contain CPU and/or CUDA")
        if "CUDA" in activities and not torch.cuda.is_available():
            raise ValueError("CUDA profiling requires an available CUDA device")
        if not 0 < config.get("max_bytes", 950_000_000) < 1_000_000_000:
            raise ValueError("max_bytes must be positive and below 1 GB")
        if "://" in config["output_dir"]:
            raise ValueError("Capture requires a local output directory")
        if config.get("profile_modules") and (config.get("with_stack") or not self.external):
            raise ValueError("Module tracking requires registered model objects and with_stack=false")
        if config.get("profile_modules") and not config.get("profile_memory"):
            raise ValueError("Module tracking requires profile_memory=true")
        self._activities = [_ACTIVITIES[name] for name in activities]
        self.last_capture, self.last_error = None, None
        ranks = config.get("ranks", [])
        self.config = config if not ranks or self.rank in ranks else None
        self._step, self._captures = 0, 0
        self._state = "armed" if self.config else "skipped"
        self._deadline = time.monotonic() + config.get("max_duration_sec", 7200)
        return self.status()

    def status(self):
        if self.config is not None and time.monotonic() >= self._deadline:
            self.config = None
            self._state = "expired"
        return {
            "active": self.config is not None,
            "status": "recording" if self._capturing else self._state,
            "step": self._step,
            "action": self._schedule(self._step).name if self.config else None,
            "captures": self._captures,
            "capture": self.last_capture,
            "error": self.last_error,
            "rank": self.rank,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
        }

    def profile_step(self):
        """Advance once at an optimizer/update boundary, independently of run-step labels."""
        if self._capturing:
            raise ValueError("Call profile_step after leaving the capture scope")
        if self.status()["active"]:
            self._step += 1
            if self._schedule_end is not None and self._step >= self._schedule_end:
                self.config = None
                self._state = "completed"
        return self.status()

    def stop(self):
        self.config = None
        self._state = "stopped"
        return self.status()

    @contextmanager
    def capture(self, step=None, operation="train"):
        if not self.status()["active"] or self._schedule(self._step) == torch.profiler.ProfilerAction.NONE:
            yield
            return
        config = self.config
        if not _DYNAMIC_CAPTURE_LOCK.acquire(blocking=False):
            self.last_error = "Another operation owns the process-wide profiler"
            yield
            return
        self._capturing = True
        try:
            with self._capture_operation(config, step, operation):
                yield
        finally:
            self._capturing = False
            _DYNAMIC_CAPTURE_LOCK.release()

    @contextmanager
    def _capture_operation(self, config, step, operation):
        self.last_error = None
        warmup = self._schedule(self._step) == torch.profiler.ProfilerAction.WARMUP
        directory = Path(config["output_dir"]) / f"rank-{self.rank:05d}-{uuid4().hex}"
        path = directory / "trace.json"
        options = {key: config.get(key, False) for key in ("record_shapes", "with_stack", "profile_memory")}
        activities = config.get("activities", ["CPU", "CUDA"])
        stack, memory, tracker = ExitStack(), None, None
        try:
            if not warmup:
                directory.mkdir(parents=True)
            profiler = stack.enter_context(
                torch.profiler.profile(
                    activities=self._activities,
                    schedule=(lambda _: torch.profiler.ProfilerAction.WARMUP) if warmup else None,
                    **options,
                )
            )
            if not warmup and config.get("profile_modules"):
                from torch.distributed._tools.mem_tracker import MemTracker

                tracker = MemTracker()
                tracker.track_external(*tree_leaves(self.external))
                stack.enter_context(tracker)
            stack.enter_context(torch.profiler.record_function(f"miles/{operation}/step={step}"))
            if not warmup and options["profile_memory"] and "CUDA" in activities:
                memory = _TorchMemoryProfiler(None, directory / "memory.pickle")
                memory.start(record_oom=False)
                torch.cuda.reset_peak_memory_stats()
        except Exception as error:
            self.last_error = str(error)
            logger.exception("Profiling setup failed; operation will continue without capture")
            try:
                stack.close()
            except Exception:
                logger.exception("Profiling cleanup failed")
            if memory is not None:
                _disable_dynamic_memory_history()
            yield
            return
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            try:
                stack.close()
                if not warmup:
                    if memory is not None:
                        memory.stop()
                        stats = {
                            "allocated_bytes": torch.cuda.memory_allocated(),
                            "reserved_bytes": torch.cuda.memory_reserved(),
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                            "device_used_bytes": torch.cuda.device_memory_used(),
                        }
                        (directory / "memory.json").write_text(json.dumps(stats))
                    profiler.export_chrome_trace(str(path))
                    if tracker is not None:
                        try:
                            _export_module_memory(tracker, path)
                        except Exception as error:
                            self.last_error = f"Module memory export failed: {error}"
                            logger.exception("Module report failed; execution trace retained")
                            path.with_suffix(".modules.json").unlink(missing_ok=True)
                            path.with_suffix(".modules.txt").unlink(missing_ok=True)
                    self.last_capture = finalize_capture(
                        directory,
                        step,
                        operation,
                        self.rank,
                        config.get("max_bytes", 950_000_000),
                        succeeded=succeeded,
                    )
                    self._captures += 1
            except Exception as error:
                self.last_error = str(error)
                logger.exception("Profiling export failed; partial artifacts retained at %s", directory)
            finally:
                if memory is not None:
                    _disable_dynamic_memory_history()


def _disable_dynamic_memory_history():
    try:
        torch.cuda.memory._record_memory_history(enabled=None)
    except Exception:
        logger.exception("Failed to disable CUDA memory history")


def finalize_capture(
    directory,
    step,
    operation,
    rank,
    max_bytes=950_000_000,
    trace_name="trace.json",
    succeeded=True,
    trace_format="chrome_trace",
):
    """Publish a local completion manifest only after every artifact passes admission."""
    files = sorted(path for path in Path(directory).iterdir() if path.is_file())
    total = 0
    for path in files:
        if path.name.endswith(".json.gz"):
            with gzip.open(path, "rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    total += len(chunk)
                    if total >= max_bytes:
                        raise ValueError("Profile exceeds its uncompressed byte budget")
        else:
            total += path.stat().st_size
        if total >= max_bytes:
            raise ValueError("Profile exceeds its byte budget; reduce capture detail")
    if not (Path(directory) / trace_name).is_file():
        raise ValueError("No execution trace was exported")
    entries = []
    for path in files:
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256")
        entries.append({"path": path.name, "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    result = {
        "step_start": step,
        "step_end": step,
        "operation": operation,
        "rank": rank,
        "hostname": socket.gethostname(),
        "trace": trace_name,
        "format": trace_format,
        "operation_succeeded": succeeded,
        "files": entries,
    }
    manifest = Path(directory) / "manifest.json"
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(manifest)
    return str(manifest)


class TrainProfiler:
    def __init__(self, args: "TrainerConfig") -> None:
        self.args = args
        self.dynamic = DynamicProfiler()
        self._torch_profiler_overall = None
        self._memory_profiler_overall = None
        if _torch_profiler_enabled(args, name="train_overall"):
            self._torch_profiler_overall = _create_torch_profiler(args, name="train_overall")
        if args.backend.record_memory_history and ("train_overall" in args.profile_target):
            self._memory_profiler_overall = _BaseMemoryProfiler.create(args)
            self._memory_profiler_overall.start()

    def start_dynamic(self, config):
        if self._torch_profiler_overall is not None or self._memory_profiler_overall is not None:
            raise ValueError("Scheduled profiling is already configured")
        return self.dynamic.start(config)

    def on_init_end(self, *external):
        self.dynamic.external = external
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


def _export_module_memory(tracker, path):
    def snapshot(value):
        if isinstance(value, dict):
            return {str(key.value if isinstance(key, Enum) else key): snapshot(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [snapshot(item) for item in value]
        return value

    result = {
        "scope": "Process-wide live tensor snapshots at module boundaries; module rows overlap",
        "peak": tracker.get_tracker_snapshot("peak"),
        "modules": {
            stat.mod_fqn: {
                "parameter_bytes": stat.parameter_mem,
                "buffer_bytes": stat.buffer_mem,
                "states": stat.snapshots,
            }
            for stat in tracker.memory_tracking.values()
        },
    }
    path.with_suffix(".modules.json").write_text(json.dumps(snapshot(result), indent=2))
    with path.with_suffix(".modules.txt").open("w") as output, redirect_stdout(output):
        tracker.display_modulewise_snapshots(depth=8, units="MiB")


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


def _create_torch_profiler(args: "TrainerConfig", name: str) -> torch.profiler.profile:
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


class _BaseMemoryProfiler:
    @staticmethod
    def create(args):
        c = {
            "torch": _TorchMemoryProfiler,
            "memray": _MemrayMemoryProfiler,
        }[args.memory_recorder]
        return c(args)

    def __init__(self, args: "TrainerConfig", path=None) -> None:
        self._path_dump = path or (
            Path(args.memory_snapshot_dir)
            / f"memory_snapshot_time{time.time()}_rank{torch.distributed.get_rank()}_{args.backend.memory_snapshot_path}"
        )

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError


class _TorchMemoryProfiler(_BaseMemoryProfiler):
    def start(self, record_oom=True):
        logger.info("Start CUDA memory history.")

        torch.cuda.memory._record_memory_history(
            max_entries=1000000,
            clear_history=True,
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

        if record_oom:
            torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    def stop(self):
        logger.info(f"Dump memory snapshot to: {self._path_dump}")
        try:
            torch.cuda.memory._dump_snapshot(self._path_dump)
        finally:
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


def profile_training_operation(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        step = kwargs.get("rollout_id", kwargs.get("batch_id"))
        if step is None and args and function.__name__ in {"train", "forward_backward", "forward_only"}:
            step = args[0]
        with self.prof.dynamic.capture(step=step, operation=function.__name__):
            result = function(self, *args, **kwargs)
        if function.__name__ in {"train", "optim_step"}:
            self.prof.dynamic.profile_step()
        return result

    return wrapped
