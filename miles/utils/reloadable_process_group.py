import logging
import os
from contextlib import contextmanager

import torch
import torch.distributed as dist

from miles.utils.memory_utils import print_memory

logger = logging.getLogger(__name__)

old_new_group_dict = {}


def monkey_patch_torch_dist():
    pid = os.getpid()
    if pid in old_new_group_dict:
        assert dist.old_new_group == old_new_group_dict[pid]
        return

    logger.info("Applying monkey patch to torch.distributed")

    old_new_group = dist.new_group
    old_new_group_dict[pid] = old_new_group
    dist.old_new_group = old_new_group

    def new_group(*args, **kwargs):
        group = old_new_group(*args, **kwargs)
        # skip none nccl group.
        if len(args) >= 3 and args[2] == "gloo" or "backend" in kwargs and kwargs["backend"] == "gloo":
            return group

        # Get ranks from arguments
        if len(args) >= 1 and args[0] is not None:
            ranks = args[0]
        elif "ranks" in kwargs and kwargs["ranks"] is not None:
            ranks = kwargs["ranks"]
        else:
            # If no ranks specified, use all ranks in world
            ranks = list(range(dist.get_world_size()))
        if len(ranks) == 1:
            return group

        group = ReloadableProcessGroup(group, inner_args=args, inner_kwargs=kwargs)
        return group

    dist.new_group = new_group

    def get_new_function(func):
        def new_function(*args, **kwargs):
            args = tuple([arg.group if isinstance(arg, ReloadableProcessGroup) else arg for arg in args])
            kwargs = {k: (v.group if isinstance(v, ReloadableProcessGroup) else v) for k, v in kwargs.items()}
            with _wrap_low_level_call():
                return func(*args, **kwargs)

        return new_function

    dist.get_rank = get_new_function(dist.get_rank)
    dist.get_world_size = get_new_function(dist.get_world_size)
    dist.get_backend = get_new_function(dist.get_backend)
    dist.get_global_rank = get_new_function(dist.get_global_rank)
    dist.get_group_rank = get_new_function(dist.get_group_rank)
    dist.get_process_group_ranks = get_new_function(dist.get_process_group_ranks)

    dist.all_reduce = get_new_function(dist.all_reduce)
    dist.all_gather = get_new_function(dist.all_gather)
    dist.all_gather_into_tensor = get_new_function(dist.all_gather_into_tensor)
    dist.all_gather_object = get_new_function(dist.all_gather_object)
    dist.all_to_all = get_new_function(dist.all_to_all)
    dist.all_to_all_single = get_new_function(dist.all_to_all_single)
    dist.broadcast = get_new_function(dist.broadcast)
    dist.broadcast_object_list = get_new_function(dist.broadcast_object_list)
    dist.reduce = get_new_function(dist.reduce)
    dist.reduce_scatter = get_new_function(dist.reduce_scatter)
    dist.reduce_scatter_tensor = get_new_function(dist.reduce_scatter_tensor)
    dist.scatter = get_new_function(dist.scatter)
    dist.gather = get_new_function(dist.gather)
    dist.barrier = get_new_function(dist.barrier)
    dist.send = get_new_function(dist.send)
    dist.recv = get_new_function(dist.recv)
    dist._coalescing_manager = get_new_function(dist._coalescing_manager)

    # p2p
    old_isend = dist.isend
    old_irecv = dist.irecv

    dist.isend = get_new_function(dist.isend)
    dist.irecv = get_new_function(dist.irecv)

    def get_new_p2pop_function(func):
        def new_function(*args, **kwargs):
            def convert(arg):
                if isinstance(arg, ReloadableProcessGroup):
                    return arg.group
                elif arg == dist.isend:
                    arg = old_isend
                elif arg == dist.irecv:
                    arg = old_irecv
                return arg

            args = (convert(arg) for arg in args)
            kwargs = {k: convert(v) for k, v in kwargs.items()}
            return func(*args, **kwargs)

        return new_function

    dist.P2POp.__new__ = get_new_p2pop_function(dist.P2POp.__new__)
    dist.P2POp.__init__ = get_new_p2pop_function(dist.P2POp.__init__)


class ReloadableProcessGroup(torch.distributed.ProcessGroup):
    """Wrapper PG whose inner group can be destroyed and rebuilt.

    Do not define python overrides for collectives here: pybind installs them
    as trampoline overrides that wrap every returned Work in
    c10d::PyProcessGroup::PyWorkHolder, which segfaults on torch 2.13.
    Collectives dispatch through the C++ backend map that
    ``_adopt_inner_backends`` registers; ``_fwd`` is for plumbing methods only.
    """

    GROUPS = {}

    def __init__(self, group, inner_args, inner_kwargs):
        super().__init__(
            rank=dist.get_rank(group),
            size=dist.get_world_size(group),
        )
        self.group = group
        self.inner_args = inner_args
        self.inner_kwargs = inner_kwargs
        self._adopt_inner_backends()
        pid = os.getpid()
        if pid not in ReloadableProcessGroup.GROUPS:
            ReloadableProcessGroup.GROUPS[pid] = []
        ReloadableProcessGroup.GROUPS[pid].append(self)

    def _adopt_inner_backends(self) -> None:
        """Register the inner group's backends on this wrapper.

        Newer torch (observed on 2.13) dispatches collectives through the C++
        backend map of the group object itself, which __getattr__ forwarding
        cannot cover. Must re-run after every reload: the fresh inner group
        carries fresh backends.
        """
        if self.group is None:
            return
        for device_type in ("cpu", "cuda"):
            device = torch.device(device_type)
            try:
                backend = self.group._get_backend(device)
            except Exception:
                continue
            backend_cls = type(backend).__name__
            if "NCCL" in backend_cls:
                backend_type = torch.distributed.ProcessGroup.BackendType.NCCL
            elif "Gloo" in backend_cls.title() or "GLOO" in backend_cls.upper():
                backend_type = torch.distributed.ProcessGroup.BackendType.GLOO
            else:
                backend_type = torch.distributed.ProcessGroup.BackendType.CUSTOM
            try:
                self._register_backend(device, backend_type, backend)
            except Exception as exc:
                logger.warning(f"Could not register {device_type} backend on reloadable group: {exc}")

    def __getattr__(self, name):
        return getattr(self.group, name)

    @staticmethod
    def destroy_process_groups():
        pid = os.getpid()
        for reloadable_group in ReloadableProcessGroup.GROUPS.get(pid, []):
            if reloadable_group.group is None:
                continue
            try:
                dist.destroy_process_group(reloadable_group.group)
            except ValueError as e:
                logger.warning(
                    f"Process group already invalid/destroyed; skipping cleanup. Exception: {e}",
                    exc_info=True,
                )

            del reloadable_group.group
            reloadable_group.group = None

    @staticmethod
    def reload_process_groups():
        pid = os.getpid()
        reloadable_groups = ReloadableProcessGroup.GROUPS.get(pid, [])
        logger.info(f"Reloading {len(reloadable_groups)} process groups in pid {pid}")
        old_new_group = old_new_group_dict.get(pid)
        for reloadable_group in reloadable_groups:
            if reloadable_group.group is not None:
                continue
            group = old_new_group(*reloadable_group.inner_args, **reloadable_group.inner_kwargs)
            reloadable_group.group = group
            reloadable_group._adopt_inner_backends()

    def rank(self) -> int:
        return self.group.rank()

    def size(self) -> int:
        return self.group.size()

    def name(self) -> str:
        return self.group.name()

    def shutdown(self) -> None:
        if self.group is not None:
            self.group.shutdown()

    def abort(self) -> None:
        if self.group is not None:
            self.group.abort()

    def _fwd(self, method, *args, **kwargs):
        inner = self.group
        if inner is None:
            raise RuntimeError("ReloadableProcessGroup: inner PG is None, call reload() first.")
        with _wrap_low_level_call():
            return getattr(inner, method)(*args, **kwargs)

    def _start_coalescing(self, *a, **kw):
        return self._fwd("_start_coalescing", *a, **kw)

    def _end_coalescing(self, *a, **kw):
        return self._fwd("_end_coalescing", *a, **kw)

    def _get_backend_name(self):
        return self._fwd("_get_backend_name")

    def _get_backend(self, *a, **kw):
        return self._fwd("_get_backend", *a, **kw)

    def _set_default_backend(self, *a, **kw):
        return self._fwd("_set_default_backend", *a, **kw)

    @property
    def bound_device_id(self):
        return self.group.bound_device_id

    @bound_device_id.setter
    def bound_device_id(self, dev):
        self.group.bound_device_id = dev


def _forward_remaining_collectives():
    """Forward every ProcessGroup collective this class does not define itself.

    Callers that resolved a collective before the monkey patch went on --
    Megatron binds `dist_reduce_scatter_func` at import time -- hand the wrapper
    straight to torch, which invokes the method on the group object. Anything
    not overridden here reaches the C++ base, whose own backend map is empty,
    and dies as "No backend type associated with device type cuda". A
    hand-written forward list silently regrows that hole whenever torch renames
    or adds a collective, which is how torch 2.13's *_single family got through.
    """
    skip = {"rank", "size", "name", "abort", "shutdown", "bound_device_id"}
    for name in dir(dist.ProcessGroup):
        if name.startswith("__") or name in skip:
            continue
        if name in vars(ReloadableProcessGroup):
            continue
        if not callable(getattr(dist.ProcessGroup, name, None)):
            continue

        def make(method):
            def forward(self, *args, **kwargs):
                return self._fwd(method, *args, **kwargs)

            forward.__name__ = method
            return forward

        setattr(ReloadableProcessGroup, name, make(name))


_forward_remaining_collectives()


def destroy_process_groups():
    """Destroy all reloadable process groups."""
    ReloadableProcessGroup.destroy_process_groups()


def reload_process_groups():
    """Reload all reloadable process groups."""
    ReloadableProcessGroup.reload_process_groups()


@contextmanager
def _wrap_low_level_call():
    try:
        yield
    except Exception as e:
        mem_info = print_memory("after torch distributed error")
        if hasattr(e, "add_note"):
            e.add_note(f"{mem_info=}")
        raise
