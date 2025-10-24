from pickle import dump

import torch


def attach_oom_dump_memory_history():
    torch.cuda.memory._record_memory_history(
        max_entries=100000,
        # record stack information for the trace events
        # trace_alloc_record_context=True,
        stacks="all",
    )

    def oom_observer(device, alloc, device_alloc, device_free):
        path_dump = f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}"
        print(f"Observe OOM, will dump snapshot to {path_dump}. ({device=} {alloc=} {device_alloc=} {device_free=})")

        # TODO use `_dump_snapshot` instead?
        snapshot = torch.cuda.memory._snapshot()
        dump(snapshot, open(path_dump, "wb"))

    torch._C._cuda_attach_out_of_memory_observer(oom_observer)
