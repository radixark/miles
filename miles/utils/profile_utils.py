import torch

def attach_oom_dump_memory_history():
    torch.cuda.memory._record_memory_history(
        # keep 100,000 alloc/free events from before the snapshot
        max_entries=100000,
        # record stack information for the trace events
        # trace_alloc_record_context=True,
        stacks="all",
    )

    def oom_observer(device, alloc, device_alloc, device_free):
        # snapshot right after an OOM happened
        print("saving allocated state during OOM")
        snapshot = torch.cuda.memory._snapshot()
        from pickle import dump

        dump(
            snapshot,
            open(f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}", "wb"),
        )

    torch._C._cuda_attach_out_of_memory_observer(oom_observer)
