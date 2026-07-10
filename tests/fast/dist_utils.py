import os
import socket
from typing import Any

import torch.distributed as dist
import torch.multiprocessing as mp


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def init_gloo(rank: int, world_size: int, *, port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def run_multiprocess(fn: Any, world_size: int = 2) -> None:
    port = find_free_port()
    mp.spawn(fn, args=(world_size, port), nprocs=world_size, join=True)
