import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

SCRIPT_DIR = Path(__file__).resolve().parent
FULLY_ASYNC_DIR = (Path(__file__).resolve().parent.parent / "examples" / "fully_async").resolve()

# Cluster-wide GPU-node ceiling for the ckpt-conversion job. Kept below the
# raw node count so ckpt conversion doesn't starve the rest of the cluster.
MAX_CONVERT_GPUS = 92
print(f'{SCRIPT_DIR=}')
print(f'{FULLY_ASYNC_DIR=}')