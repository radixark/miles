#!/usr/bin/env python3
# doc-dev: docs/ci/02-docker-build.md
"""GPU smoke probe for a freshly built Miles image.

Runs inside the image (docker run --gpus) on a build node, before anything is
pushed. Seconds-scale sanity only — the deep verification is the CI suite; this
gate exists for the failure class where a green build ships a broken runtime
environment (e.g. TE once imported fine at build time but died at first use in
every GPU job because a --no-deps install dropped onnxscript).
"""

import os

import torch

assert torch.cuda.is_available(), "no CUDA device visible in the image"
x = torch.ones(8, device="cuda")
assert float(x.sum()) == 8.0, "CUDA tensor math broken"

import sglang  # noqa: E402,F401

# The real TE import (not just metadata): pulls in onnxscript and friends.
import transformer_engine.pytorch  # noqa: E402,F401

import miles  # noqa: E402,F401

assert os.path.exists("/usr/local/bin/all_reduce_perf"), "nccl-tests binaries missing"

print(f"smoke OK: torch {torch.__version__}, cuda {torch.version.cuda}, " f"device {torch.cuda.get_device_name(0)}")
