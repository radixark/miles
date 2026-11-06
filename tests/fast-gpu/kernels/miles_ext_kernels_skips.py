"""The skips of the miles-ext-kernels submodule's tests that the kernel harness accepts: hardware limits, per op.

``test_miles_ext_kernels.py`` (one GPU per op) and ``test_miles_ext_kernels_multi_gpu.py`` (four GPUs) fail on every
skip whose reason this file does not list for the op of the skipped test: the ``<op>`` of ``tests/cake/<op>/``,
``_runtime`` for the shared runtime. A listed reason is a limit of the runner, documented by the test that skips: fewer
GPUs or less free memory than the test needs, two GPUs of different parts, a GPU whose kernels or tools do not serve one
case. A missing dependency, a missing CUDA device and an unsupported GPU are never listed: on the runners the harness is
registered for, such a skip would hide the kernel coverage it exists for.

An entry is the reason the test passes to ``pytest.skip`` or to a ``skipif`` mark, verbatim; a ``{name}`` in it stands
for a value the test formats in at run time.
"""

import re

SKIP_ALLOWLIST: dict[str, tuple[str, ...]] = {
    "_runtime": (),
    "gdn": (
        # multi_gpu: a context-parallel case needs a GPU per rank
        "context parallelism {world_size} needs {world_size} GPUs; {visible} visible",
        # torch.profiler records no CUDA API call on some parts (R200 under CUDA 13), so launches cannot be counted
        "torch.profiler (torch {torch}, CUDA {cuda}) records no CUDA API call on this device ({name}, compute "
        "capability {major}.{minor}, {sm_count} SMs): the launch of one known PyTorch kernel left no LaunchKernel "
        "event either, so launches and synchronisations cannot be counted here",
    ),
    "kda": (
        # multi_gpu: a context-parallel group needs a GPU per rank, on one node
        "the {world}-rank groups need {world} GPUs of one node, {visible} visible",
    ),
    "conv1d": (
        # the largest case needs 40 GiB of free GPU memory
        "needs 40 GiB free, {free} GiB available",
        # multi_gpu: a context-parallel case needs a GPU per rank
        "needs {world_size} GPUs, {visible} visible",
    ),
    "gated_rmsnorm": (),
    "rmsnorm": (
        # multi_gpu: the sequence-parallel case runs two ranks
        "needs 2 GPUs",
    ),
    "dsa": (
        # multi_gpu: the second-device cases need two GPUs of one compute capability
        "needs two GPUs",
        "the two GPUs differ in compute capability",
    ),
    "mhc": (
        # SM107a serves hidden size 4096 only
        "hidden 5120 is not served on this device",
        "hidden 7168 is not served on this device",
        # multi_gpu: the second-device case needs two GPUs of one model
        "needs two GPUs",
        "the second GPU is another model",
    ),
    "gqa": (
        # multi_gpu: two GPUs of one model and SM count
        "needs two GPUs",
        "the two GPUs differ in model or SM count: {models}",
    ),
    "dense_gemm": (
        # fast's known accuracy misses come from the fast tables of 148-SM SM103a parts; other parts run other programs
        "the known accuracy misses of fast are on 148-SM sm_103a parts",
        # a part whose fast tables pick no weight-gradient program of their own for the tested shapes
        "fast runs no program of its own for these shapes on {arch} with {sm_count} SMs",
        # multi_gpu: two devices of one part
        "two CUDA devices required",
        "two devices of one SM100a / SM103a / SM107a part required",
    ),
}

# One line of pytest's -rs summary: "SKIPPED [2] tests/cake/gdn/test_gdn_kcp_gpu.py:543: <reason>"
_SKIPPED = re.compile(r"SKIPPED \[(\d+)\] ([^\s:]+)(?::\d+)?: (.*)")


def skips(stdout: str) -> list[tuple[int, str, str]]:
    """``(count, test file, reason)`` of every line of pytest's ``-rs`` summary in ``stdout``."""
    lines = (_SKIPPED.fullmatch(line) for line in stdout.splitlines())
    return [(int(count), path, reason) for count, path, reason in (m.groups() for m in lines if m)]


def _pattern(entry: str) -> re.Pattern[str]:
    """``entry`` as a pattern: its text verbatim, each ``{name}`` any non-empty text."""
    return re.compile(".+".join(map(re.escape, re.split(r"\{\w+\}", entry))))


def is_allowed(path: str, reason: str) -> bool:
    """Whether ``SKIP_ALLOWLIST`` lists ``reason`` for the op of the test file ``path`` (``tests/cake/<op>/...``)."""
    parts = path.split("/")
    op = parts[2] if len(parts) > 3 and parts[:2] == ["tests", "cake"] else None
    return any(_pattern(entry).fullmatch(reason) for entry in SKIP_ALLOWLIST.get(op, ()))
