"""GPU-cardinality lint for the 4-GPU H200 suite.

Every test file registered to the `stage-c-4-gpu-h200` suite must have its
GPU-cardinality anchors (NUM_GPUS, --actor-num-gpus-per-node,
--rollout-num-gpus-per-engine) and parallelism flags
(--tensor-model-parallel-size, --context-parallel-size,
--expert-model-parallel-size, --expert-tensor-parallel-size,
--pipeline-model-parallel-size) all consistent with `world_size <= 4` per
the formula `world_size = max(TP * CP, EP * ETP) * PP`. Single-axis bounds
checked here are necessary (not sufficient) — a file passes if every individual
parallelism flag is within the per-axis ceiling that 4 GPUs can support.

Specificity is enforced by a negative sanity check: h100-exclusion files MUST
still trigger at least one match, otherwise the regex set is vacuous.
"""

import glob
import re

from tests.ci.ci_register import collect_tests, register_cpu_ci

# This lint itself runs on stage-a-cpu (it only reads files; no GPU needed).
register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])


_H200_SUITE = "stage-c-4-gpu-h200"
_H100_EXCLUSIONS = (
    "tests/e2e/ckpt/test_glm47_flash_ckpt.py",
    "tests/e2e/megatron/test_glm47_flash_r3_mtp.py",
    "tests/e2e/megatron/test_qwen3_30B_A3B_r3.py",
    "tests/e2e/megatron/test_qwen3_4B_ppo.py",
    "tests/e2e/megatron/test_qwen3_5_35B_A3B_cp.py",
)


# Each pattern matches a token that would push world_size above 4 on a
# 4-GPU host. Per-axis ceilings:
#   NUM_GPUS              <= 4  -> reject 8 (most common stale anchor)
#   --actor-num-gpus-per-node <= 4 -> reject 8
#   --rollout-num-gpus-per-engine <= 4 -> reject 5..N
#   --tensor-model-parallel-size <= 4 -> reject 5..N
#   --context-parallel-size <= 2 -> reject 3..N (CP=2 already saturates with TP=2)
#   --expert-model-parallel-size <= 4 -> reject 5..N
#   --expert-tensor-parallel-size = 1 by convention in MoE tests -> reject 2..N
#   --pipeline-model-parallel-size = 1 -> reject 2..N
_LINT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("NUM_GPUS=8", re.compile(r"\bNUM_GPUS\s*=\s*8\b")),
    ("--actor-num-gpus-per-node 8", re.compile(r"--actor-num-gpus-per-node\s+8\b")),
    (
        "--rollout-num-gpus-per-engine >=5",
        re.compile(r"--rollout-num-gpus-per-engine\s+([5-9]|[1-9]\d+)\b"),
    ),
    (
        "--tensor-model-parallel-size >=5",
        re.compile(r"--tensor-model-parallel-size\s+([5-9]|[1-9]\d+)\b"),
    ),
    (
        "--context-parallel-size >=3",
        re.compile(r"--context-parallel-size\s+([3-9]|[1-9]\d+)\b"),
    ),
    (
        "--expert-model-parallel-size >=5",
        re.compile(r"--expert-model-parallel-size\s+([5-9]|[1-9]\d+)\b"),
    ),
    (
        "--expert-tensor-parallel-size >=2",
        re.compile(r"--expert-tensor-parallel-size\s+([2-9]|[1-9]\d+)\b"),
    ),
    (
        "--pipeline-model-parallel-size >=2",
        re.compile(r"--pipeline-model-parallel-size\s+([2-9]|[1-9]\d+)\b"),
    ),
]


def _discover_e2e_files() -> list[str]:
    return [
        f
        for f in glob.glob("tests/e2e/**/*.py", recursive=True)
        if not f.endswith("/conftest.py")
        and not f.endswith("/__init__.py")
        and "/sglang_patch/sglang_server.py" not in f
        and "/sglang/utils/" not in f
        and "short/test_dumper.py" not in f
    ]


def _h200_files() -> list[str]:
    e2e = _discover_e2e_files()
    return [r.filename for r in collect_tests(e2e, sanity_check=False) if r.suite == _H200_SUITE]


def _scan_file(path: str) -> list[tuple[str, str]]:
    with open(path) as f:
        text = f.read()
    hits: list[tuple[str, str]] = []
    for label, pat in _LINT_PATTERNS:
        for m in pat.finditer(text):
            hits.append((label, m.group(0)))
    return hits


def test_h200_suite_has_no_eight_gpu_anchors():
    files = _h200_files()
    assert files, (
        "stage-c-4-gpu-h200 has no registered tests — either the suite is empty "
        "or the discovery glob is broken; check tests/ci/run_suite.py and the "
        "(b)-class migration PR has landed."
    )

    failures: list[str] = []
    for path in files:
        hits = _scan_file(path)
        if hits:
            joined = "; ".join(f"{label} -> {snippet!r}" for label, snippet in hits)
            failures.append(f"{path}: {joined}")

    assert not failures, (
        "GPU-cardinality lint: the following files are registered to "
        f"{_H200_SUITE} but still contain anchors that exceed the 4-GPU "
        "budget:\n  " + "\n  ".join(failures)
    )


def test_h100_exclusion_files_still_trip_the_lint():
    """Specificity check: ensure the regex set is not vacuous.

    Files explicitly kept on stage-c-8-gpu-h100 by user spec must still
    contain at least one >=5-GPU reference each — otherwise the pattern set
    would have lost specificity and the positive test above becomes a no-op.
    """
    for path in _H100_EXCLUSIONS:
        hits = _scan_file(path)
        assert hits, (
            f"specificity check failed: {path} is an h100-exclusion file but "
            "no GPU-cardinality lint pattern matched. Either the file has "
            "been reduced (in which case it could move to h200) or the lint "
            "patterns have drifted from the real 8-GPU markers."
        )
