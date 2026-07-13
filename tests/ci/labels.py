"""Canonical CI label registry.

Tests declare a domain label set in `register_cuda_ci(..., labels=[...])` and
`register_cpu_ci(..., labels=[...])`. The PR-side trigger for each label is
`run-ci-<key>`: each entry below MUST have a matching `run-ci-<key>` label in
the GitHub repo (maintainer-managed).

Adding a new label:
1) Add an entry below.
2) Create the matching `run-ci-<key>` label in GitHub repo Settings -> Labels.
   The workflow does not need editing -- the generic stage job filters tests
   by labels at runtime.

The workflow-only scope labels are intentionally NOT listed here. `run-ci-all`
uses `--match-all-labels` without exclusions; `run-ci-image` and nightly pair
it with scope-specific `--exclude-labels` values (handled in run_suite.py).
"""

KNOWN_LABELS: dict[str, str] = {
    "megatron": "Megatron-LM training tests",
    "model-scripts": "Model launch script smoke tests",
    "sglang": "SGLang patch / equivalence tests",
    "fsdp": "FSDP training tests",
    "short": "Short 8-GPU smoke tests",
    "long": "Long-running training tests",
    "ckpt": "Checkpoint save / load tests",
    "lora": "LoRA training tests",
    "precision": "Numerical precision parity tests",
    "ft-short": "Fault-tolerance trainer comparison tests (no_failure / deterministic / with_failure)",
    "ft-long": "Fault-tolerance trainer soak tests (random-crash survival, realistic-gsm8k convergence)",
    "weight-update": "Weight update tests",
    "replay": "Routing / indexer replay tests",
    "qwen35": "Qwen3.5-35B-A3B MTP / spec-v2 e2e tests",
}
