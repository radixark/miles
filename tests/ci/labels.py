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

The meta-labels `run-ci-image` / `run-ci-all` are intentionally NOT listed
here: they bypass the per-test labels filter and run the full suite via the
`--match-all-labels` flag (handled in run_suite.py).
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
}
