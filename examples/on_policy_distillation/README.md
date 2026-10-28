# On-Policy Distillation Examples

The canonical OPD documentation lives in
[`docs/advanced/on-policy-distillation.md`](../../docs/advanced/on-policy-distillation.md).
Keep the algorithm description, arguments, teacher-mode comparison, and
Rethinking OPD top-k recipe there so we do not maintain two copies.

This directory contains runnable examples:

- `run-qwen3-8B-opd.sh`: SGLang teacher server OPD. This script enables
  Rethinking OPD with `--opd-log-prob-top-k 16`, `--opd-top-k-strategy only-student`,
  and `--opd-reward-weight-mode student_p`.
- `run-qwen3-8B-opd-multi-teacher.sh`: Multi-teacher OPD with per-sample routing.
  Math prompts are scored by a Qwen3-32B teacher and code prompts by a
  Qwen3-Coder-30B-A3B teacher, selected via `--opd-teacher-urls` and a per-row
  `{"metadata": {"opd_teacher": ...}}` tag in the dataset.
- `run-qwen3-8B-opd-megatron.sh`: Megatron-loaded teacher OPD.
- [`mopd_reasoning_gym/`](mopd_reasoning_gym/README.md): Qwen3.6-35B-A3B distilled from two
  reasoning_gym GRPO teachers (caesar_cipher, simple_geometry) with sampled-token OPD, launched by
  `scripts/run_qwen3_6_35b_a3b_mopd.py` on two B200 or GB300 nodes, synchronous or one-step overlap.

Use `--opd-log-prob-top-k 0` to run the original sampled-token OPD path.
