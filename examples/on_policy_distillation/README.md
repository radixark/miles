# On-Policy Distillation Examples

The canonical OPD documentation lives in
[`docs/advanced/on-policy-distillation.md`](../../docs/advanced/on-policy-distillation.md).
Keep the algorithm description, arguments, teacher-mode comparison, and
Rethinking OPD top-k recipe there so we do not maintain two copies.

This directory contains runnable examples:

- `run-qwen3-8B-opd.sh`: SGLang teacher server OPD. This script enables
  Rethinking OPD with `--opd-log-prob-top-k 16`, `--opd-top-k-strategy only_stu`,
  and `--opd-reward-weight-mode student_p`.
- `run-qwen3-8B-opd-megatron.sh`: Megatron-loaded teacher OPD.

Use `--opd-log-prob-top-k 0` to run the original sampled-token OPD path.
