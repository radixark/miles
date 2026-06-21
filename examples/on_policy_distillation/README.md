# On-Policy Distillation Examples

The canonical OPD documentation lives in
[`docs/advanced/on-policy-distillation.md`](../../docs/advanced/on-policy-distillation.md).
Keep the algorithm description, arguments, teacher-mode comparison, and
Rethinking OPD top-k recipe there so we do not maintain two copies.

This directory contains runnable examples:

- `run-qwen3-8B-opd.sh`: SGLang teacher server OPD. This script enables
  Rethinking OPD with `--opd-log-prob-top-k 16`, `--opd-top-k-strategy only-student`,
  and `--opd-reward-weight-mode student_p`.
- `run-qwen3-8B-opd-megatron.sh`: Megatron-loaded teacher OPD.
- `run-qwen3.6-35B-A3B-glm5.2-cross-tokenizer.sh`: **cross-tokenizer** OPD — a
  GLM5.2 teacher (different tokenizer) distilled into a Qwen3.6-35B-A3B student via
  DPCA chunk alignment (`--opd-teacher-tokenizer`). The teacher runs as a separate
  SGLang server; see the "Cross-Tokenizer OPD" section of the canonical doc and the
  two-node topology in `k8s/cross-tokenizer-opd.yaml`.

Use `--opd-log-prob-top-k 0` to run the original sampled-token OPD path.
