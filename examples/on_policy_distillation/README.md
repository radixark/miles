# On-Policy Distillation Examples

The canonical OPD documentation lives in
[`docs/advanced/on-policy-distillation.md`](../../docs/advanced/on-policy-distillation.md).
Keep the algorithm description, arguments, teacher-mode comparison, and
Rethinking OPD top-k recipe there so we do not maintain two copies.

This directory contains runnable examples:

- `run-qwen3-8B-opd.sh`: SGLang teacher server OPD. This script enables
  Rethinking OPD with `--opd-log-prob-top-k 16`, `--opd-top-k-strategy only-student`,
  `--opd-top-k-scoring-block-size 32`, and `--opd-reward-weight-mode student_p`.
- `run-qwen3-8B-opd-megatron.sh`: Megatron-loaded teacher OPD.

Use `--opd-log-prob-top-k 0` to run the original sampled-token OPD path.

The SGLang example uses Qwen3-8B as the student and Qwen3-32B as the teacher.
Use the same official script for the two control/treatment pairs:

```bash
# Student candidate set: legacy global union, then position blocks.
OPD_TOP_K_STRATEGY=only-student OPD_TOP_K_SCORING_BLOCK_SIZE=0 \
  bash examples/on_policy_distillation/run-qwen3-8B-opd.sh
OPD_TOP_K_STRATEGY=only-student OPD_TOP_K_SCORING_BLOCK_SIZE=32 \
  bash examples/on_policy_distillation/run-qwen3-8B-opd.sh

# Teacher candidate set: legacy global union, then position blocks.
OPD_TOP_K_STRATEGY=only-teacher OPD_TOP_K_SCORING_BLOCK_SIZE=0 \
  bash examples/on_policy_distillation/run-qwen3-8B-opd.sh
OPD_TOP_K_STRATEGY=only-teacher OPD_TOP_K_SCORING_BLOCK_SIZE=32 \
  bash examples/on_policy_distillation/run-qwen3-8B-opd.sh
```

`OPD_TOP_K`, `OPD_TOP_K_STRATEGY`, and `OPD_TOP_K_SCORING_BLOCK_SIZE` are environment overrides for controlled comparisons. Set `OPD_TOP_K_SCORING_BLOCK_SIZE=0` only when reproducing the legacy response-wide candidate union.

For `only-teacher`, blocked opposite-model rescoring snapshots the student router's current weight version and accepts the assembled rows only when every block reports that version. A version change discards the partial rows and retries the complete student-scoring transaction; retry exhaustion fails instead of emitting mixed-version scores. This makes the blocked path safe to compose with fully asynchronous rollout while preserving its normal stale-data semantics.
