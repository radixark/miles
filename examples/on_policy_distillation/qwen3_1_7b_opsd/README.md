# Privileged-context self-distillation (Qwen3-1.7B)

Train student LoRA adapters with forward KL against a frozen teacher that sees the
reference solution. Teacher and student start from the same base checkpoint; the
student generates from the problem alone.

Based on [Self-Distilled Reasoner](https://arxiv.org/abs/2601.18734). All OPSD logic
lives in this example. It uses existing custom reward, sample-conversion, and loss
hooks, with no core OPD changes or monkeypatches.

## Run

Requires a Miles training environment with Megatron and SGLang, plus `math_verify`.
Run from the repository root on a dedicated single node with at least four GPUs:

```bash
pip install math_verify
python -m examples.on_policy_distillation.qwen3_1_7b_opsd.run_qwen3_1_7b_opsd \
    --model-dir /root/models --data-dir /root/datasets --output-dir /root/shared_data
```

The launcher downloads Qwen3-1.7B, OpenThoughts-114k-math, and AIME24; converts the
checkpoint; and renders the datasets before training. The default eight-GPU layout
uses two student GPUs, five rollout GPUs, and one frozen-teacher GPU. The teacher
GPU is excluded from Ray's resources. The teacher starts after Ray initialization
and stops when training exits. Existing Ray clusters and `CUDA_VISIBLE_DEVICES`
overrides are intentionally unsupported by this single-node launcher.

## Implementation

`prepare_data.py` renders student prompts with thinking disabled and teacher/eval
prompts with thinking enabled. Reference solutions are only in teacher prompts.

`opsd.py` scores each generated response once with the privileged teacher. Its
conversion hook delegates standard sample handling to Miles and supplies existing
`target_tokens` and `loss_weights` fields. Each holds a flattened response-length
by top-k array of teacher token IDs or probabilities. The custom loss reshapes them
after sharding and batching, and reads student probabilities from training logits.

The objective is the unnormalized top-k sum of
`teacher_p * (log(teacher_p) - log(student_p))`. Clipping applies to each vocabulary
contribution before summing. Teacher probabilities retain their full-vocabulary
normalization; `opd_teacher_coverage` reports the retained mass.

Configure the example through `config.yaml`:

```yaml
opsd_top_k: 256
opsd_kl_coef: 1.0
opsd_kl_clip: 0.05
```

Set `opsd_kl_clip: null` for unclipped forward KL. Core `--use-opd` and advantage
computation stay disabled. This recipe requires Megatron, Ray object storage,
TP=CP=1, fixed global batch size, and the same teacher/student tokenizer.
Dynamic microbatch sizing remains enabled.

## Recipe and validation

The recipe uses lr 5e-6, batch 32, LoRA rank 64/alpha 128, 1024-token completions,
temperature 1.1, and 100 updates. Evaluation uses AIME24 Avg@12 with thinking enabled,
temperature 1.0, top-p 0.95, top-k -1, and 38912 new tokens. Boxed-answer grading
handles zero-padded labels and disables signal-based timeouts in reward threads.

CPU tests cover loss values and gradients, padding/masking, teacher-response
alignment, data transport, and launcher commands. This example-only implementation
has not been rerun in GPU training; no benchmark result is claimed for it.
