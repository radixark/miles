---
title: On-Policy Distillation
description: Train a student on its own rollouts with a teacher's token-level probabilities as a reverse-KL signal, composable with GRPO, PPO, and other estimators.
---
On-policy distillation (OPD) trains a student model on its own rollouts while using a teacher model's token-level probabilities as the distillation signal. In Miles, the teacher signal is converted into a per-token reverse-KL penalty and applied after the selected RL advantage estimator has produced token advantages. This lets the same OPD recipe compose with GRPO, PPO, REINFORCE++, GSPO, and other estimators.

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--use-opd` | Enable on-policy distillation. Required flag to use OPD. |
| `--opd-type` | Type of OPD: `sglang` or `megatron`. Required when `--use-opd` is set. |
| `--opd-kl-coef` | OPD KL penalty coefficient (default: 1.0). Controls the weight of the distillation signal relative to the RL advantage. |
| `--opd-log-prob-top-k` | Number of top-k tokens retained for the Rethinking OPD token reward. `0` uses sampled-token OPD; `16` matches the paper recipe default. |
| `--opd-top-k-strategy` | Top-k token set strategy: `only-student`, `only-teacher`, `intersection`, `union`, or `xor`. |
| `--opd-reward-weight-mode` | Weighting scheme for top-k rewards: `student_p`, `teacher_p`, or `none`. |
| `--opd-teacher-urls` | Optional multi-teacher routing map (`NAME=URL` pairs, SGLang mode only). Routes each sample to a teacher by `sample.metadata[--opd-teacher-key]`; reserved name `default` is the fallback. Unset = single teacher at `--rm-url`. |
| `--opd-teacher-key` | Metadata key holding the teacher name for routing (default: `opd_teacher`). |
| `--opd-privileged-context-key` | Metadata key holding teacher-only context (SGLang mode). Appended to the last user message of the teacher's prompt; the student never sees it. Unset = disabled. |
| `--opd-teacher-load` | Path to teacher Megatron checkpoint. **Required** when `--opd-type=megatron`, **must not be set** when `--opd-type=sglang`. |
| `--opd-teacher-ckpt-step` | Optional checkpoint step for teacher model. |

## How It Works

OPD modifies the advantage computation by subtracting a KL penalty term that encourages the student to match the teacher's output distribution:

$$
\hat{A}_t = A_t - \lambda_{\text{opd}} \cdot D_{\text{KL}}(P_{\text{student}} \| P_{\text{teacher}})_t
$$

Where $A_t$ is the original advantage from the base estimator (e.g., GRPO), $\lambda_{\text{opd}}$ is `--opd-kl-coef`, and $D_{\text{KL}}$ is the token-level reverse KL divergence.

The implementation follows the additive OPD training recipe described in the [Thinking Machines OPD blog](https://thinkingmachines.ai/blog/on-policy-distillation/), with an additional SGLang top-k reward mode from [Rethinking On-Policy Distillation](https://arxiv.org/abs/2604.13016).

## Rethinking OPD Top-K Reward

SGLang OPD supports the top-k token reward recipe from [Rethinking On-Policy Distillation](https://arxiv.org/abs/2604.13016). Set `--opd-log-prob-top-k` above zero to request student rollout top-logprobs, score the same sequence with the teacher, and aggregate a weighted reverse-KL estimate over a selected token set at each response position.

The token set is controlled by `--opd-top-k-strategy`:

| Strategy | Token set |
|----------|-----------|
| `only-student` | Student top-k tokens, with teacher logprobs queried for those IDs. |
| `only-teacher` | Teacher top-k tokens, with student logprobs queried for those IDs. |
| `intersection` | Tokens appearing in both top-k sets. |
| `union` | Tokens appearing in either top-k set, with duplicates removed. |
| `xor` | Tokens appearing in exactly one top-k set. |

`--opd-reward-weight-mode` controls whether each selected token is weighted by student probability, teacher probability, or uniformly. For compatibility, `--opd-log-prob-top-k=0` keeps the original sampled-token OPD path.

## Two Teacher Modes

### SGLang Mode (`--opd-type sglang`)

The teacher runs on an external SGLang server. Teacher log-probs are obtained during the rollout phase.

**When to use**: The teacher has a different architecture from the student, or the teacher is too large to load alongside the training model.

**How it works**:
1. An external SGLang server runs the teacher model.
2. During rollout, the custom reward function (`miles.rollout.on_policy_distillation.reward_func`) sends each sample to the teacher server to obtain token-level log-probs.
3. With `--opd-log-prob-top-k=0`, the custom post-processing function trims sampled-token teacher log-probs to the response span and stores them in `sample.teacher_log_probs`.
4. With `--opd-log-prob-top-k>0`, it computes the Rethinking OPD weighted top-k reverse-KL estimate and stores it in `sample.opd_reverse_kl`.
5. During training, the stored OPD penalty is subtracted from the selected estimator's advantages.

**Configuration**:
```bash
--use-opd
--opd-type sglang
--opd-kl-coef 1.0
--opd-log-prob-top-k 16
--opd-top-k-strategy only-student
--opd-reward-weight-mode student_p
--custom-rm-path miles.rollout.on_policy_distillation.reward_func
--custom-reward-post-process-path miles.rollout.on_policy_distillation.post_process_rewards
--rm-url http://<TEACHER_IP>:<TEACHER_PORT>/generate
```

### Multi-Teacher Routing (SGLang mode only)

`--opd-teacher-urls` routes each sample to a task-specific teacher, e.g. a math
specialist for math prompts and a code specialist for code prompts. Each sample
is still scored by exactly one teacher, so scoring cost is identical to
single-teacher OPD and the loss is unchanged — `teacher_log_probs` /
`opd_reverse_kl` are per-sample and do not care which teacher produced them.

**How it works**:
1. Tag each prompt with a teacher name in its dataset metadata column
   (read via `--metadata-key`, default `metadata`):
   ```json
   {"prompt": "...", "metadata": {"opd_teacher": "math"}}
   {"prompt": "...", "metadata": {"opd_teacher": "code"}}
   ```
2. Map names to teacher endpoints with `--opd-teacher-urls NAME=URL ...`.
   The reserved name `default` is the fallback for samples whose name is
   missing or unknown; without a `default` entry such samples raise an error
   (failing loudly beats silently distilling from the wrong teacher).
3. `reward_func` resolves the URL per sample; everything downstream is
   unchanged.

**Configuration** (on top of the SGLang-mode flags above; `--rm-url` is ignored
when the routing map is set):
```bash
--opd-teacher-urls math=http://<H1>:<P1>/generate code=http://<H2>:<P2>/generate default=http://<H1>:<P1>/generate
--opd-teacher-key opd_teacher   # metadata key holding the teacher name (default)
```

> **Notes**: All teachers must share the student's tokenizer — scoring sends
> `input_ids` and gathers per-token-id log-probs. Works with both the
> sampled-token path and the top-k path (student-side scoring still goes to the
> student router). For throughput, point multiple names (or one name backed by
> an sglang router) at replicas; `--opd-teacher-urls` is for *different*
> teachers, not load balancing.

### Privileged Context (SGLang mode only)

`--opd-privileged-context-key` lets the teacher see information the student does
not, following [Self-Distilled Reasoner: On-Policy Self-Distillation for Large
Language Models](https://arxiv.org/abs/2601.18734). The student generates from the public prompt as usual; the teacher then
scores *that same response* on a prompt carrying private context: a hint, a
reference solution, a grader's correction. The resulting log-probs are a sharper
target than the teacher could produce unaided, while the student keeps learning
a policy it can run without the extra context at inference time.

The metadata value is a plain string appended verbatim to the end of the last
user message, so the wording of the hint lives in your dataset rather than
hard-coded in miles.

This works across all three forms `sample.prompt` can take (see
`miles/utils/data.py`): a **message list**, a **template-rendered string** (with
`--apply-chat-template`), or a **raw untemplated string** (without it, from a
plain-text prompt column). For a message list, miles appends to the last message
and renders. For a raw string there is no template structure, so the context goes
on the end. For a rendered string, it derives that template's closing
sequence, e.g. `<|im_end|>\n<|im_start|>assistant\n` for ChatML,
`<end_of_turn>\n<start_of_turn>model\n` for Gemma, by rendering a probe message
and splitting on it, then splices the context in just before it. Nothing is
hard-coded per model, and probed through the same helper that rendered the prompt
so DeepSeek and Inkling checkpoints (which use different renderers) agree. Which
form a string is in is verified rather than inferred: a prompt ending in that
sequence is spliced whatever `--apply-chat-template` says, and with the flag on a
prompt that does not end in it raises rather than being mangled. The message-list and
rendered routes produce byte-identical text.

**How it works**:
1. Rollout is untouched: the student sees only the public prompt.
2. At scoring time, `sample.metadata[--opd-privileged-context-key]` is inserted at
   the end of the teacher's last user message, and the student's response tokens
   are appended verbatim.
3. The teacher scores that sequence. Because the response stays at the tail,
   the usual response-span extraction is unchanged.
4. When rendering from messages, the result is checked to confirm the context
   survived the template, since Gemma-2 applies `content | trim` and can drop it.

Samples without the metadata key are scored normally, so one dataset can mix
privileged and plain examples.

The reverse-KL estimate is aligned by *response position*, and the response
tokens are identical on both sides, so only the teacher's conditioning prefix
differs. Under top-k, the student is still re-scored on the prompt it actually saw;
only the teacher reads the privileged context.

**Configuration** (on top of the SGLang-mode flags above):
```bash
--opd-privileged-context-key opd_privileged_context
```

> **Notes**: The teacher prompt is longer than the student's, so a
> near-context-limit sample can overflow `--rollout-max-context-len`. Such a sample
> is scored without privileged context rather than raising, since an exception there
> would kill the run, and the count of samples actually scored with context is logged.
> The rendered path assumes the prompt ends in a user turn; a conversation ending in
> another role renders the same way under ChatML and the context is appended to that
> final turn. The metadata value must
> be a non-empty string. If the configured key matches no samples, a warning is
> logged with the observed count rather than the run silently degrading to ordinary
> self-distillation. A chat template that rewrites message content (rather
> than only wrapping it) cannot be spliced safely; with
> `--apply-chat-template` on that raises on the first privileged sample, not at
> startup.

### Megatron Mode (`--opd-type megatron`)

The teacher model is loaded directly into Megatron via `--opd-teacher-load`. Teacher log-probs are computed during the training forward pass.

**When to use**: The teacher has the same architecture as the student/reference model and fits in GPU memory.

**How it works**:
1. The teacher model is loaded as an additional Megatron model during initialization.
2. During the training forward pass, the teacher model computes log-probs for each sample.
3. The KL penalty is computed inline and applied to advantages.

**Configuration**:
```bash
--use-opd
--opd-type megatron
--opd-kl-coef 1.0
--opd-teacher-load /path/to/teacher_torch_dist
```

> **Note**: The teacher checkpoint must be in Megatron format (`torch_dist` or `torch`). You can convert from HuggingFace format using `tools/convert_hf_to_torch_dist.py`.

## Running the Examples

Complete example scripts are provided in `examples/on_policy_distillation/`:

### SGLang Teacher

```bash
# 1. Download models and data
hf download Qwen/Qwen3-32B --local-dir /root/Qwen3-32B
hf download Qwen/Qwen3-8B --local-dir /root/Qwen3-8B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k

# 2. Convert student model
cd /root/miles
MODEL_ARGS_LINE="$(python3 miles/utils/external_utils/model_args_utils.py qwen3-8B)" || exit 1
read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-8B \
    --save /root/Qwen3-8B_torch_dist

# 3. Run
bash examples/on_policy_distillation/run-qwen3-8B-opd.sh
```

### Megatron Teacher

```bash
# 1. Convert both student and teacher models to Megatron format
# 2. Run
bash examples/on_policy_distillation/run-qwen3-8B-opd-megatron.sh
```

### Privileged-Context Self-Distillation

Same prerequisites as the SGLang teacher above, minus the 32B download, since the
teacher server runs the same Qwen3-8B checkpoint the student starts from. The
script derives its own privileged dataset from `dapo-math-17k` by attaching each
problem's verified answer as teacher-only metadata.

```bash
bash examples/on_policy_distillation/run-qwen3-8b-opsd.sh
```

## Preliminary Results

Using Qwen3-8B-Base model SFT-ed on part of the [OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) dataset, on-policy distillation with a Qwen3-32B teacher on the remaining data yields:

|                                  | Pass@1 |
|-----------------------------------------------|--------|
| Qwen3-8B-Base + SFT                           | 76%    |
| Qwen3-8B-Base + SFT + On-Policy Distillation  | 94%    |

## References

- [Thinking Machines: On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
- [Rethinking On-Policy Distillation](https://arxiv.org/abs/2604.13016)
