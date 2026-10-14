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
| `--opd-teacher-adapters` | Optional multi-teacher routing map over LoRA adapters (`NAME=ADAPTER` pairs, SGLang mode only). Each `ADAPTER` is a `lora_name` the teacher server was launched with, so specialist teachers share one server and one frozen base. Routes by the same key as `--opd-teacher-urls`, including `default`. |
| `--opd-teacher-key` | Metadata key holding the teacher name for routing (default: `opd_teacher`). |
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

### LoRA Teachers (SGLang mode only)

When every specialist teacher is a LoRA adapter over the *same* frozen base —
for example a shared `M0` plus separately trained coding, math and agentic
adapters — `--opd-teacher-adapters` routes by adapter instead of by endpoint.
All teachers then share one server, one base copy, and one `--rm-url`:

```bash
--rm-url http://<TEACHER_IP>:<TEACHER_PORT>/generate
--opd-teacher-adapters math=math_lora code=code_lora agentic=agentic_lora default=math_lora
--opd-teacher-key opd_teacher
```

The teacher server is launched once with the adapters resident:

```bash
python3 -m sglang.launch_server --model-path /path/to/M0 \
    --enable-lora \
    --lora-paths '{"lora_name":"math_lora","lora_path":"/path/to/adapters/math","pinned":true}' \
                 '{"lora_name":"code_lora","lora_path":"/path/to/adapters/code","pinned":true}' \
                 '{"lora_name":"agentic_lora","lora_path":"/path/to/adapters/agentic","pinned":true}' \
    --max-loras-per-batch 4 --max-lora-rank 32 --lora-backend triton
```

`--max-loras-per-batch` counts the base model as one slot and SGLang caps pinned
adapters at `max_loras_per_batch - 1`, so three pinned teachers need four slots.
Pinning is worth it here: teachers are few, hot and permanent, so keeping them
resident removes eviction and reload latency from the scoring path.

Routing semantics are unchanged — each sample is still scored by exactly one
teacher, scoring cost is identical to single-teacher OPD, and the loss does not
change. Two properties come for free with this layout:

- **Tokenizer agreement is automatic.** Teachers sharing the student's base
  cannot disagree with it about tokenization, which the URL-routed path requires
  but does not check.
- **N teachers cost one base copy plus N adapters**, not N model replicas.

The two maps compose: a name in `--opd-teacher-urls` gets its own endpoint, a
name in `--opd-teacher-adapters` alone is scored at `--rm-url` with that adapter,
and a name in both is scored at its own endpoint with that adapter. So a dense
specialist on its own server and a shared-base adapter can be sibling teachers
in one run.

Adapter directories written by the multi-LoRA Tinker gateway's
`save_weights_for_sampler` (see [LoRA Training and Serving](/advanced/lora)) are
directly loadable here; so is any PEFT adapter over the same base.

See `examples/on_policy_distillation/run-qwen3-8B-opd-lora-teachers.sh` for a
complete launcher.

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

## Preliminary Results

Using Qwen3-8B-Base model SFT-ed on part of the [OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) dataset, on-policy distillation with a Qwen3-32B teacher on the remaining data yields:

|                                  | Pass@1 |
|-----------------------------------------------|--------|
| Qwen3-8B-Base + SFT                           | 76%    |
| Qwen3-8B-Base + SFT + On-Policy Distillation  | 94%    |

## References

- [Thinking Machines: On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
- [Rethinking On-Policy Distillation](https://arxiv.org/abs/2604.13016)
