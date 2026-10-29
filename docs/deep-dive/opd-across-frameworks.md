---
title: On-policy distillation in Miles, open-instruct and Prime-RL
description: Where each framework runs the OPD teacher, how it handles a teacher of another size and several teachers, and how the teacher signal enters the loss, traced through the TMax OPD port.
---
This page compares how three RL frameworks implement on-policy distillation (OPD): the student samples its own
rollouts, a frozen teacher scores every sampled token, and the per-token reverse KL
$\log \pi_{\text{student}} - \log \pi_{\text{teacher}}$ is the training signal. It describes the designs; the TMax port's
use of Miles OPD is in [TMax Terminal-RL](/examples/tmax) (Findings → On-policy distillation), its open work in the
port's [roadmap](https://github.com/shatu/miles/blob/shashank/tmax-terminal-rl/examples/tmax/ROADMAP.md), and the
investigation that prompted this page in its
[INVESTIGATIONS.md](https://github.com/shatu/miles/blob/shashank/tmax-terminal-rl/examples/tmax/INVESTIGATIONS.md).

- **Miles:** branch `shashank/tmax-opd` (v0.1.3 OPD code; upstream main has no OPD changes since v0.1.3 as of
  2026-10-26), plus the port's launcher (`examples/tmax/run_tmax_qwen3_5.py`) and generate wrapper
  (`examples/tmax/generate.py`). Upstream docs: [On-policy distillation](/advanced/on-policy-distillation).
- **open-instruct:** the terminal OPD runs on branch `opd` (`scripts/general_agent/terminal/rl/opd/`,
  `open_instruct/grpo_fast.py`), the `opd_maskreason`, `mopd` and `opd_cuda13_*` branches.
- **Prime-RL:** `PrimeIntellect-ai/prime-rl` at `eca75ea` (2026-10-26): `src/prime_rl/orchestrator/algo/opd.py`,
  `src/prime_rl/trainer/rl/loss.py`, `docs/algorithms.md`.

## At a glance

| | Miles | open-instruct | Prime-RL |
|---|---|---|---|
| Where the teacher runs | `--opd-type megatron`: in the trainer, a CPU weight copy swapped into the student's own Megatron model for one forward pass per step. `--opd-type sglang`: an SGLang server, scored at rollout time | In the trainer: a second Hugging Face model on every learner (ZeRO-3, no CPU offload), one no-grad forward pass per step | An external vLLM endpoint (`name` + `base_url`) that the user deploys; scored by the orchestrator as each episode finishes |
| Teacher of another size | SGLang mode only (Megatron mode needs the student's exact architecture) | Yes: any size; with Ulysses SP and a different attention geometry every rank scores the full unsplit sequence | Yes: any model with the student's tokenizer |
| Who places the teacher | The TMax launcher: SGLang servers as Ray actors on the rollout nodes (`opd_teacher_gpus`) | Shares the learner GPUs | Nobody: the `rl` entrypoint only logs that frozen endpoints must already be serving |
| Several teachers | Per-sample routing by a metadata key (`--opd-teacher-urls`, SGLang mode only) | `mopd` branch: routing, mixture, max and min over teachers | Per-environment: each task source can run its own `opd` with its own teacher |
| Student log-prob in the signal | The rollout's (behavior policy) with `--use-rollout-logprobs`, else the trainer's recompute before the update | The engine's (`--use_vllm_logprobs`); `opd_cuda13_*` adds `--opd_student_logprobs learner` | The trainer's live log-prob, inside the loss |
| How the signal enters the loss | Subtracted from the GRPO advantage, $A - c\,(\log \pi_\text{student} - \log \pi_\text{teacher})$, then the policy loss (DPPO in TMax) | The same | Its own `ref_kl` loss component: $-(\log \pi_\text{teacher} - \log \pi_\theta)$, detached, times the importance ratio $\pi_\theta / \pi_\text{inference}$ |
| Trust region | DPPO binary-TV mask (threshold 0.1) on the policy loss | DPPO binary-TV mask (0.1) | One-sided: drop tokens whose probability fell more than 0.2 below the inference probability |
| Clip on the OPD term | None | None on `opd`; `opd_adv_clip` on `mopd` | None |
| Mixing with RL rewards | Additive OPD (`--opd-kl-coef` on top of GRPO), or pure (TMax `opd_pure`) | `--opd_pure` or additive | Per environment: GRPO and OPD environments in one batch, each loss component normalized by its own token count |
| Sampling restrictions | None enforced | None enforced | Rejects OPD with top-p / top-k sampling: teacher log-probs are full-vocabulary, the trainer's would be renormalized |

## Where the teacher runs, and teachers of another size

**Miles, Megatron mode** (`miles/backends/megatron_utils/actor.py:430`, `:711`). The teacher is not a second model:
`load_other_checkpoint("teacher", …)` loads the teacher's `torch_dist` checkpoint into the student's own Megatron
model and keeps a pinned CPU copy, exactly as for the reference model (`"ref"`). Each step `_switch_model("teacher")`
copies it in, computes `teacher_log_probs`, and switches back. It costs no extra GPU memory, but the teacher's weights
must fit the student's tensors, which is why the upstream docs restrict it to "the same architecture as the
student/reference model". Upstream PR #2295 (open, last updated 2026-08-18) moves this teacher onto its own nodes as
a separate Ray actor group, still with the student's architecture (`copy.deepcopy(args)`, and it ships last-layer
hidden states for the student's output layer, which needs an equal hidden size).

**Miles, SGLang mode** (`miles/rollout/on_policy_distillation.py:355`). Miles sends each finished sample's token ids
to an SGLang server (`--rm-url`) and reads the log-prob of every response token. Upstream runs this as the reward
function; with session server v2 the agent's reward is already set and Miles never calls the reward function for
agentic samples, so the TMax port scores in a generate-function wrapper instead (`examples/tmax/generate.py:35`).
The port's launcher starts the servers (`examples/tmax/run_tmax_qwen3_5.py:618`): `opd_teacher_gpus` SGLang
processes of one GPU each (SGLang TP > 1 breaks Qwen3.5), each a detached Ray actor holding its GPU, on the last nodes
in Miles' placement order, so training keeps its whole nodes and the teacher GPUs come out of the rollout side. The
wrapper sends each sample to the least busy server (`examples/tmax/generate.py:91`).

**open-instruct** (`grpo_fast.py`, the `opd_teacher_model_name_or_path` branch of the learner setup). The teacher is a
second model on every learner, loaded like the reference policy, so any size works. When its attention geometry
differs from the student's under Ulysses SP, every SP rank runs the full unsplit sequence and reshards the log-probs
(`_compute_teacher_logprobs_full_sequence`). The teacher costs learner memory and one forward pass per step; it
needs no extra GPUs.

**Prime-RL** (`configs/algorithm.py:47`, `entrypoints/rl.py:192`). Prime-RL hosts only the trainable policy. A teacher
is a `FrozenModelConfig`: a served model name and a `base_url`. Prime-RL never launches, scales or updates it; one
endpoint per teacher, with any router behind it. The orchestrator's `OPDAlgorithm.score_episode` prefill-scores each
trainable branch's full token ids under the teacher (vLLM `prompt_logprobs`) and attaches the log-probs of the
sampled tokens to the trace (`orchestrator/algo/opd.py`, `algo/routing.py`). Teacher scoring is never stale: frozen
models keep a stable prefix cache.

## Several teachers

- **Miles:** `--opd-teacher-urls NAME=URL …` routes each sample, by `sample.metadata[--opd-teacher-key]`, to exactly
  one teacher, with a `default` fallback (`on_policy_distillation.py:45`); SGLang mode only. Open upstream PRs: #3116
  (candidate-level multi-teacher OPD over top-k), #1322 (teacher ensembles), #3887 (teachers as LoRA adapters).
- **open-instruct:** the `mopd` branch combines teachers per token (mixture, max, min) or routes per sample, with
  per-teacher KL and routing-fraction metrics and an OPD advantage clip.
- **Prime-RL:** per-environment algorithms (`docs/algorithms.md`, Per-Env Algorithms). An environment that sets
  `algo.type = "opd"` with its own `algo.teacher` distills from that teacher while others in the same run train with
  GRPO (`configs/debug/algo/mixed_grpo_opd.toml`). Each sample has one teacher; there is no per-token combination.

## How the teacher signal enters the loss

**Miles and open-instruct.** The reverse KL is computed once, before training, and folded into the advantage:
`advantages[i] = adv - opd_kl_coef * (old_student_log_prob - teacher_log_prob)`
(`miles/backends/training_utils/loss/hub/opd.py:89`). Under `--use-rollout-logprobs` the student log-prob is the
rollout's (`loss/objective.py:56`), the policy that generated the sample, so with fully async staleness of up to
`max_weight_staleness` versions a token keeps the advantage it had when it was sampled. A token the teacher rejects
(teacher log-prob around -36 on a token the student sampled at -0.5) keeps an advantage of about -35 across every
update it is trained in, even after earlier updates have pushed it down. Without `--use-rollout-logprobs` the
student term is the trainer's recompute (`log_probs`), but then the rollout log-probs also stop anchoring the policy
loss's ratio, which removes DPPO's correction for stale samples. open-instruct's `--opd_student_logprobs learner`
(on `opd_cuda13_*`) is the same choice.

**Prime-RL** (`trainer/rl/loss.py:288`). OPD is a separate loss component, `ref_kl`, that reads no advantage:

$$
\mathcal{L}_{ref\_kl} = -\operatorname{sg}\big[\log \pi_\text{teacher} - \log \pi_\theta\big]\cdot
\frac{\pi_\theta}{\pi_\text{inference}}
$$

with $\log \pi_\theta$ the trainer's live log-prob, the importance ratio capped at $10^4$, and tokens with
$\pi_\theta - \pi_\text{inference} < -0.2$ masked (`:307`). The signal shrinks as the trainer lowers a rejected token,
while the ratio still corrects for staleness and trainer/inference mismatch. The total loss is
$\sum\mathcal{L}_{rl}/N_{rl} + \sum\mathcal{L}_{ce}/N_{ce} + \sum\mathcal{L}_{ref\_kl}/N_{ref\_kl}$, each $N$ the
global count of that component's tokens, so OPD tokens neither dilute nor inflate GRPO's per-token step. Token
staleness beyond `orchestrator.max_staleness` (default 8) is masked out of the loss.

## What it means in practice

- **The KL to the base model spikes under pure OPD.** Miles logs it as `train/kl_loss` (k3, coefficient 0;
  open-instruct's `objective/kl2_avg`), and k3 is unclamped (`loss/hub/math_utils.py:168`): a token whose log-prob an
  update lowered by $d$ nats adds about $e^d$. The OPD advantages of the few tokens the teacher rejects outright (down
  to -36, about 0.01% of tokens) drive exactly such drops. Both frameworks show it early in a run: open-instruct's
  9B OPD run logged 10,911 at step 6, Miles' 22,571 at step 3. It then settles; it is not a weight-loading bug (the
  dumped reference and rollout log-probs agree to 0.007 per token).
- **Early training goes through a transient** with advantage-based OPD. In open-instruct's 9B run the
  train/rollout gap rose about 10× (to 0.15), the DPPO mask grew to about 10% of tokens and the teacher's log-prob on
  student samples dipped (to -0.96) around steps 10 to 22, then settled by steps 25 to 30. Miles' 9B run shows the
  same rise from step 9 (gap 0.18, 11% masked at step 17), with training entropy up from 0.3 to 0.9. The stale student
  term above is a plausible driver; Prime-RL's live-log-prob loss is the design that avoids it.
- **A teacher of another size is a deployment question everywhere but open-instruct.** Miles' SGLang mode and Prime-RL
  both score with a separate inference server; Prime-RL leaves its deployment to the user, the TMax launcher places
  it on the rollout nodes. Only open-instruct runs a teacher of another size on the training GPUs.
