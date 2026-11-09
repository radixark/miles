# CE-to-L1 distillation on language models

This research example runs independent single-GPU Hugging Face training workers
from the recorded Miles main revision. It does not use the Miles Ray/Megatron RL
trainer: keeping one frozen teacher beside each student permits many independent
full-distribution fits without adding full-vocabulary network transfers.

`data.py` creates prompt-disjoint UltraChat subsets and SGLang teacher continuations.
`loss.py` implements exact full-vocabulary probability CE/L1 with a chunked head
and recomputed backward. `probe.py` checks gradients, token mappings and entropy.
`train.py` fine-tunes every student parameter; `evaluate.py` performs validation
path integration and real sequential rejection/correction/bonus sampling.
`pipeline.py` assigns one process to each GPU and chooses learning rates centrally.
`entropy.py` audits positional entropy on held-out trajectories. `collect.py`
extracts compact seed/prompt metrics on each node after completion; `analyze.py`
computes paired bootstrap intervals and fitting curves without moving checkpoints.
`python -m examples.experimental.spec_draft_transition.probe_analysis` checks
the complete reporting path against known paired differences and zero interaction.
`probe_context.py` checks the sequential sampler against an exactly enumerable
teacher whose next-token probabilities depend on the last two tokens.
`probe_amp.py` independently checks BF16-autocast gradients against eager
autograd, including float32 student hidden states, near-matching distributions
and the real pretrained model heads. Custom forward saves the BF16 operands
actually used by the head, and backward recomputes the same probabilities.
`finalize.py` waits for completion, validates selected rates against actual run
configs, collects node metrics, updates W&B summaries and builds the final report.
After completion, `benchmark_queue.py` times the predeclared seed 20001
checkpoint in every condition. Three paired teacher/speculative replicates use
separate free GPUs, 128 shared prompts, a 256-token cap and alternating mode
order. These are reference implementation timings, not a production speed claim.

The manifest owns temperatures, seeds, initial rates, budgets and run identities.
Pure CE, pure L1 and the fixed cosine blend all use the same training token budget.
The blend is `w*CE + (1-w)*L1`, with
`w=(1+cos(pi*u/(T-1)))/2`, zero-based update `u`, and no loss rescaling.
Two dedicated tuning seeds select initial rates by mean validation expected K8
accepted-prefix length. Ten separate confirmation seeds then use the chosen rates.
A boundary rate winner pauses that condition pending an extended grid. Conditions
with completed, interior rate selection can begin confirmation while others tune.

The teacher is frozen Qwen3-4B, the student is pretrained Qwen3-0.6B, and teacher
temperatures are 0.6/1.0/1.4. All generation uses unrestricted sampling, non-thinking
prompts, 2,048 total tokens and at most 1,024 response tokens. Each temperature
gets its own generated corpus; all methods within that condition share its data.
The corpus contains 80,000 training, 2,000 validation and 2,000 test conversations.
Training draws 16,384 response positions per update for 2,000 updates. Students
use float32 parameters/Adam moments with bfloat16 autocast; inference is bfloat16.
The teacher output softmax is evaluated in float32 over every model output channel.
Checkpoint validation logs probability overlap, KL, CE and entropy on a fixed
token sample distributed across conversations. Final checkpoints are primary;
no best-test or best-checkpoint selection is permitted.

The primary test uses 2,000 held-out prompts and up to 32 sequential K8 verification
rounds, stopping at EOS. K4/K16 use the same first 256 test prompts as secondary
checks. Report accepted draft-prefix tokens per verification round, accounting
for EOS and teacher bonus tokens separately. The six smooth-minus-baseline
contrasts across temperatures require a familywise interval adjustment. Entropy
interaction contrasts and throughput are secondary; temperature changes both the
target probabilities and the teacher's visited histories.
The report also gives unadjusted secondary CE-minus-L1 contrasts, their
high-minus-low interaction, and accepted/emitted lengths at K4/K8/K16.

Runtime credentials belong outside the checkout. Only the data directory is
served for inter-node shard synchronization. Each run saves its full CLI config,
W&B URL, fitting curves and final checkpoint; the manifest and source revision
must be archived before launching fits. Keep artifacts on a sufficiently large
scratch volume and pin all model/dataset revisions in the runtime provenance.

Install dependencies with `uv pip install -r pyproject.toml` from this directory in a GPU environment that
already has a compatible SGLang/PyTorch runtime. Worker CLI fields use underscores;
run each module with `--help` for the complete argument list.
