# Validation-triggered hard/soft CE → L1 distillation

This extension reuses the frozen Qwen3-4B → Qwen3-0.6B teacher-generated
UltraChat corpus and existing CE, L1, cosine-transition and constant-mixture
baselines. It runs independent single-GPU fits on shi-h200-2 (8 H200).
Teacher temperatures: 0.6, 1.0, 1.4; student temperature: 1.
Full vocabulary: 151,936. No entropy filtering, top-k or top-p truncation.
Training: 2,000 updates × 16,384 response positions; microbatch 2; FP32 master
weights/Adam moments and BF16 operations; gradient checkpointing; AdamW
(0.9,0.999), epsilon 1e-8, weight decay .01. No gradient clipping.
LR: 100-step warmup then cosine to 1% peak, unchanged across the loss switch.
Teacher and student model/tokenizer revisions and all run IDs are in manifest.json.

## Loss and switch rule (fixed before training)

Before the switch: 0.1 hard CE + 0.9 soft CE.
After the switch: 0.9 L1 + 0.1 soft CE.
Hard CE uses the exact stored teacher-sampled next token at each prefix,
including the first response token predicted from the final prompt token.
Soft CE uses the full frozen teacher distribution at that temperature.
L1 is the sum of absolute probability differences over vocabulary (range 0–2),
averaged over response positions; no coefficient rescaling.

Every 100 updates, evaluate full-distribution L1 on the same fixed held-out
validation positions (budget 4,096 positions, up to 32 spread per conversation).
Write B(s) for the best measured L1 through step s. At the earliest check
500 <= s < 2000 where (B(s−300)−B(s))/B(s−300) < 0.005, switch for the next
training update. This requires less than 0.5% relative best-L1 improvement
over 300 updates. The switch is one-way; optimizer state and global LR schedule
are retained. There is no forced switch if the criterion never fires.
Each run writes every decision, best L1, relative improvement, and switch step
to metrics.jsonl and transition.json, with the same records logged to W&B.
No test data determine the switch or learning rate.

## Tuning and evaluation

Tune the new method independently on seeds 10001/10002, initially peak LR
1e-5, 3e-5, 1e-4; extend a boundary winner using pre-reserved unique run IDs.
Select by the same validation expected accepted-prefix metric (K=8,
128 prefixes × 4 proposal paths), then train ten fresh confirmation seeds
20001–20010 per temperature. All runs retain the same 2,000-update budget.
The additional validation checks are included when reporting wall time.
Confirmation: actual sequential speculative rejection sampling, K=8,
2,000 held-out prompts, at most 32 rounds, stop at EOS; K=4/16 secondary on
the first 256 prompts. Accepted proposals, residual corrections and teacher
bonus tokens all update history. Report accepted draft proposals per round,
not a production serving speedup. Reuse paired baseline seeds and prompt IDs.
Plots show mean ± one bootstrap standard error; inference uses paired crossed
seed/prompt bootstrap intervals and labels its multiplicity correction.

Run `python -m examples.experimental.spec_draft_transition.preflight_adaptive`
on CUDA before any training. This checks teacher-label shifts, plateau/no-plateau
cases, the irreversible switch and both stage losses/gradients through BF16 AMP.
The existing pyproject already declares all dependencies; no new package is needed.
