# Qwen3.5-35B-A3B Self-Distillation on a Single Node (RLVR teacher → OPD)

A reproducible two-phase on-policy-distillation (OPD) example for the
**Qwen3.5-35B-A3B** MoE on a **single 8×H200 node**, using the **in-process
Megatron teacher** (`--opd-type megatron`, no separate teacher server).

It differs from the sibling examples in three ways:

1. **Real MoE at scale on one node.** The 2-node/16-GPU recipe is re-tiled to 8 GPUs.
2. **A genuinely diverged teacher.** `run-qwen3-8B-opd-megatron.sh` uses `teacher == base`
   (a mechanism demo where the reverse-KL is ~0). Here Phase 1 *trains* the teacher
   with RLVR so it is measurably better and more concise than the base — the
   prerequisite for OPD to actually move the student.
3. **Self-distillation is the only valid option here.** Qwen3.5 has its own tokenizer
   (vocab 248320); the smaller Qwen3 models (vocab 151936) are not token-compatible,
   so a cross-model teacher would be invalid. Teacher and student are the same family.

## Pipeline

```
Phase 1  (phase1_rlvr_teacher.sh)         Phase 2  (phase2_opd_selfdistill.sh)
base 35B  --RLVR (GRPO, lr 1e-5)-->  teacher        base 35B (student)
          better + concise                     |  <-- reverse-KL (--opd-type megatron)
          (eval 0.83 -> 0.89)                  teacher (Phase-1 ckpt, in-process)
```

## Single-node parallelism (world = 8)

The original recipe was 2 nodes × 8 GPUs (`TP2 PP1 CP2 EP8 ETP1`, DP4). On one node
we keep the same dims and only halve DP:

| dim | value | check |
|-----|-------|-------|
| TP  | 2 | decoder `TP*PP*CP = 2` ; `8 % 2 = 0` → DP = 4 |
| PP  | 1 | |
| CP  | 2 | shards the long (~17k) sequence so 24k context fits |
| EP  | 8 | `num_experts 256 % 8 = 0` ; expert `ETP*EP*PP = 8` → expert_dp = 1 |
| ETP | 1 | expert_dp(1) ≠ dp(4) is allowed (miles rank order ends in `pp`) |

`--colocate` time-shares the train and rollout phases (each fits 143 GB separately,
not summed); `--optimizer-cpu-offload` puts Adam state on host RAM; the model is a
hybrid linear-attention MoE so the KV cache is small. Peak ≈ 124 GB / 143 GB per GPU.

## Reproduce

**0. Prereqs** — model + torch_dist checkpoint, and the train/eval split:

```bash
# model (and mcore conversion, see ../README.md for convert_hf_to_torch_dist usage)
#   ${MODEL_DIR}/Qwen3.5-35B-A3B  and  ${MODEL_DIR}/Qwen3.5-35B-A3B_torch_dist
# disjoint, seeded train/eval split (eval is held out from BOTH phases):
python make_split.py --src /path/to/dapo-math-17k.jsonl --out-dir ${DATA_DIR}
#   -> ${DATA_DIR}/dapo_train.jsonl (16886)  ${DATA_DIR}/dapo_eval.jsonl (512)
```

**1. Phase 1 — train the teacher** (watch `rollout/raw_reward` climb and
`eval/dapo_heldout` rise above the base ~0.83):

```bash
MODEL_DIR=... DATA_DIR=... OUTPUT_DIR=/persistent/ckpt-teacher \
  bash phase1_rlvr_teacher.sh
```

**2. Phase 2 — distill the teacher into the base student**:

```bash
# pure OPD (default): training reward = 0, only the teacher reverse-KL drives learning
TEACHER_LOAD=/persistent/ckpt-teacher DATA_DIR=... \
  bash phase2_opd_selfdistill.sh

# grounded OPD: correctness reward (raw_reward == accuracy, climbs) + teacher reverse-KL
MODE=grounded TEACHER_LOAD=/persistent/ckpt-teacher DATA_DIR=... \
  bash phase2_opd_selfdistill.sh
```

`OUTPUT_DIR` / the teacher checkpoint must live on **persistent** storage. On a
KubeRay pod the head can be recreated and wipe the container overlay (`/root`); a
node-local disk (e.g. `/node_public`) survives and makes runs resumable.

## Results (DAPO-math, held-out 512, eval @ 24k cap, temp 0.6)

**Phase 1 — RLVR teacher** (lr 1e-5):

| step | eval/dapo_heldout | eval response length |
|------|-------------------|----------------------|
| 0 (base) | 0.828 | 14,070 |
| 5        | **0.887** | **6,248** |

The teacher becomes both more accurate **and** ~2× more concise.

**Phase 2 — pure OPD** (student = base, teacher = Phase-1 step-5 ckpt; reward = 0):

| step | eval/dapo_heldout | eval response length | opd_reverse_kl |
|------|-------------------|----------------------|----------------|
| 0 (base) | 0.840 | 14,070 | — |
| 5        | 0.852 | **6,132** | 0.045 → 0.013 |

With **zero task reward**, pure reverse-KL distillation transfers the teacher's
concise behavior to the base student — eval length **−57%** with accuracy
preserved/slightly up (the +1.2 pt is within the ~1.6 pt eval SE; the robust,
headline effect is the efficiency transfer). A nonzero, shrinking `opd_reverse_kl`
confirms the teacher genuinely differs from the student and the student is
converging onto it.

## Gotchas (each cost a wasted run to find)

- **Reward grader.** `--rm-type deepscaler` requires a `</think>` tag and returns 0
  otherwise; Qwen3.5 reasons inline (no tag) → every reward 0. `--rm-type math`
  only reads `\boxed{}`; `--rm-type dapo` only `Answer:`. Use the format-agnostic
  `rm.reward_func` (accepts either). Always pass `--label-key label` for the
  `{prompt, label}` DAPO jsonl, or `Sample.label` is `None` and reward reads 0.
- **Context length.** The 35B's DAPO chain-of-thought is ~14–17k tokens. An 8k
  response cap truncates ~95% of rollouts mid-reasoning → reward ~0. Use ≥24k
  (CP2 makes 24–32k feasible).
- **`--opd-teacher-load` path.** Point at the checkpoint **parent** dir (contains
  `latest_checkpointed_iteration.txt`), not an `iter_XXXXXXX` subdir. The subdir
  has no metadata → silent fallback to base → teacher == student → `opd_reverse_kl ≈ 0`.
  Sanity check: in the rollout log, `teacher_log_probs` should differ from
  `rollout/log_probs`.
- **Teacher must diverge.** A few RLVR steps at lr 1e-6 barely move the weights, so
  the teacher ≈ base and OPD is inert (`opd_reverse_kl ≈ 5e-4`). lr 1e-5 diverges it
  fast (`opd_reverse_kl ≈ 5e-2`). `--opd-kl-coef` cannot amplify a ~0 KL.
- **Memory.** `with_ref = (--use-kl-loss or --kl-coef≠0)`. Dropping `--use-kl-loss`
  keeps only student + teacher (2×35B) in memory; the teacher reverse-KL is the
  regularizer. Adding it loads a 3rd model and risks OOM.

## References
- ../README.md (served-teacher OPD), ../run-qwen3-8B-opd-megatron.sh (in-process teacher)
- https://thinkingmachines.ai/blog/on-policy-distillation/
