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

## Run on GB200 / GB300 (CUDA 13, Blackwell) — `phase2_gb200.sh`

The recipe above targets a single **8×H200** node. Blackwell nodes (GB200/GB300)
have **4 GPUs/node**, so `world = 8` becomes **2 nodes × 4 GPUs** — same parallel
dims (`TP2 PP1 CP2 EP8 ETP1`, DP4), only the node tiling changes. `phase2_gb200.sh`
is the GB200 variant of `phase2_opd_selfdistill.sh`; the deltas (all validated on
2× GB200, reproducing the base eval `0.84` / `~14k`) are:

- **Tiling** — `--actor-num-nodes 2 --num-gpus-per-node 4` (override via
  `ACTOR_NUM_NODES` / `GPUS_PER_NODE`). Pin both nodes to one NVLink (MNNVL) domain
  so the EP8 all-to-all stays on the NVLink fabric.
- **sglang backends** (cf. `scripts/run_qwen3_5_35b_a3b_mtp_cp2_ep8.py`) —
  `--sglang-moe-runner-backend flashinfer_cutlass`, `--sglang-attention-backend
  trtllm_mha`, and `--moe-token-dispatcher-type flex`. The default triton fused-MoE
  mis-shards routed experts on the megatron→sglang weight sync
  (`fused_moe_triton ... _load_w13`: `tensor a (64) vs b (2048)`), and FA3 is SM≤90
  only (Blackwell is SM 10.x).
- **NCCL** — `NCCL_NVLS_ENABLE=0` (multi-node Blackwell NVLS bind fails
  `ncclCommInitRank`); keep `NCCL_MNNVL_ENABLE=1`.
- **k8s** — if a `prometheus` Service exists in the namespace, set
  `PROMETHEUS_PORT=9090` (kube injects a `tcp://…:9090` URL that breaks miles'
  `int(PROMETHEUS_PORT)`).

`phase2_gb200.sh` already sets the sglang/MoE backends and folds
`NCCL_NVLS_ENABLE=0` + `PROMETHEUS_PORT=9090` into the Ray runtime env. Run it on the
Ray head in the CUDA-13 ARM64 miles image, with `MILES_DIR` pointing at the repo:

```bash
ACTOR_NUM_NODES=2 GPUS_PER_NODE=4 MILES_DIR=/workspace/miles \
MODEL_DIR=... DATA_DIR=... TEACHER_LOAD=/persistent/ckpt-teacher OUTPUT_DIR=/persistent/ckpt-opd-pure \
  bash phase2_gb200.sh          # MODE=pure (default) | MODE=grounded
```

## Run Phase 2 only (skip Phase 1)

If you already have a teacher checkpoint, skip Phase 1 and run Phase 2 directly —
point `--opd-teacher-load` (`TEACHER_LOAD`) at the teacher's **torch_dist parent
dir** (the one containing `latest_checkpointed_iteration.txt`). You still need the
base model (`--hf-checkpoint` + the `--ref-load` torch_dist) and the data split, but
no Phase-1 run.

If your teacher is in **HuggingFace** format, convert it to torch_dist first with
`convert_gb200.sh` (a thin wrapper over `tools/convert_hf_to_torch_dist.py` carrying
the Qwen3.5 `MODEL_ARGS`):

```bash
# teacher: HF safetensors -> Megatron torch_dist parent dir
bash convert_gb200.sh /path/to/teacher-hf /persistent/ckpt-teacher
# (and the base, if you don't have Qwen3.5-35B-A3B_torch_dist yet)
bash convert_gb200.sh ${MODEL_DIR}/Qwen3.5-35B-A3B ${MODEL_DIR}/Qwen3.5-35B-A3B_torch_dist

TEACHER_LOAD=/persistent/ckpt-teacher MODEL_DIR=... DATA_DIR=... \
  bash phase2_gb200.sh          # or phase2_opd_selfdistill.sh on 8×H200
```

> **Teacher expert layout.** The public `Qwen/Qwen3.5-35B-A3B` ships *fused* experts
> (`mlp.experts.gate_up_proj`); a teacher round-tripped through
> `convert_torch_dist_to_hf` may ship *unfused* per-expert weights
> (`mlp.experts.{i}.gate_proj.weight`). `miles_plugins/mbridge/qwen3_5.py` now
> autodetects both for the main layers (mirroring the existing MTP-expert
> autodetect), so either layout converts without manual re-fusing.

## Results (DAPO-math, held-out 512, eval @ 24k cap, temp 0.6)

**Phase 1 — RLVR teacher** (lr 1e-5):

| step | eval/dapo_heldout | eval response length |
|------|-------------------|----------------------|
| 0 (base) | 0.828 | 14,070 |
| 5        | **0.887** | **6,248** |

The teacher becomes both more accurate **and** ~2× more concise. This Phase-1
teacher checkpoint is published at
[**cm00cm/Qwen3.5-35B-A3B-DAPO-RLVR-teacher**](https://huggingface.co/cm00cm/Qwen3.5-35B-A3B-DAPO-RLVR-teacher)
(weights only) and can be used directly as the Phase-2 teacher via
`--opd-teacher-load` after `convert_hf_to_torch_dist.py`.

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

**Phase 2 — grounded OPD** (correctness reward + teacher reverse-KL):

| step | rollout/raw_reward | train length | opd_reverse_kl |
|------|--------------------|--------------|----------------|
| 1 | 0.637 | 18,778 | 0.045 |
| 2 | **0.910** | **7,665** | 0.014 |

With the correctness reward kept, `rollout/raw_reward` (== accuracy) climbs while
the student simultaneously adopts the teacher's concise responses (18.8k → 7.7k).
The shrinking `opd_reverse_kl` (0.045 → 0.014) shows the student converging onto
the teacher. (At lr 1e-5 the RLVR reward alone also drives accuracy up — Phase 1
is the controlled view of that — so grounded OPD's `raw_reward` climb reflects
RLVR + the teacher pull combined; the pure-OPD run above isolates OPD's effect.)

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
- Phase-1 teacher checkpoint: https://huggingface.co/cm00cm/Qwen3.5-35B-A3B-DAPO-RLVR-teacher
- ../README.md (served-teacher OPD), ../run-qwen3-8B-opd-megatron.sh (in-process teacher)
- https://thinkingmachines.ai/blog/on-policy-distillation/
