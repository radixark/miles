# TOP recipes

Launchers for the true-on-policy runs reported in the TOP blog post, one per model. Each runs every arm of its
comparison with the same data, batch, reward, optimizer, speculation and evaluation.

The math recipes have two arms:

- `--arm top`: true on-policy training. Trainer and rollout log-probs are equal, so the rollout log-probs are the old
  policy and no correction is applied.
- `--arm standard`: Miles' standard trainer and serving kernels, with rollout routing replay (R3) and truncated
  importance sampling (TIS) correcting the mismatch.

The text-to-SQL recipe has five arms; its docstring tables them.

| Launcher | Model | Hardware | Reproduces |
|---|---|---|---|
| `run_deepseek_v4_flash_math.py` | DeepSeek-V4-Flash-Base | 64 H200 (8 x 8), colocated | DeepSeek-V4-Flash training, 100 steps; with `--timing`, the DeepSeek-V4-Flash whole-step timing pair |
| `run_glm5_2_math.py` | GLM-5.2-FP8 | 64 H200 (8 x 8), colocated | GLM-5.2 training, 100 steps; with `--timing`, the GLM-5.2 whole-step timing pair |
| `run_qwen3_6_text_to_sql.py` | Qwen3.6-35B-A3B | 64 H200: 4 trainer nodes, 4 rollout nodes | The async text-to-SQL arms, staleness up to 8 versions, steps 0-59 (translated, see below) |

`--timing` runs 6 steps at lr 0 with no evaluation, checkpoints or dynamic sampling, so both arms sample the same
policy every step. The post reports the medians of steps 1-5.

Each launcher's docstring states its recipe and its arms' layouts and kernels.

## Running

```bash
python examples/infra_features/true_on_policy/recipes/run_glm5_2_math.py prepare
MILES_SCRIPT_EXTERNAL_RAY=1 python examples/infra_features/true_on_policy/recipes/run_glm5_2_math.py train --arm top
MILES_SCRIPT_EXTERNAL_RAY=1 python examples/infra_features/true_on_policy/recipes/run_glm5_2_math.py train --arm standard
```

- **Before `train`:** start Ray on all eight nodes and put the HF checkpoint under `--model-dir`.
- **`prepare`:**
  - downloads DAPO-Math-17K (`zhuzilin/dapo-math-17k`) and the DAPO authors' AIME-2024 (`BytedTsinghua-SIA/AIME-2024`);
  - rewrites AIME-2024's answer instruction to Miles' `\boxed{}` wording (`math_data.py`);
  - for GLM-5.2, writes the eval config;
  - for the DeepSeek standard arm (`--arm standard`), converts the checkpoint to `torch_dist`.
- **`--optimizer-offload-dir`:** set it to node-local NVMe. Both math recipes offload the actor and the Adam state to
  disk between rollouts.
- **Text-to-SQL:** run `examples/text_to_sql/run_qwen3_6_35b_a3b_text_to_sql.py prepare` first. It installs the
  recipe package and prepares the checkpoint and data. Make a rollout node the Ray head: the rollout manager and the
  grader's SQLite scratch run there.

## Rewards

`math_rewards.py` holds the two rewards. Each is Miles' boxed grade (`--rm-type math`) with one shaping term:

- **DeepSeek-V4-Flash:** DAPO's soft overlong penalty, factor 0.5 over the last 4,096 tokens of the 14,000-token cap.
- **GLM-5.2:** finish or fail. A response cut at the 4,096-token cap scores 0. GLM-5.2 boxes answers while still
  thinking, so without this a guess boxed early in an unfinished response would score as correct.

Both report the plain grade as `acc`, which is what evaluation reports.

## Differences from the runs in the post

### Math recipes

The argv and environment match the runs' recorded launches flag for flag, except for:

- **Paths:** checkpoints, data, output and offload directories, now set by launcher flags.
- **Debugging and monitoring:** these flags, hooks and variables are dropped. None changes the training math.
  - `--dump-details` and `--use-miles-dashboard`.
  - A per-step memory hook (timing runs).
  - A before-step hook that saved Adam only with the final checkpoint (DeepSeek TOP).
  - `TOP_SURROGATE_CHECK=0`, which is the default anyway.
- **Flags and variables the current tree no longer has or no longer needs:**
  - `--true-on-policy-matched-tp1` and `--sglang-true-on-policy-matched-tp1`. TOP runs TP1 only.
  - `--sglang-true-on-policy-contract`. It is redundant: `--true-on-policy-contract` also sets the engines' contract,
    and the timing runs already left it out.
  - The trainer's `--moe-router-dtype fp32` under TOP. The TOP router computes at serving's precision, and the TOP
    model definitions drop the flag.
  - `MILES_OPTIMIZER_STATE_IN_RAM` (GLM-5.2), which the current tree does not read.
- **`NCCL_NVLS_ENABLE` (DeepSeek-V4-Flash TOP):** the run left it unset, which leaves NVLink SHARP to NCCL's default.
  Miles' launcher now sets it from the node's topology, to 1 on NVLink nodes. The other arms set it to 0, as the runs
  did.
- **Code versions:** the runs predate this tree. They used the TOP and standard code of their time, so a rerun on
  this tree runs the same recipe on newer kernels, not a byte-for-byte repeat.

### Text-to-SQL

This recipe is a translation. The runs used async features this tree replaces or lacks. Apart from those, its model
arguments, TOP flags and the rest of the argv match the five runs' recorded launches:

- **Staleness control:** the runs passed `--weight-staleness-policy wait`, which held a group older than 8 versions
  back until it fit. This tree replaces that policy with `--async-submit-within-staleness`: the producer submits a
  group only while the buffered and in-flight groups stay within 8 versions' worth. Both keep staleness at most 8,
  but the groups in flight, and so the staleness mix, can differ from the runs'.
- **Version check:** the runs re-scored stale tokens against snapshots of their sampling weights every 5 steps
  (`--version-check-*`). That check is diagnostic and absent here, so the figures' version-check panels have no
  counterpart.
- **Context cap:** the runs capped an episode at 29,696 tokens with `--rollout-max-context-len`. This tree's episode
  reads the same cap from `--max-seq-len`, which also lowers each request's `max_new_tokens` so that the episode fits.
- **Not needed or not present:**
  - `--top-allow-routing-replay` (TOP + R3 arm): this tree accepts R3 under TOP for the Qwen router.
  - The per-episode rollout log function: it was logging only, and this tree's episode module does not have it.
- **Run length and checkpoints:** every arm runs 60 steps, the range the figures plot. The two uncapped TOP arms were
  launched for 160 steps and saved every 40; the figures plot their steps 0-59. Every arm pins cosine decay to 160 steps, so the lr
  curve over steps 0-59 is the runs'.
- **Paths and `NCCL_NVLS_ENABLE`:** as for the math recipes. The runs left `NCCL_NVLS_ENABLE` unset.

## Not included

These runs from the post have no recipe here:

- **DeepSeek-V4.1-Flash.** The TOP arm's launch command was not kept. The standard arm needs a weight-sync quantizer
  with FP32 scales that this tree does not have.
- **Decode and prefill benchmarks.** They ran from a measurement harness tied to the cluster they were measured on.
