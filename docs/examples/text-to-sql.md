---
title: "Text-to-SQL"
description: "Multi-turn text-to-SQL RL on BIRD-Platinum with a read-only SQLite tool, following River's recipe on Qwen3.6-35B-A3B."
# Generated from examples/text_to_sql/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
Multi-turn RL that teaches a model to answer questions over a SQLite database: it explores with up to five
`execute_sql_query` calls, then commits one query in a `<solution>` block, which is executed and compared with the
gold query's result. This is River's text-to-SQL recipe
([blog](https://river.ai/achieving-frontier-text-to-sql),
[code](https://github.com/riverai-org/river-recipes/tree/main/text-to-sql)) on miles, trained on BIRD-Platinum
(the ReViSQL authors' verified BIRD tasks) and evaluated on its validation split and Arcwise-Plat-SQL.

The prompt, the read-only SQLite tool, the grader and the reward terms are the recipe's own modules (the
`text_to_sql_recipe` package, Apache-2.0, pinned in the launcher); only its pure modules are imported, never the
River client. miles supplies the episode loop and training.

## Files

| File | Role |
|---|---|
| `sql_episode.py` | `generate`: the episode as a miles custom generate function. `reward_func`: grades the committed query and applies the recipe's reward terms. |
| `convert_dataset.py` | Writes the recipe's data release as miles jsonl (`train` 2,033, `validation` 398, `arcwise` 498). |
| `run_qwen3_6_35b_a3b_text_to_sql.py` | Qwen3.6-35B-A3B full-parameter launcher with the recipe's `qwen3.6-35b-a3b` rollout settings and MoE routing replay. |

## Run

```bash
python examples/text_to_sql/run_qwen3_6_35b_a3b_text_to_sql.py prepare
python examples/text_to_sql/run_qwen3_6_35b_a3b_text_to_sql.py train
```

`prepare` installs the recipe package, downloads and converts the checkpoint, downloads the ~34 GB data release, and
writes the jsonl.
`--output-dir` must sit on a real disk: besides the checkpoints, the grader keeps a private copy of every database
(~34 GB) there and streams ten very large gold results through temp indexes (~20 GB).
Pass `--wandb-project` (and `--wandb-team`) to log to wandb.

## Result

One 8xH200 node, 160 steps (5 epochs), about 8 minutes per step including evaluation every 5 steps:

| Step | validation | Arcwise |
|---|---|---|
| 0 | 61.3 | 52.6 |
| 15 | 84.7 | 79.7 |
| 100 | 89.7 | 88.4 |
| 160 | 90.2 | 90.0 |

The jump by step 15 is mostly the model learning to think shorter: at step 10, 13% (validation) and 22% (Arcwise)
of episodes still spend a turn's 3,072 tokens thinking and never answer; by step 15 under 2% do. Accuracy is flat after
step 100, when 70-85% of training groups score 16/16 and carry no GRPO advantage.

This run predates three changes that bring the episode closer to the recipe: the reward terms now come from the
recipe preset (`--text-to-sql-recipe`; the run applied no tool-format penalty), a turn the context budget cuts is no
longer graded, and a grade that times out is retried. Gold queries now share the SQL concurrency limit and a failed
one is retried by the next sample.

## Reward

| Outcome | Reward |
|---|---|
| correct | 1 |
| incorrect | 0 |
| no `<solution>` (including a turn cut before it, and any turn the context budget cuts) | -1 |
| unsafe query (anything but a read-only SELECT; ends the rollout) | -1 |

minus 0.1 for each evidence rule the trajectory skips (restate the evidence in the first two turns, check the
answer against it in the answer turn), minus 0.05 per invalid tool call up to 0.2. The terms are the recipe preset's
(`--text-to-sql-recipe`, default `qwen3.6-35b-a3b`), less the parts not ported below. Evaluation scores `correct`
only.

## Differences from the recipe

The recipe trains LoRA rank 32 at lr 5e-5; the launcher trains every parameter at lr 1e-6 with CPU-offloaded Adam.
Not ported yet:

- CISPO. With one optimizer step per rollout the importance ratio is ~1 and GRPO's clip never binds, so the two give
  nearly the same gradient; it matters once rollouts are asynchronous.
- The VeriEQL equivalence penalty (0.2 off a matching answer VeriEQL refutes).
- The curriculum: training hints after 16/16 failures, learnability weighting from epoch 2.
- The re-draft penalty and the repetition / collapse guards.
- Running every gold query before training. Gold results are computed on first use and cached per rollout process.
