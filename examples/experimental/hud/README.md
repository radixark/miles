# Computer-use RL on HUD environments

Trains a VLM to operate a real GUI through [HUD](https://hud.ai) environments,
using HUD's published tasksets as-is. Nothing under `miles/` is modified.

**Result** — Qwen3-VL-4B-Instruct, GRPO, 8×H200, 32 episodes per step, 2048 in a
browser: over 16 steps the mean game score went 200 → 865 and the *worst*
episode of a batch went 0 → 540. The second number is the one that matters —
a lucky episode was already worth ~800 points before training, so what training
bought is reliability, not a higher ceiling. It plateaus around step 11 because
~89 key presses cannot produce much more than ~950 points; `hud_keys_per_turn`
moves that, more steps do not.

| seam | file |
|---|---|
| `--custom-generate-function-path` | [`rollout.py`](rollout.py) — multi-turn episode loop |
| `--custom-rm-path` | [`rollout.py`](rollout.py) `reward_func` |
| `--custom-config-path` | [`hud2048_config.yaml`](hud2048_config.yaml) |
| interaction env | [`hud_task_env.py`](hud_task_env.py) |

Anything named `hud2048_*` or `run_hud2048` is this one taskset; the rest is not
supposed to know which taskset it is running.

## The task row is the interface

A HUD taskset row describes a whole episode, so the adapter has no per-task code:

```json
{
  "mcp_config":    {"browser": {"headers": {"Mcp-Image": "hudevals/hud-browser:0.1.6"}}},
  "setup_tool":    {"name": "launch_app", "arguments": {"app_name": "2048"}},
  "evaluate_tool": {"name": "evaluate", "arguments": {"name": "game_2048_max_number", ...}}
}
```

**Format.** This reads the taskset row format the `hud-evals/*` datasets publish
— `mcp_config`, `setup_tool`, `evaluate_tool` — which is what their environment
images serve. HUD's current spec (v6) defines a task as a template inside an
environment package, reached over HUD's own control channel; adding a v6 leg
replaces `mcp_client.py`, the fetch in `make_hud_data.py` and the two tool calls
in `hud_task_env.py`, and leaves the rollout, reward and sandbox-lifecycle
plumbing — most of this example — unchanged.

`make_hud_data.py` puts the row in the prompt file's `metadata` column;
`--metadata-key` carries it to `Sample.metadata`; the env boots that image, makes
that setup call, and grades with that evaluate call. The model acts by writing
`Action: press("Left","Down")` / `click(x, y)` / `type("...")` / `done()` — plain
text rather than tool calls, because the loss is on the tokens the policy
generated, so an action has to *be* those tokens.

## Running it

```bash
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir /root/models/Qwen3-VL-4B-Instruct

# 2048-basic is a one-row taskset, so --repeat fills the file with independent
# episodes of it; a taskset with real rows uses --repeat 1.
python -m examples.experimental.hud.make_hud_data \
    --dataset hud-evals/2048-basic --repeat 256 --output /root/hud2048_train.jsonl

export DAYTONA_API_KEY=dtn_...                        # or ~/.config/daytona/api_key
export MILES_SCRIPT_OUTPUT_DIR=/persistent/hud2048    # checkpoints and rollout dumps
python -m pytest examples/experimental/hud/tests/ -q  # offline: no GPU, no network
python examples/experimental/hud/run_hud2048.py       # 8 GPUs, single node
```

Three cheaper rungs below a training run, in the order worth trying them:

| | covers | costs |
|---|---|---|
| `pytest examples/experimental/hud/tests/` | the translation between task row, model text and MCP calls | nothing |
| `python -m examples.experimental.hud.smoke_episode --dataset ...` | image boots, MCP answers, the row's setup and grade calls work, screenshots arrive | one sandbox |
| `MILES_SCRIPT_MODE=debug_rollout_only python .../run_hud2048.py` | the same with a real policy driving it, no training step | GPUs |

## Pointing it at another taskset

`--dataset` is the only thing that changes:

```bash
python -m examples.experimental.hud.make_hud_data --dataset hud-evals/OSWorld-Gold --output ...
```

2048 is what this is verified on, end to end, through training. Other tasksets
(`hud-evals/OSWorld-Gold`, `SheetBench-50`) need no adapter code by
construction, but nothing beyond that is tested: their images are far heavier
than a browser one, and a desktop task will exercise `click` / `type` where 2048
only ever presses keys. `smoke_episode.py` is the check to run first — it costs
one sandbox and no GPUs.

The action vocabulary is the one thing that is ours rather than the row's, so it
is also where a new taskset needs work: verbs live in `hud_task_env.py` `_do`,
and the description the model is taught them by is `DSL_SUFFIX` in
`make_hud_data.py`. Keys are passed through as written (`press("ctrl+c")` is a
chord, `press("Left","Down")` a sequence), but 2048's prompt only ever advertises
the arrows.

Two knobs are worth understanding before you turn them up, because both trade
against rollout speed: every turn re-prefills the whole screenshot history
through the vision encoder, so `max_turns` and `hud_screenshot_width` cost more
than they look like they do — at 960px × 14 turns a cache-missing episode
re-prefilled ~5.5k tokens at 14 tok/s, slow enough to stall a batch behind one
engine. `hud_inference_timeout_s` bounds that failure. The rest of the config is
annotated in [`hud2048_config.yaml`](hud2048_config.yaml).
