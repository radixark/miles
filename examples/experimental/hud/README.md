# Computer-use RL on HUD environments

A model that sees a screen and works the keyboard and mouse gets better at
using a computer by practising: it attempts a task in a real GUI thousands of
times, and each episode's own grade tells it which attempts were good. This
example runs that loop on [HUD](https://hud.ai) v6 environments — any HUD env
package whose tasks are workable through a screen trains on the same files.

The split: HUD owns the episode — it boots the environment, runs the agent
loop, and grades the result — while Miles owns training, and the two meet at
the token ids the policy emitted. Nothing under `miles/` is modified.

The worked example throughout is
[2048](https://en.wikipedia.org/wiki/2048_%28video_game%29) — the sliding-tile
puzzle where equal tiles merge and double, and every merge scores — played in
a real browser from HUD's public starter (`hud init --preset browser`, no API
key; it ships 2048, a todo app, and graded task templates for both).
Everything named `hud2048_*` or `run_hud2048` is that recipe; the rest does
not know which taskset it is running.

| seam | file |
|---|---|
| `--custom-generate-function-path` | [`rollout.py`](rollout.py) — HUD run → one Miles training sample |
| `--custom-rm-path` | [`rollout.py`](rollout.py) `reward_func` — the task template's own grade |
| `--custom-config-path` | [`hud2048_config.yaml`](hud2048_config.yaml) |
| the agent | [`agent.py`](agent.py) + [`computer_tool.py`](computer_tool.py) |
| sglang ↔ HUD token contract | [`sglang_compat.py`](sglang_compat.py) |

## How the pieces divide

A HUD v6 task is a template inside an environment package — its first `yield`
is the prompt, its second is the reward — and `task.run(agent, runtime=...)`
boots the environment (one Daytona sandbox per episode, deleted on exit), runs
the agent, and grades. On top of that this example adds three pieces, each
reasoned about in its own file: a screen for self-hosted models
([`computer_tool.py`](computer_tool.py) — HUD only ships computer-use tools
for provider-native protocols), a token-id bridge to stock sglang
([`sglang_compat.py`](sglang_compat.py)), and the stitch that turns one
episode's per-turn token ids into one verified training sequence
([`rollout.py`](rollout.py)).

Reward shaping is choosing which task template to train on — the tradeoff, and
why this recipe trains on `reach_score` with a retuned target, is annotated in
[`hud2048_config.yaml`](hud2048_config.yaml).

## Running it

```bash
pip install hud daytona asyncssh asyncvnc    # the harness's extra deps
hud init v6browser --preset browser          # downloads the starter into ./v6browser

# Give the page the whole screen. As scaffolded, the browser's own chrome takes
# both the pixels the board needs and the clicks meant for it, so the policy
# sees a cropped board and focuses the sidebar; kiosk mode removes it.
sed -i 's/1280x800x24/1280x1200x24/; s/--window-size=1280,800/--window-size=1280,1200/;
        s/"--no-first-run",/"--no-first-run", "--kiosk", "--disable-infobars",/' v6browser/env.py

hf download Qwen/Qwen3-VL-4B-Instruct --local-dir /root/models/Qwen3-VL-4B-Instruct
python -m examples.experimental.hud.make_hud_data \
    --env-dir /root/v6browser --task-ids 2048-score-5000 \
    --args-json '{"target_score": 1024}' --repeat 256 --output /root/hud2048_train.jsonl

mkdir -p ~/.config/daytona && echo dtn_... > ~/.config/daytona/api_key   # sandboxes
export MILES_SCRIPT_OUTPUT_DIR=/persistent/hud2048    # checkpoints and rollout dumps
python -m pytest examples/experimental/hud/tests/ -q  # offline: no GPU, no network
MILES_SCRIPT_MODE=smoke python examples/experimental/hud/run_hud2048.py   # 2 episodes
python examples/experimental/hud/run_hud2048.py       # 8 GPUs, single node
```

Two episodes is the right size for the middle rung: the bugs a rollout finds
are per-episode, and what a wider rollout batch *cannot* show you — reward
spread inside a group, weight sync, hours-long stability — needs real
training. So go from `MODE=smoke` to a short real run, not to a bigger smoke.

What to expect from the full run: 20 GRPO steps take about five hours on
8×H200 (~15 min/step, 32 episodes each). Mean episode reward roughly doubles
(0.037 → ~0.07 against the retuned `target_score` of 1024 — a mean game score
of about 38 → 72), by the last steps
every episode in the batch scores, and no group loses its reward variance —
the model learns to reliably reach the board and spend its full move budget
rather than to play brilliantly, which is what a linear score reward buys
first.

The env snapshot is built by Daytona cloud-side from the package's
`Dockerfile.hud` on first use and rebuilt whenever the package's content hash
changes — a laptop and a training node each build their own unless the
directory is byte-identical (mind stray `__pycache__`).

## Pointing it at another taskset

`--env-dir` and `--task-ids` are the change; any HUD env package whose tasks
are workable through a screen runs on the same files, and the browser starter's
todo app already does. Sizing lives in `hud2048_config.yaml`, annotated with
what each number trades against.

Keep `--sglang-tool-call-parser` matched to the model: without it, tool calls
stay unparsed text and every episode ends at turn one.
