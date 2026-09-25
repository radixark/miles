# TMax Terminal-RL

GRPO on the 14.6k TMax terminal tasks
([`allenai/tmax-15k-open-instruct`](https://huggingface.co/datasets/allenai/tmax-15k-open-instruct)),
the Miles port of open-instruct's Terminal-RL recipe with the `swerl_vanillux_sandbox` environment.
Every episode runs in a fresh container from the task's prebuilt image; the only tool is `bash`;
the reward is the task's `tests/test.sh` verdict, uploaded only at submit time.

| File | Role |
|---|---|
| `run_tmax_qwen3_5_9b.py` | Launcher: `prepare` (download, extract, prompt JSONL, torch_dist conversion) and `execute` |
| `tmax_agent_function.py` | Agent loop for `agentic_tool_call.generate`, plus the over-sampling `abort` hook |
| `tmax_sandbox.py` | Per-episode container on a Docker-API socket (Podman), ported from open-instruct |
| `generate.py` | Reward function (`None` drops the group) and `agent/*` metrics |
| `tmax_prompts.py` | open-instruct's system prompt |

## Requirements

- A Docker-compatible API socket, normally `podman system service --time=0 unix:///run/podman/podman.sock`,
  whose containers can pull `docker.io/hamishi740/swerl-tmax-v3:*` (~400 MB each; log in to avoid rate limits,
  or pass `TMAX_REGISTRY_USERNAME` / `TMAX_REGISTRY_PASSWORD`).
- `pip install docker` in the training image.
- Inside a Beaker task: `BEAKER_ALLOW_SUBCONTAINERS=1`, and a `containers.conf` like open-instruct's
  `docker/podman/containers.conf` (host network, cgroups disabled).

## Run

```bash
python examples/tmax/run_tmax_qwen3_5_9b.py prepare --num-gpus-per-node 4
# synchronous, colocated
python examples/tmax/run_tmax_qwen3_5_9b.py execute --num-gpus-per-node 4
# fully async: 2 GPUs for SGLang, 2 for Megatron, groups at most 4 weight versions stale
python examples/tmax/run_tmax_qwen3_5_9b.py execute --num-gpus-per-node 4 --async-rollout-gpus 2
```

On 80 GB+ GPUs add `--tensor-model-parallel-size 1 --no-optimizer-cpu-offload`. For a smoke test use a small
`--num-prompts`, `--rollout-batch-size 4 --n-samples-per-prompt 4`, and `--no-filter-zero-std`: with 4 samples
Qwen3.5-9B almost always solves a task on every sample or on none, so the zero-variance filter would drop
every group and the step would never fill.

## Gotchas

- `--tito-model qwen35` does not set SGLang's parsers. Without `--sglang-reasoning-parser qwen3
  --sglang-tool-call-parser qwen3_coder` (the launcher passes both) tool calls come back as plain text and
  every episode ends after one turn.
- `allenai/tmax-*` checkpoints default to thinking off; Miles' `qwen35` template defaults to thinking on.
  Pass `--apply-chat-template-kwargs '{"enable_thinking": false}'` to match how they were trained.
- A step waits for its slowest episode. Long-thinking 40-turn episodes take ~20 minutes; cap them with
  `TMAX_EPISODE_TIMEOUT_S` or use `--async-rollout-gpus`.
- `--fully-async` selects its own rollout function, so `agent/*` metrics are only logged in sync mode; the
  per-episode `episode done:` log lines are written in both.
