# Train GLM-4.7-Flash on TB2 against a `harbor-private` agent server (2-node example)

End-to-end recipe for a small (2-node) GLM-4.7-Flash agentic-async training
run that uses the `miles_agent_server.py` shipped on the [`harbor-private`
branch `shi/rebase-on-upstream-v0.7.0`][harbor-private-branch] as its
rollout backend.

This is the **training-side** companion to
[`examples/eval/terminal_bench_via_agent_server/`](../../../eval/terminal_bench_via_agent_server/),
which is the eval-side client for the same agent server. Both target the
same `POST /run` endpoint; this one drives it from
`run-glm47-flash-agentic-async.py` instead of a one-shot Python client.

## What gets exercised

```
trainer-side (rcli pod, this node)             agent-server (external host)
+--------------------+                         +-----------------------------+
|  Megatron actor    |  weights -> sglang ---->|                             |
|  RolloutManager    |                         |   miles_agent_server.py     |
|     ^              |  POST /run         ---->|     (harbor-private branch) |
|     | GRPO updates |                         |     spawns Docker -> agent  |
+-----+--------------+                         |     -> verifier -> reward   |
                                               +-----------------------------+
```

The training loop dispatches one `/run` per (prompt, sample) and receives a
verifier-scored reward back. The agent-server side is what we're really
validating: it must keep up with the trainer's batch cadence without
crashing under sustained load.

## Prerequisites

### 1. A running `miles_agent_server` from harbor-private

On a Tailscale-reachable host (we use `aws-agent-server` internally), check
out the [`harbor-private` branch][harbor-private-branch] and run the agent
server as documented in
[its README](https://github.com/radixark/harbor-private/blob/shi/rebase-on-upstream-v0.7.0/README.md).
The default port is 8080; the dashboard at 8081.

For training-style workloads `OPENAI_API_KEY=dummy` is correct (LLM
credentials flow per request, not from server env).

### 2. The 23-task TB2 variance jsonl on shared GPFS

Each line is `{"prompt": <task-instruction>, "metadata": {"instance_id":
<task-name>}}`. The 23 task names are the TB2 instances where vanilla
GLM-4.7-Flash passes 1-3 of 4 trials (the "improvable" middle band). Place
the file on the cluster's shared volume so it survives pod recreate:

```bash
# inside the head pod
/cluster_personal/job_workspaces/<your-job>/tb2_train.jsonl
```

### 3. A current miles checkout

Run from a current `miles` checkout — `origin/main` is fine. Older topic
branches may carry stale launcher flags or out-of-date torch / sglang
glue code; this example was validated against main.

```bash
cd /workspace/miles
git checkout origin/main
```

## Launch

The launcher hard-codes `--prompt-data /root/swe_train.jsonl`; symlink it
to the TB2 jsonl so the same launcher works without editing.

```bash
# inside the head pod
ln -sf /cluster_personal/job_workspaces/<your-job>/tb2_train.jsonl /root/swe_train.jsonl

cd /workspace/miles
bash examples/experimental/swe-agent-v2/train-tb2-via-harbor-private-server/launch.sh <run-tag>
```

Pass any short tag (`pr-smoke`, `260527-2n-v1`, ...) — it threads through
`--save-dir`, `--save-traces-dir`, and `--wandb-run-name` so multiple
attempts don't collide.

The full launcher invocation (from `launch.sh`, with optional env
overrides expanded):

```bash
python examples/experimental/swe-agent-v2/run-glm47-flash-agentic-async.py \
    --num-nodes 2 --train-num-nodes 1 --skip-prepare \
    --max-seq-len 65536 \
    --save-dir "${OUTPUT_ROOT}/GLM-4.7-Flash_2node_tb2_<run-tag>/" \
    --save-traces-dir "${OUTPUT_ROOT}/flash-2node-traces-<run-tag>/traces" \
    --rollout-batch-size 4 --n-samples-per-prompt 8 --global-batch-size 32 \
    --save-interval 5 \
    --agent-server-url "$AGENT_SERVER_URL" \
    --wandb-project glm47-flash-agentic-async \
    --wandb-run-name <run-tag>
    # optional, only added when the corresponding env var is set:
    # --router-external-host "$ROUTER_EXTERNAL_HOST"
    # --wandb-team           "$WANDB_TEAM"
```

`AGENT_SERVER_URL` defaults to `http://agent-server:8080`; set it to wherever
your `miles_agent_server` is reachable from the trainer pod.
`OUTPUT_ROOT` defaults to `/workspace` (writable storage on the trainer
pod that the rollout / save / traces dirs are created under).

If the agent server cannot reach the trainer at its default service
name (e.g. you're running the agent server on a different host/network),
set `ROUTER_EXTERNAL_HOST` to a hostname the agent server can dial back
to the trainer's session-server through. Otherwise leave it unset.

## Sanity checks (in order)

1. `Job 'raysubmit_<id>' submitted successfully` in the launch log within
   ~30s.
2. `ray job status <id> --address http://localhost:8265` reports
   `RUNNING` within ~2 min.
3. `wandb: 🚀 View run at https://wandb.ai/.../runs/<wandb-id>` appears in
   the log within ~3 min. **Record the `<wandb-id>`**; the S3 ckpt prefix
   is `<YYMMDD>-<wandb-id>` once `--save-interval` kicks in.
4. The agent-server dashboard at `:8081` starts seeing `mini-swe-agent`
   trials with the 23 task names from the TB2 variance band. If you see
   different task names, the dataset symlink didn't take.
5. First `rollout 0:` and `step 0:` log lines typically appear 30-60 min
   after launch (long because of sglang weight broadcast on cold pods);
   subsequent iters at 10-20 min/step.

If `/tmp/<launch-log>` stops growing but `ray job status` says `RUNNING`,
the shell's `tee` pipe died (e.g. ssh session ended) but training is
fine — use `ray job logs <id>` to read live progress instead.

## Tuning knobs

| Knob | Default here | Rationale |
|---|---|---|
| `--num-nodes 2 --train-num-nodes 1` | 1 trainer + 1 rollout | Minimum useful split; 1-node would co-locate and serialise the loop. |
| `--rollout-batch-size 4 --n-samples-per-prompt 8` | 32 trials/iter | Same shape as the 5-node baseline so iter cadence is comparable. |
| `--max-seq-len 65536` | 65536 | The agent server's `poll_steps` wrapper enforces this; raising it costs more truncation, lowering trips it more often. |
| `--sglang-mem-fraction-static 0.72` | 0.72 | Empirically avoids sglang OOM after weight-transfer broadcast. |
| `--save-interval 5` | every 5 iters | Checkpoints land under `--save-dir`; sync to S3 separately per your team's playbook. |

## Related

- [`examples/eval/terminal_bench_via_agent_server/`](../../../eval/terminal_bench_via_agent_server/) — same agent server, used from the **eval** side.
- [`harbor-private` branch `shi/rebase-on-upstream-v0.7.0`][harbor-private-branch] — the agent server code this example talks to.

[harbor-private-branch]: https://github.com/radixark/harbor-private/tree/shi/rebase-on-upstream-v0.7.0
