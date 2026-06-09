# Terminal-Bench via miles_agent_server

Evaluate a DeepSeek-V4-Pro (or any OpenAI-compatible) endpoint on the full
89-task **Terminal-Bench 2.0** benchmark by driving an already-running
`miles_agent_server` over its `/run` HTTP endpoint.

This example is purely client-side — no training, no Ray, no Megatron. It
just submits `POST /run` requests concurrently and reads back rewards. The
heavy lifting (Docker container per task, mini-swe-agent inside it,
verifier scoring) all happens on the agent-server host.

Rather than driving Terminal-Bench's own `tb` CLI as a local subprocess, we
reuse a shared, long-running agent server, which is much closer to how RL
training-loop rollouts interact with the same backend.

## Prerequisites

1. **A running `miles_agent_server`** from the
   [`shi/rebase-on-upstream-v0.7.0` branch of harbor-private][branch], with
   the server's `$HARBOR_TASKS_DIR` populated with the 89 Terminal-Bench 2.0
   tasks. The server's README explains setup; the short version is:

   ```bash
   harbor datasets download terminal-bench -o ./harbor_tasks --export

   export OPENAI_API_KEY=dummy
   export OPENAI_API_BASE=dummy
   HARBOR_TASKS_DIR=$PWD/harbor_tasks \
   python miles_agent_server.py \
       --port 8080 --dashboard-port 8081 \
       --max-concurrent 32 \
       --trials-dir ./trials/$(date +%Y%m%d_%H%M%S)
   ```

   `OPENAI_API_KEY=dummy` is correct here: training-style workloads (this
   example included) send the real LLM credential per request, not via the
   server's env.

2. **Network reachability** between this script and the agent server's
   `--port`. From a kubernetes training pod, that typically means a
   Tailscale-egress service; from a laptop you may want a local Tailnet
   FQDN or `localhost` if the server runs on the same host.

3. **DeepSeek API key** exported in your shell:

   ```bash
   export DEEPSEEK_API_KEY=<your-key>
   ```

   The script reads the key only from the environment — it is never read
   from a file, never logged, and never put on the command line. The
   default env var is `DEEPSEEK_API_KEY`; override with `--api-key-env` if
   you keep your key under a different variable.

4. **`registry.json`** from the harbor-private checkout (so the script knows
   what the 89 task names are). Pass its path with `--registry-path`.

## Run

```bash
python eval_tb_deepseek_v4_pro.py \
    --server-url   http://<agent-server-host>:8080 \
    --registry-path <path-to-harbor-private>/registry.json \
    --n-trials-per-task 4 \
    --max-concurrent 16
```

This dispatches `89 × 4 = 356` requests, capped at 16 in flight at a time,
and writes:

- `tb_deepseek_v4_pro_results.jsonl` — streamed per-trial rows as they
  complete (so you can `tail -f` while it runs).
- `tb_deepseek_v4_pro_results.parquet` — aggregated table, written once at
  the end via `polars`.

Each row has: `instance_id`, `trial_idx`, `status_code`, `elapsed_sec`,
`reward`, `exit_status`, plus the server's `agent_metrics` / `eval_report`
nested objects (stripped from the parquet for typing; full content remains
in the jsonl).

## Tuning notes

- `--max-concurrent 16` is conservative. The agent server can typically
  sustain 32–128 simultaneous mini-swe-agent Docker trials before sglang or
  Docker-daemon side resources push back. Watch the server's
  `/api/sessions` dashboard during a run; bump up if you see headroom.
- `--agent-name terminus-2` switches to the host-process agent. In that
  case the agent server itself must have `OPENAI_API_KEY` set to a real
  value, because the host agent's LiteLLM client reads it from the server
  process env (Docker agents instead receive the credential per request).
- `--max-seq-len 65536` matches the harbor-private branch's
  poll-steps wrapper. Lower values will trigger early
  `SequenceLengthLimitExceeded` exits more often, which is sometimes
  desirable for shorter-context evals.
- `--per-request-timeout-sec 3600` is the client-side cap. The server's
  default agent/verifier timeouts are independent (1800s each) and will
  return a graceful `AgentTimeoutError`/`VerifierTimeoutError` exit status
  in the response body before the client timeout would fire — so client
  timeouts only matter as a safety net for total network-level hangs.

## Quick aggregations

```python
import polars as pl
df = pl.read_parquet("tb_deepseek_v4_pro_results.parquet")

# Overall pass rate across (89 tasks * 4 trials = 356 trials)
print(df.filter(pl.col("reward").is_not_null()).select(
    pass_rate=(pl.col("reward") == 1.0).mean()
))

# Per-task pass rate
print(df.group_by("instance_id").agg(
    n=pl.col("reward").is_not_null().sum(),
    n_pass=(pl.col("reward") == 1.0).sum(),
).sort("n_pass", descending=True))
```

[branch]: https://github.com/radixark/harbor-private/tree/shi/rebase-on-upstream-v0.7.0
